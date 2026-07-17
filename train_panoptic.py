import os
import json
import random
import argparse
import logging
import sys
import shutil
import cv2
import numpy as np
from PIL import Image
# PanopticAPI imports
from panopticapi.utils import id2rgb
from pycocotools import mask as mask_util
# Detectron2 imports
# Add local detectron2 to path
sys.path.insert(0, os.path.abspath('./detectron2'))
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_panoptic_separated
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOPanopticEvaluator, SemSegEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger

# --- Setup Logger at global scope ---
logger = logging.getLogger("detectron2")

def prepare_panoptic_data(args):
    """
    Orchestrates the entire data preparation process using paths from the args object.
    """
    logger.info("--- Starting Full Dataset Preparation ---")
    
    with open(args.source_train_instance_json, 'r') as f: training_data = json.load(f)
    if os.path.exists(args.source_val_instance_json):
        with open(args.source_val_instance_json, 'r') as f: validation_data = json.load(f)
    else:
        logger.warning(f"Validation JSON {args.source_val_instance_json} not found. Using empty validation data.")
        validation_data = {"images": [], "annotations": [], "categories": training_data['categories'], "info": training_data.get("info", {}), "licenses": training_data.get("licenses", [])}
    with open(args.types_json, 'r') as f: types_map = json.load(f)

    # Clean up old generated directories
    for d in [args.train_panoptic_root, args.val_panoptic_root, args.train_sem_seg_root, args.val_sem_seg_root]:
        if os.path.exists(d): shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    # --- Create category mappings ---
    valid_categories = training_data['categories']
    cat_id_to_name = {cat['id']: cat['name'] for cat in valid_categories}
    
    thing_classes = [cat for cat in valid_categories if types_map.get(cat['name']) != 'stuff']
    stuff_classes = [cat for cat in valid_categories if types_map.get(cat['name']) == 'stuff']
    void_class = 255  # Standard ignore value for void areas
    
    # --- Create the three required ID mappings ---
    # 1. For instance JSONs (things only, 1-based, as expected by COCO loader)
    instance_thing_id_map = {cat['id']: i + 1 for i, cat in enumerate(thing_classes)}
    # 2. For semantic training masks (stuff only, 1-based as per docstring)
    semantic_stuff_id_map = {cat['id']: i + 1 for i, cat in enumerate(stuff_classes)}
    # 3. For final panoptic files (global, 1-based for evaluation)
    num_thing_classes = len(thing_classes)
    global_panoptic_map = {cat['id']: i + 1 for i, cat in enumerate(thing_classes)}
    global_panoptic_map.update({cat['id']: num_thing_classes + i + 1 for i, cat in enumerate(stuff_classes)})

    annotations_by_image = {}
    for ann in training_data['annotations']:
        if ann.get("bbox") and ann['bbox'][2] > 0 and ann['bbox'][3] > 0:
            annotations_by_image.setdefault(ann['image_id'], []).append(ann)

    # --- FIX: index validation_data's own annotations separately, then merge ---
    val_annotations_by_image = {}
    for ann in validation_data.get('annotations', []):
        if ann.get("bbox") and ann['bbox'][2] > 0 and ann['bbox'][3] > 0:
            val_annotations_by_image.setdefault(ann['image_id'], []).append(ann)
    
    for img_id, anns in val_annotations_by_image.items():
        annotations_by_image.setdefault(img_id, []).extend(anns)
    # --- end fix ---

    train_images = [img for img in training_data['images'] if img['id'] in annotations_by_image]
    val_images = [img for img in validation_data['images'] if img['id'] in annotations_by_image]
    logger.info(f"Split of the dataset: {len(train_images)} train images, {len(val_images)} validation images.")
    
    # --- Generate all required files for both splits ---
    def process_split(images, instance_output_path, panoptic_output_path, panoptic_mask_dir, semantic_mask_dir):
        img_ids = {img['id'] for img in images}
        
        # 1. Create Instance JSON for this split
        instance_anns = []
        for img_id in img_ids:
            for ann in annotations_by_image.get(img_id, []):
                if ann['category_id'] in instance_thing_id_map:
                    new_ann = ann.copy()
                    new_ann['category_id'] = instance_thing_id_map[ann['category_id']]
                    new_ann['iscrowd'] = ann.get('iscrowd', 0) 
                    instance_anns.append(new_ann)
        
        remapped_thing_categories = [{'id': i + 1, 'name': cat['name']} for i, cat in enumerate(thing_classes)]
        with open(instance_output_path, 'w') as f:
            json.dump({"info": training_data.get("info", {}), "licenses": training_data.get("licenses", []), "categories": remapped_thing_categories, "images": images, "annotations": instance_anns}, f)
        
        # 2. Create Panoptic JSON and Masks for evaluation
        panoptic_anns = []
        remapped_panoptic_categories = [{'id': global_panoptic_map[cat['id']], 'name': cat['name'], 'isthing': 1 if cat in thing_classes else 0} for cat in valid_categories]
        remapped_panoptic_categories.sort(key=lambda x: x['id'])

        for img in images:
            panoptic_mask = np.zeros((img['height'], img['width']), dtype=np.int32)
            segments_info = []
            
            for ann in annotations_by_image.get(img['id'], []):
                original_cat_id = ann['category_id']
                global_cat_id = global_panoptic_map[original_cat_id]
                is_thing = types_map.get(cat_id_to_name[original_cat_id]) != 'stuff'
                
                if is_thing:
                    segment_id = ann['id'] * 1000 + global_cat_id
                else:
                    segment_id = global_cat_id

                # First, get or calculate the area
                if 'area' in ann:
                    area = ann['area']
                else:
                    # If area is not pre-computed, calculate it from the segmentation mask
                    area = int(mask_util.area(ann['segmentation']))

                # Now, create the complete dictionary for segments_info
                segments_info.append({
                    "id": segment_id, 
                    "category_id": global_cat_id,
                    "iscrowd": ann.get("iscrowd", 0),
                    "bbox": ann['bbox'], # Also ensure bbox is included
                    "area": area         # Add the required area key
                })
                
                if isinstance(ann['segmentation'], list):
                    for seg in ann['segmentation']:
                        poly = np.array(seg).reshape(-1, 2).astype(np.int32)
                        cv2.fillPoly(panoptic_mask, [poly], segment_id)
                else:
                    rle = ann['segmentation']
                    decoded_mask = mask_util.decode(rle)
                    panoptic_mask[decoded_mask == 1] = segment_id
            
            panoptic_anns.append({"image_id": img['id'], "file_name": os.path.splitext(img['file_name'])[0] + ".png", "segments_info": segments_info})
            Image.fromarray(id2rgb(panoptic_mask)).save(os.path.join(panoptic_mask_dir, os.path.splitext(img['file_name'])[0] + ".png"))
            
        with open(panoptic_output_path, 'w') as f:
            json.dump({"info": training_data.get("info", {}), "licenses": training_data.get("licenses", []), "categories": remapped_panoptic_categories, "annotations": panoptic_anns}, f)
        
        # 3. Generate Semantic Masks for training
        for img in images:
            sem_seg_mask = np.full((img['height'], img['width']), void_class, dtype=np.uint8) # Start with a temporary value

            # First, paint all "stuff" areas with their 1-based contiguous IDs
            for ann in annotations_by_image.get(img['id'], []):
                if ann['category_id'] in semantic_stuff_id_map:
                    contiguous_id = semantic_stuff_id_map[ann['category_id']]
                    if isinstance(ann['segmentation'], list):
                        for seg in ann['segmentation']:
                            poly = np.array(seg).reshape(-1, 2).astype(np.int32)
                            cv2.fillPoly(sem_seg_mask, [poly], contiguous_id)
                    else:
                        rle = ann['segmentation']
                        decoded_mask = mask_util.decode(rle)
                        sem_seg_mask[decoded_mask == 1] = contiguous_id
            
            # Now, paint all "thing" areas on top with the "things" class (0)
            for ann in annotations_by_image.get(img['id'], []):
                if ann['category_id'] in instance_thing_id_map:
                    if isinstance(ann['segmentation'], list):
                        for seg in ann['segmentation']:
                            poly = np.array(seg).reshape(-1, 2).astype(np.int32)
                            cv2.fillPoly(sem_seg_mask, [poly], 0)  # Use 0 for "things" class
                    else:
                        rle = ann['segmentation']
                        decoded_mask = mask_util.decode(rle)
                        sem_seg_mask[decoded_mask == 1] = 0 # "Use 0 for "things" class
            
            # # Any area not covered by stuff or things should also be ignored
            # sem_seg_mask[sem_seg_mask == 255] = ignore_value
            Image.fromarray(sem_seg_mask).save(os.path.join(semantic_mask_dir, os.path.splitext(img['file_name'])[0] + ".png"))

    process_split(train_images, args.train_instance_split_json, args.train_panoptic_split_json, args.train_panoptic_root, args.train_sem_seg_root)
    process_split(val_images, args.val_instance_split_json, args.val_panoptic_split_json, args.val_panoptic_root, args.val_sem_seg_root)

    logger.info("--- Finished Full Dataset Preparation ---")
    return instance_thing_id_map, semantic_stuff_id_map, global_panoptic_map, cat_id_to_name, void_class

def setup(args):
    """ Main setup function """
    setup_logger(output=args.output_dir)

    instance_thing_id_map, semantic_stuff_id_map, global_panoptic_map, cat_id_to_name, void_class = prepare_panoptic_data(args)

    # Register datasets
    train_dataset_name = f"{args.dataset_name}_panoptic_train"
    val_dataset_name = f"{args.dataset_name}_panoptic_val"

    # ─────────────────────────────────────────────────────────────────
    # FIX: Build metadata mappings that are CONSISTENT with global_panoptic_map.
    #
    # The panoptic JSON/masks generated by prepare_panoptic_data() use
    # global_panoptic_map IDs: things = 1..N, stuff = N+1..N+M.
    #
    # Detectron2's COCOPanopticEvaluator needs thing_dataset_id_to_contiguous_id
    # and stuff_dataset_id_to_contiguous_id to translate the MODEL's internal
    # contiguous prediction ids back into the SAME id space used by the
    # panoptic ground truth (i.e. global_panoptic_map ids), so they can be
    # compared correctly.
    #
    # - Model's thing head predicts contiguous ids 0..N-1 (one per thing class,
    #   in the order of instance_thing_id_map).
    #   -> dataset id (global_panoptic_map space) for the i-th thing class is i+1.
    #   -> thing_dataset_id_to_contiguous_id = {dataset_id: contiguous_id}
    #                                         = {i+1: i for i in range(N)}
    #
    # - Model's stuff head predicts contiguous ids 0..M (0 = "thing" placeholder,
    #   1..M = stuff classes, in the order of semantic_stuff_id_map).
    #   -> dataset id (global_panoptic_map space) for the i-th stuff class is N+i+1.
    #   -> stuff_dataset_id_to_contiguous_id = {dataset_id: contiguous_id}
    #                                         = {N+i+1: i+1 for i in range(M)}
    #
    # This replaces the OLD (buggy) version which kept stuff_dataset_id_to_contiguous_id
    # keyed by the ORIGINAL raw category ids instead of global_panoptic_map ids -
    # causing a silent mismatch between predicted and ground-truth id spaces
    # (this was the root cause of invalid SQ>100 and all-zero Stuff results).
    # ─────────────────────────────────────────────────────────────────
    num_things = len(instance_thing_id_map)
    num_stuff = len(semantic_stuff_id_map)

    thing_dataset_id_to_contiguous_id = {i + 1: i for i in range(num_things)}
    stuff_dataset_id_to_contiguous_id = {num_things + i + 1: i + 1 for i in range(num_stuff)}
    # Add a harmless placeholder so SemSegEvaluator does not crash when it
    # encounters the "thing" pixel marker (contiguous id 0) while encoding
    # predictions - this does not affect the real per-class IoU/accuracy
    # numbers, which come from a separate confusion-matrix accumulation step.
    stuff_dataset_id_to_contiguous_id[0] = 0

    metadata = {
        "thing_classes": [cat_id_to_name[cat_id] for cat_id in instance_thing_id_map.keys()],
        "stuff_classes": [cat_id_to_name[cat_id] for cat_id in semantic_stuff_id_map.keys()],
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
    }

    logger.info(f"thing_dataset_id_to_contiguous_id: {thing_dataset_id_to_contiguous_id}")
    logger.info(f"stuff_dataset_id_to_contiguous_id: {stuff_dataset_id_to_contiguous_id}")

    register_coco_panoptic_separated(
        name=train_dataset_name, metadata=metadata,
        image_root=os.path.abspath(os.path.join(args.img_train_dir, f"{args.dataset_name}_train")),
        panoptic_root=os.path.abspath(args.train_panoptic_root),
        panoptic_json=os.path.abspath(args.train_panoptic_split_json),
        sem_seg_root=os.path.abspath(args.train_sem_seg_root),
        instances_json=os.path.abspath(args.train_instance_split_json)
    )
    register_coco_panoptic_separated(
        name=val_dataset_name, metadata=metadata,
        image_root=os.path.abspath(os.path.join(args.img_train_dir, f"{args.dataset_name}_val")),
        panoptic_root=os.path.abspath(args.val_panoptic_root),
        panoptic_json=os.path.abspath(args.val_panoptic_split_json),
        sem_seg_root=os.path.abspath(args.val_sem_seg_root),
        instances_json=os.path.abspath(args.val_instance_split_json)
    )

    # Configure model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.model_name + ".yaml"))
    
    cfg.DATASETS.TRAIN = (train_dataset_name + "_separated",)
    cfg.DATASETS.TEST = (val_dataset_name + "_separated",)

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(instance_thing_id_map)
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(semantic_stuff_id_map) + 1 # +1 for "things" ignore class
    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE = void_class # Standard ignore value

    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.MAX_ITER = args.iter
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.OUTPUT_DIR = args.output_dir

    cfg.MODEL.DEVICE='cuda'
    
    return cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detectron2 Panoptic Segmentation Training and Evaluation")
    parser.add_argument("--dataset-name", type=str, default="snow_classes", help="The base name of the dataset files")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation on the validation set")
    parser.add_argument("--iter", type=int, default=5000, help="Number of training iterations")
    args = parser.parse_args()
    
    # --- Centralized Path and Model Definitions ---
    base_dir = ".." 
    args.img_train_dir = os.path.join(base_dir, "img_train")
    
    # Input file paths
    args.source_train_instance_json = os.path.join(args.img_train_dir, f"{args.dataset_name}_train.json")
    args.source_val_instance_json = os.path.join(args.img_train_dir, f"{args.dataset_name}_val.json")
    args.types_json = os.path.join(args.img_train_dir, f"{args.dataset_name}_types.json")
    
    # Generated instance splits
    args.train_instance_split_json = os.path.join(args.img_train_dir, f"{args.dataset_name}_instance_train_split.json")
    args.val_instance_split_json = os.path.join(args.img_train_dir, f"{args.dataset_name}_instance_val_split.json")
    
    # Generated panoptic files
    args.train_panoptic_split_json = os.path.join(args.img_train_dir, f"{args.dataset_name}_panoptic_train.json")
    args.val_panoptic_split_json = os.path.join(args.img_train_dir, f"{args.dataset_name}_panoptic_val.json")
    args.train_panoptic_root = os.path.join(args.img_train_dir, f"{args.dataset_name}_train_panoptic_seg")
    args.val_panoptic_root = os.path.join(args.img_train_dir, f"{args.dataset_name}_val_panoptic_seg")
    
    # Generated semantic segmentation files
    args.train_sem_seg_root = os.path.join(args.img_train_dir, f"{args.dataset_name}_train_semantic_seg")
    args.val_sem_seg_root = os.path.join(args.img_train_dir, f"{args.dataset_name}_val_semantic_seg")
    
    # Model and output directory
    args.model_name = "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x"
    args.output_dir = os.path.join("models", "detectron2_models", args.model_name, args.dataset_name)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- Run Setup and Training/Evaluation ---
    logger.info("Starting setup...")
    cfg = setup(args)
    # Save the config in model output directory
    cfg_file_path = os.path.join(cfg.OUTPUT_DIR, "config.yaml")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    with open(cfg_file_path, 'w') as f:
        f.write(cfg.dump())
    logger.info(f"Configuration saved to {cfg_file_path}")
    logger.info(f"Using model: {args.model_name} for dataset: {args.dataset_name}")
    logger.info(f"Output directory: {cfg.OUTPUT_DIR}")

    if args.eval_only:
        logger.info("--- Starting Evaluation ---")
        if not os.path.exists(cfg.OUTPUT_DIR) or not os.path.exists(os.path.join(cfg.OUTPUT_DIR, "model_final.pth")):
            raise FileNotFoundError(f"Output directory {cfg.OUTPUT_DIR} does not exist. Please run training first.")
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        val_dataset_name = f"{args.dataset_name}_panoptic_val"

        panoptic_evaluator = COCOPanopticEvaluator(val_dataset_name + "_separated", output_dir=os.path.join(cfg.OUTPUT_DIR, "panoptic_evaluation"))
        semseg_evaluator = SemSegEvaluator(
            val_dataset_name + "_separated",
            distributed=False,
            output_dir=os.path.join(cfg.OUTPUT_DIR, "semseg_evaluation"),
        )
        val_loader = build_detection_test_loader(cfg, val_dataset_name + "_separated")

        predictor = DefaultPredictor(cfg)

        logger.info("--- SemSegEvaluator (semantic head, stuff classes) ---")
        semseg_results = inference_on_dataset(predictor.model, val_loader, semseg_evaluator)
        logger.info(semseg_results)

        logger.info("--- COCOPanopticEvaluator (panoptic PQ/SQ/RQ) ---")
        panoptic_results = inference_on_dataset(predictor.model, val_loader, panoptic_evaluator)
        logger.info(panoptic_results)
    else:
        logger.info("--- Starting Training ---")
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.model_name + ".yaml")
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=True)
        trainer.train()