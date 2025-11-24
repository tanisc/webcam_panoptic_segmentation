import os
import argparse
import glob
from tqdm import tqdm
import cv2
import sys
import json
# Detectron2 imports
# Add local detectron2 to path
sys.path.insert(0, os.path.abspath('./detectron2'))
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

def get_class_info(source_instance_json, types_json):
    """
    Reads source JSON files to determine the number and names of thing and stuff classes.
    """
    with open(source_instance_json, 'r') as f:
        data = json.load(f)
    with open(types_json, 'r') as f:
        types_map = json.load(f)
        
    categories = data['categories']
    all_classes = [cat['name'] for cat in categories]
    thing_classes = [cat['name'] for cat in categories if types_map.get(cat['name']) != 'stuff']
    stuff_classes = [cat['name'] for cat in categories if types_map.get(cat['name']) == 'stuff']
    
    return thing_classes, stuff_classes, all_classes


def setup_cfg(args):
    """
    Creates a configuration from scratch and loads a trained model.
    """
    cfg = get_cfg()
    # Load the base model config
    cfg.merge_from_file(model_zoo.get_config_file(args.model_name + ".yaml"))
    
    # Get the number of classes from the dataset definition files
    thing_classes, stuff_classes, all_classes = get_class_info(args.source_instance_json, args.types_json)

    # Set the number of classes for the model heads
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(stuff_classes) + 1 # +1 for the thing class

    # Set the model weight file path
    cfg.MODEL.WEIGHTS = os.path.join(args.model_output_dir, "model_final.pth")
    
    # Set the score threshold for detections
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    
    cfg.freeze()
    return cfg

def main(args):
    """
    Main function to run inference.
    """
    # Set up the configuration and create the predictor
    cfg = setup_cfg(args)
    predictor = DefaultPredictor(cfg)

    # --- Manually create metadata for visualization ---
    # This avoids issues with registration and ensures the visualizer has the correct class names.
    thing_classes, stuff_classes, all_classes = get_class_info(args.source_instance_json, args.types_json)
    
    # Create a temporary metadata object
    metadata = MetadataCatalog.get(f"{args.dataset_name}_inference_meta")
    metadata.thing_classes = thing_classes
    metadata.stuff_classes = ["thing"] + stuff_classes # Add placeholder for the 'thing' class

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Find all JPG images in the input directory
    image_paths = glob.glob(os.path.join(args.input_dir, '*.jpg'))
    # random.shuffle(image_paths)
    image_paths.sort()
    print(f"Found {len(image_paths)} images to process.")

    # Process each image and get calculations
    with open(os.path.join(args.output_dir,"results.csv"),'w') as results_file:
        row = ['date','img_fname'] + thing_classes + stuff_classes + ['\n']
        results_file.write(','.join(row))
    for path in tqdm(image_paths, desc="Processing images"):
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Could not read image {path}. Skipping.")
            continue

        # If the image is already in the output directory, skip it
        output_path = os.path.join(args.output_dir, os.path.basename(path))
        img_date = 'T'.join(os.path.basename(path).split('.')[0].split('_')[-2:])
        if os.path.exists(output_path):
            print(f"Output file {output_path} already exists. Skipping.")
            continue
            
        # Perform inference
        panoptic_seg, segments_info = predictor(img)["panoptic_seg"]
        
        
        # Create a visualizer with our manually created metadata
        v = Visualizer(img[:, :, ::-1], metadata, scale=1.2)
        
        # Draw the panoptic segmentation predictions
        out = v.draw_panoptic_seg(panoptic_seg.to("cpu"), segments_info, alpha=0.25)
        
        # Save the image
        cv2.imwrite(output_path, out.get_image()[:, :, ::-1])

        category_areas = {cat : 0 for cat in thing_classes + stuff_classes}
        for s, segment in enumerate(segments_info):
            segment.update({"category" : thing_classes[segment["category_id"]] if segment["isthing"] else stuff_classes[segment["category_id"]-1] })
            segments_info[s] = segment

            category_areas[segment['category']] += segment['area']

        with open(os.path.join(args.output_dir,"results.csv"),'a') as results_file:
            results_file.write(','.join(list(map(str,
                    [img_date,os.path.basename(path)] + [category_areas[cat] for cat in thing_classes + stuff_classes] + ['\n']
                    ))))
        json.dump(segments_info,open(output_path + '.json','w'),indent=4)

    print(f"Inference complete. Visualizations saved to: {args.output_dir}")
    print(f"Calculations saved to: {args.output_dir} /results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detectron2 Panoptic Segmentation Inference")
    parser.add_argument("--dataset-name", type=str, default="snow_4cam", help="The base name of the dataset to locate the trained model.")
    args = parser.parse_args()

    # --- Centralized Path and Model Definitions ---
    model_name = "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x"
    
    # Path to the specific model's output directory (relative to this script's location in fpn/)
    args.model_output_dir = os.path.join("models", "detectron2_models", model_name, args.dataset_name)
    args.model_name = model_name

    # Path to the directory where all dataset-related files are (relative to fpn/)
    args.img_train_dir = os.path.join("..", "img_train")
    
    # Add paths to source JSON files for class counting and metadata creation
    args.source_instance_json = os.path.join(args.img_train_dir, f"{args.dataset_name}_train.json")
    args.types_json = os.path.join(args.img_train_dir, f"{args.dataset_name}_types.json")
    
    # --- Define Input and Output Directories for Inference ---
    args.input_dir = os.path.join("..", "img", args.dataset_name)
    args.output_dir = os.path.join("..", "img_seg", args.dataset_name, "detectron2", model_name)

    main(args)
