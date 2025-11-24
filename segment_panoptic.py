import warnings
import os
import logging

# --- SUPPRESS WARNINGS ---
# 1. Suppress Python UserWarnings (Fixes the torch.meshgrid warning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 2. Suppress Abseil/C++ logs (Fixes the W1123 fx_tracing warnings)
# '2' filters out warnings, leaving only errors.
os.environ['GLOG_minloglevel'] = '2' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# 3. Suppress internal torch/detectron loggers
logging.getLogger("detectron2").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("torch.fx").setLevel(logging.ERROR)
# -------------------------

import os
import argparse
import glob
from tqdm import tqdm
import cv2
import sys
import json
import random
import numpy as np
import csv
import torch
import torch.multiprocessing as mp
import queue
import time

# Detectron2 imports
sys.path.insert(0, os.path.abspath('./detectron2'))
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

def get_class_info(source_instance_json, types_json):
    with open(source_instance_json, 'r') as f:
        data = json.load(f)
    with open(types_json, 'r') as f:
        types_map = json.load(f)
        
    categories = data['categories']
    all_classes = [cat['name'] for cat in categories]
    thing_classes = [cat['name'] for cat in categories if types_map.get(cat['name']) != 'stuff']
    stuff_classes = [cat['name'] for cat in categories if types_map.get(cat['name']) == 'stuff']
    return thing_classes, stuff_classes, all_classes

def setup_cfg(args, device_id):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.model_name + ".yaml"))
    thing_classes, stuff_classes, _ = get_class_info(args.source_instance_json, args.types_json)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(stuff_classes) + 1 
    cfg.MODEL.WEIGHTS = os.path.join(args.model_output_dir, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = f'cuda:{device_id}'
    cfg.freeze()
    return cfg

def worker_process(gpu_id, task_queue, progress_queue, args):
    """
    Worker that consumes tasks and signals progress.
    """
    # 1. Optimization: Disable CPU threading for CV2
    cv2.setNumThreads(0) 

    # 2. Setup
    try:
        cfg = setup_cfg(args, gpu_id)
        predictor = DefaultPredictor(cfg)
    except Exception as e:
        print(f"[GPU {gpu_id}] Error setting up model: {e}")
        return

    thing_classes, stuff_classes, _ = get_class_info(args.source_instance_json, args.types_json)
    metadata = MetadataCatalog.get(f"{args.dataset_name}_inference_meta_{gpu_id}")
    metadata.thing_classes = thing_classes
    metadata.stuff_classes = ["thing"] + stuff_classes

    partial_csv_path = os.path.join(args.output_dir, f"results_part_{gpu_id}.csv")
    
    while True:
        try:
            # Get task with timeout
            path = task_queue.get(timeout=2)
        except queue.Empty:
            # No more work
            break

        # We use a finally block to ensure progress is updated 
        # even if we skip the image or hit an error
        try:
            img = cv2.imread(path)
            if img is None:
                # Skip but report progress
                continue 

            output_path = os.path.join(args.output_dir, os.path.basename(path))
            img_date = 'T'.join(os.path.basename(path).split('.')[0].split('_')[-2:])
            
            # Skip if exists
            if os.path.exists(output_path) and not args.overwrite:
                continue

            # Inference
            panoptic_seg, segments_info = predictor(img)["panoptic_seg"]
            
            # Visualization
            if not args.skip_vis:
                v = Visualizer(img[:, :, ::-1], metadata, scale=1)
                out = v.draw_panoptic_seg(panoptic_seg.to("cpu"), segments_info, alpha=0.25)
                cv2.imwrite(output_path, out.get_image()[:, :, ::-1])

            # Calculations
            category_areas = {cat : 0 for cat in thing_classes + stuff_classes}
            for s, segment in enumerate(segments_info):
                segment.update({"category" : thing_classes[segment["category_id"]] if segment["isthing"] else stuff_classes[segment["category_id"]-1] })
                segments_info[s] = segment
                category_areas[segment['category']] += segment['area']

            # Write Partial CSV
            row_data = [img_date, os.path.basename(path)] + [category_areas[cat] for cat in thing_classes + stuff_classes]
            with open(partial_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row_data)

            # Save JSON
            json.dump(segments_info, open(output_path + '.json', 'w'), indent=4)

        except Exception as e:
            print(f"\n[GPU {gpu_id}] Error processing {path}: {e}")
        
        finally:
            # Signal to main process that 1 task is done (regardless of success/skip)
            progress_queue.put(1)

def main(args):
    if not torch.cuda.is_available():
        print("Error: CUDA not available.")
        return

    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs available.")

    os.makedirs(args.output_dir, exist_ok=True)

    image_paths = glob.glob(os.path.join(args.input_dir, '*.jpg'))
    image_paths.sort()
    total_images = len(image_paths)
    print(f"Found {total_images} images to process.")

    if total_images == 0:
        return

    # --- IPC Setup ---
    manager = mp.Manager()
    task_queue = manager.Queue()      # Holds file paths
    progress_queue = manager.Queue()  # Holds completion signals

    # Fill Task Queue
    for path in image_paths:
        task_queue.put(path)

    # --- Start Workers ---
    mp.set_start_method('spawn', force=True)
    processes = []
    
    print(f"Starting {num_gpus} workers...")
    for gpu_id in range(num_gpus):
        p = mp.Process(target=worker_process, args=(gpu_id, task_queue, progress_queue, args))
        p.start()
        processes.append(p)

    # --- Main Process acts as Progress Monitor ---
    # instead of p.join(), we loop until we have received 'total_images' signals
    
    pbar = tqdm(total=total_images, desc="Total Progress", unit="img")
    
    completed_count = 0
    while completed_count < total_images:
        try:
            # Wait for a signal from any worker
            _ = progress_queue.get(timeout=5) 
            pbar.update(1)
            completed_count += 1
        except queue.Empty:
            # Check if processes are still alive (in case of crash)
            if not any(p.is_alive() for p in processes):
                print("\nAll workers have died unexpectedly!")
                break
    
    pbar.close()

    # Ensure all processes close cleanly
    for p in processes:
        p.join()

    # --- Merge and Sort Results ---
    print("Merging and sorting results...")
    thing_classes, stuff_classes, _ = get_class_info(args.source_instance_json, args.types_json)
    header = ['date','img_fname'] + thing_classes + stuff_classes
    
    all_results = []

    for gpu_id in range(num_gpus):
        part_file = os.path.join(args.output_dir, f"results_part_{gpu_id}.csv")
        if os.path.exists(part_file):
            with open(part_file, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row:
                        all_results.append(row)
            os.remove(part_file)

    # Sort by date (1st column)
    all_results.sort(key=lambda x: x[0])

    final_csv_path = os.path.join(args.output_dir, "results.csv")
    with open(final_csv_path, 'w', newline='') as results_file:
        writer = csv.writer(results_file)
        writer.writerow(header)
        writer.writerows(all_results)

    print(f"Inference complete. Results saved to: {final_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default="snow_4cam")
    parser.add_argument("--skip-vis", action='store_true', help="Skip visualization for speed.")
    parser.add_argument("--overwrite", action='store_true', help="Overwrite existing files.")
    args = parser.parse_args()

    # Paths
    model_name = "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x"
    args.model_output_dir = os.path.join("models", "detectron2_models", model_name, args.dataset_name)
    args.model_name = model_name
    args.img_train_dir = os.path.join("..", "img_train")
    args.source_instance_json = os.path.join(args.img_train_dir, f"{args.dataset_name}_train.json")
    args.types_json = os.path.join(args.img_train_dir, f"{args.dataset_name}_types.json")
    args.input_dir = os.path.join("..", "img", args.dataset_name)
    args.output_dir = os.path.join("..", "img_seg", args.dataset_name, "detectron2", model_name)

    main(args)