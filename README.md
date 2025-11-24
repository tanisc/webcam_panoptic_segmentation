# Webcam Panoptic Segmentation

This repository provides tools for training and running inference using **Detectron2** for panoptic segmentation. It is designed to process webcam imagery, calculate segmentation areas (separating "things" from "stuff"), and output statistical data alongside visualizations.

## 📂 Expected Directory Structure

**Critical:** The scripts in this repository utilize relative paths (e.g., `../img_train`). To ensure they work correctly, your file system must be organized as follows:

```text
project_root/
├── img/                       # Input images for INFERENCE
│   └── snow_4cam/             # (Example dataset name)
│       ├── image_001.jpg
│       └── ...
│
├── img_train/                 # Input data for TRAINING
│   ├── snow_4cam_train        # Image files annotated for training
│   ├── snow_4cam_train.json   # COCO-format instance annotations
│   ├── snow_4cam_val          # Image files annotated for validation
│   ├── snow_4cam_val.json     # Validation annotations
│   └── snow_4cam_types.json   # Class mapping (defines 'stuff' vs 'things')
│
├── img_seg/                   # Output folder (Generated automatically)
│   └── snow_4cam/
│       └── detectron2/...     # Inference results (CSV + images)
│
└── webcam_panoptic_segmentation-main/  # <--- THIS REPOSITORY
    ├── detectron2_install.sh
    ├── train_panoptic.py
    ├── segment_panoptic.py
    └── ...
````

## 🛠 Installation

Prerequisites: Linux with CUDA support.

1.  **Create and activate a Conda environment:**

    ```bash
    conda create -n fpn python=3 -y
    conda activate fpn
    ```

2.  **Configure PyTorch Version (Crucial):**
    The installation script `detectron2_install.sh` is currently hardcoded for **Linux with CUDA 13**.

    Before running the script, open `detectron2_install.sh` and locate the following line:

    ```bash
    pip3 install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu130](https://download.pytorch.org/whl/cu130)
    ```

    **You must modify this line** to match your specific Operating System and CUDA version (e.g., changing `cu130` to `cu118` for CUDA 11.8 or `cpu` for CPU-only; please refer to pytorch website).

3.  **Run the installation script:**
    Once configured, run the script to install system dependencies and build Detectron2.

    ```bash
    chmod +x detectron2_install.sh
    ./detectron2_install.sh
    ```

    *Note: Type `y` when prompted if you have activated the environment.*

## 🧠 Data Preparation

### 1\. The Types JSON

The system requires a `_types.json` file alongside your training data to distinguish between countable objects ("things") and amorphous regions ("stuff").

**Example `snow_4cam_types.json`:**

```json
{
    "person": "thing",
    "car": "thing",
    "sky": "stuff",
    "snow": "stuff"
}
```

### 2\. Sorting Categories (Required)

You **must** sort your JSON categories alphabetically before training. The training scripts rely on this specific order to map category names to IDs correctly. If skipped, labels may be mismatched.

Use the provided utility script to fix your data:

```bash
python sort_json.py ../img_train/your_training_data.json ../img_train/{name}_train.json
python sort_json.py ../img_train/your_validation_data.json ../img_train/{name}_val.json
```

## 🚂 Training

The `train_panoptic.py` script automatically converts your standard COCO instance JSONs into the Panoptic format required by Detectron2 (generating semantic masks and panoptic maps on the fly).

**Basic Usage:**

```bash
python train_panoptic.py --dataset-name snow_4cam --num-gpus 1
```

**Arguments:**

  * `--dataset-name`: The prefix of your data files (e.g., looks for `..img_train/{name}_train.json`).
  * `--num-gpus`: Number of GPUs to use for distributed training.
  * `--num-epochs`: Total training duration (default: 500).

**⚠️ Important Note on Resuming vs. Restarting:**
By default, the training script attempts to resume from the last checkpoint if one exists.

  * **To Continue Training:** Simply run the script again. It will pick up where it left off (adding more iterations).
  * **To Restart / New Data:** If you have changed your training data or want to start from scratch, **you must manually delete the output folder** (e.g., `models/detectron2_models/COCO.../snow_4cam/`) before running the script. If you do not delete it, the model may crash or produce garbage results due to mismatched data shapes.

**Output:**
Models and checkpoints are saved to: `webcam_panoptic_segmentation-main/models/detectron2_models/`

## 🔮 Inference

The `segment_panoptic.py` script runs the trained model on raw images found in `../img/{dataset_name}`. It utilizes multi-processing to maximize GPU throughput.

**Basic Usage:**

```bash
python segment_panoptic.py --dataset-name snow_4cam
```

**Arguments:**

  * `--dataset-name`: Matches the folder name in `../img/`.
  * `--skip-vis`: Set this flag to generate CSV data only (skips saving overlay images) for faster processing.
  * `--overwrite`: Force reprocessing of images that already have output files.

**Output:**
Results are saved to `../img_seg/{dataset_name}/detectron2/...`

1.  **`results.csv`**: A merged CSV file containing the area (pixel count) for every class per image, sorted by date.
2.  **Visualizations**: Overlay images showing segmentation.
3.  **JSONs**: Per-image JSON files containing raw segment info.

## 📝 Script Descriptions

  * **`detectron2_install.sh` / `.py`**: Automates the complex installation of Detectron2 and its dependencies.
  * **`train_panoptic.py`**: Handles data preprocessing (COCO -\> Panoptic) and manages the training loop.
  * **`segment_panoptic.py`**: The production inference script. Includes logic for multi-GPU worker queues and data aggregation.
  * **`sort_json.py`**: Utility to alphabetize and re-index COCO JSON categories.
  * **`stash/`**: Contains older versions of training and segmentation scripts.

<!-- end list -->

Author
Cemal Melih Tanis
