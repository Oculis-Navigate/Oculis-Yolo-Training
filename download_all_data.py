import os
import shutil
from src.coco import download_coco_dataset
from src.roboflow import download_dataset
from src.remap import remap_labels
from src.invert import invert_yolo_data, invert_yolo_data_from_roboflow
from src.merger import merge_yolo_datasets
from src.ai import train_model, test_model
from ultralytics import YOLO
from src.utils import copy_random_half_files

from data2cvat import move_and_rename_files

"""

Training Strategy:

1. Download COCO dataset and remove all buses  
2. Download Singapore Bus Data and crop out all buses
3. Stitch singapore buses onto COCO dataset background images
4. Merge stitched dataset with original singapore bus dataset 
5. Train model on merged dataset

"""

pretrained_model = YOLO("models/busDetector_Distilled.pt")

# Download the Singapore Bus Data

ROBOFLOW_PATHS_UNLABELED = [
    ["https://universe.roboflow.com/taku-3grva/bus-detextion", {0: [0]}, "train"],
    ["https://universe.roboflow.com/ram-khlww/bus-emts1-chcch", {0: [11]}, "train"],
    ["https://universe.roboflow.com/test-project-csgdb/bus-route-number-testdataset", {0: [2]}, "val"],
]

output_paths_labeled = []

for path, labels, split in ROBOFLOW_PATHS_UNLABELED:
    dataset_path = download_dataset(path, output_dir="data", extract=True)
    invert_yolo_data(dataset_path)
    remap_labels(f"{dataset_path}/labels/train", labels)
    remap_labels(f"{dataset_path}/labels/valid", labels)
    output_paths_labeled.append(dataset_path)

    invert_yolo_data_from_roboflow(dataset_path + "/data.yaml")

# Collect datasets without labels

ROBOFLOW_PATHS_UNLABELED = [
    # ["https://universe.roboflow.com/nanyang-polytechnic-rskkz/nanyang-poly---block-502-bus-stops", "val"],
    # ["https://universe.roboflow.com/nanyang-polytechnic/sbs-bus-numbers-uqimt", "val"],
    # ["https://universe.roboflow.com/nanyang-polytechnic-rskkz/busdigits", "train"]
]

output_paths_no_labels = []

for path, split in ROBOFLOW_PATHS_UNLABELED:
    # Download and extract the dataset (returns directory path, not zip path)
    dataset_dir = download_dataset(path, output_dir="data", extract=True)
    
    print(f"Dataset extracted to: {dataset_dir}")
    
    # Now run invert on the extracted directory
    invert_yolo_data(dataset_dir)

    invert_yolo_data_from_roboflow(dataset_dir + "/data.yaml")