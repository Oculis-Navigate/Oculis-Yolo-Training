import os
from src.coco import download_coco_dataset
from src.roboflow import download_dataset
from src.remap import remap_labels
from src.invert import invert_yolo_data
from src.merger import merge_yolo_datasets

# Download the COCO dataset

coco_dir = "data/bus-coco"
download_coco_dataset(coco_dir)

# Rename to train val test

os.rename(f"{coco_dir}/images/train2017", f"{coco_dir}/train")
os.rename(f"{coco_dir}/images/val2017", f"{coco_dir}/val")
os.rename(f"{coco_dir}/labels/train2017", f"{coco_dir}/train/labels")
os.rename(f"{coco_dir}/labels/val2017", f"{coco_dir}/val/labels")

# Relabel with only buses left

remap_labels(f"{coco_dir}/train/labels", {0: [5]})
remap_labels(f"{coco_dir}/val/labels", {0: [5]})

# Download the Singapore Bus Data

dataset_path = download_dataset("https://universe.roboflow.com/taku-3grva/bus-detextion")

# Invert the data

invert_yolo_data(dataset_path)

# Relabel the Singapore Bus Data

remap_labels(f"{dataset_path}/train/labels", {0: [0, 1, 2, 3, 4]})
remap_labels(f"{dataset_path}/valid/labels", {0: [0, 1, 2, 3, 4]})

# Merge the two datasets
merge_yolo_datasets(
    [
        coco_dir,
        dataset_path
    ],
    "data/bus-yolo"
)