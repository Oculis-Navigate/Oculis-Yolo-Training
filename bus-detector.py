import os
from src.coco import download_coco_dataset
from src.roboflow import download_dataset
from src.remap import remap_labels
from src.invert import invert_yolo_data
from src.merger import merge_yolo_datasets
from src.ai import train_model, test_model
from src.data_wrangler import YOLODataset

"""

Training Strategy:

1. Download COCO dataset and remove all buses  
2. Download Singapore Bus Data and crop out all buses
3. Stitch singapore buses onto COCO dataset background images
4. Merge stitched dataset with original singapore bus dataset 
5. Train model on merged dataset

"""


# Download the COCO dataset

coco_dir = "data/bus-coco"
download_coco_dataset(coco_dir)

# Rename to train val test
os.rename(f"{coco_dir}/images/train2017", f"{coco_dir}/images/train")
os.rename(f"{coco_dir}/images/val2017", f"{coco_dir}/images/val")
os.rename(f"{coco_dir}/labels/train2017", f"{coco_dir}/labels/train")
os.rename(f"{coco_dir}/labels/val2017", f"{coco_dir}/labels/val")

# Relabel with only buses left
remap_labels(f"{coco_dir}/labels/train", {0: [5]})
remap_labels(f"{coco_dir}/labels/val", {0: [5]})


coco_dataset = YOLODataset(coco_dir, ["bus"], ["train", "val"])
coco_dataset.remove_classes_inplace([0])

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