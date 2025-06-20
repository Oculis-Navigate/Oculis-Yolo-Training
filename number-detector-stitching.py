import os
import shutil
from src.coco import download_coco_dataset
from src.roboflow import download_dataset
from src.remap import remap_labels
from src.invert import invert_yolo_data
from src.merger import merge_yolo_datasets
from src.ai import train_model, test_model
from src.data_wrangler import YOLODataset
from ultralytics import YOLO
from src.bus_number_stitch import BusNumberStitcher
from src.stitching import StitchingCrops

stitching_output = "stitching"
stitching_crops = "stitching/crops"
stitching_backgrounds = "stitching/backgrounds" 

pretrained_model = YOLO("models/best_16jun_3.pt")


# ROBOFLOW_PATHS_UNLABELED = [
#     ["https://universe.roboflow.com/ram-khlww/bus-emts1-chcch", 
#      {16: [11], 0: [0], 1: [1], 2: [2], 3: [3], 4: [4], 5: [5], 6: [6], 7: [7], 8: [8], 9: [9], 10: [10], 11: [12], 12: [16], 13: [13], 14: [14], 15: [15]}, 
#      "train"],
# ]

# output_paths_labeled = []

# for path, labels, split in ROBOFLOW_PATHS_UNLABELED:
#     dataset_path = download_dataset(path, output_dir="data_new")
#     invert_yolo_data(dataset_path)
#     remap_labels(f"{dataset_path}/labels/train", labels)
#     remap_labels(f"{dataset_path}/labels/valid", labels)

#     dataset = YOLODataset(dataset_path, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"], ["train", "valid"]) 
#     dataset.crop_images(stitching_crops, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])


stitch_crops_dataset = StitchingCrops(stitching_crops + "/train")

ROBOFLOW_PATHS_UNLABELED = [
    ["https://universe.roboflow.com/nanyang-polytechnic-rskkz/nanyang-poly---block-502-bus-stops", 
     {0:[0,1,2,3]}, 
     "train"],
]

output_paths_labeled = []

for path, labels, split in ROBOFLOW_PATHS_UNLABELED:
    dataset_path = download_dataset(path, output_dir="data_new")
    invert_yolo_data(dataset_path)
    remap_labels(f"{dataset_path}/labels/train", labels)
    remap_labels(f"{dataset_path}/labels/valid", labels)

    dataset = YOLODataset(dataset_path, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "E", "e", "G", "M", "W"], ["train", "valid"]) 
    
    bus_stitcher = BusNumberStitcher(dataset, stitch_crops_dataset, pretrained_model)
    bus_stitcher.stitch("data_stitched")