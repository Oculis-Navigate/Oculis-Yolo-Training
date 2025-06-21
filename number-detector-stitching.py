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


ROBOFLOW_PATHS_UNLABELED = [
    ["https://universe.roboflow.com/ram-khlww/bus-emts1-chcch", 
     {16: [11], 0: [0], 1: [1], 2: [2], 3: [3], 4: [4], 5: [5], 6: [6], 7: [7], 8: [8], 9: [9], 10: [10], 11: [12], 12: [16], 13: [13], 14: [14], 15: [15]}, 
     "train"],
     ["https://universe.roboflow.com/nanyang-polytechnic-rskkz/busdigits", 
     {0: [0], 1: [1], 2: [2], 3: [3], 4: [4], 5: [5], 6: [6], 7: [7], 8: [8], 9: [9]}, 
     "train"],
]

output_paths_labeled = []

for path, labels, split in ROBOFLOW_PATHS_UNLABELED:
    dataset_path = download_dataset(path, output_dir="data_new")
    invert_yolo_data(dataset_path)
    remap_labels(f"{dataset_path}/labels/train", labels)
    remap_labels(f"{dataset_path}/labels/valid", labels)

    dataset = YOLODataset(dataset_path, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"], ["train", "valid"]) 
    dataset.crop_images(stitching_crops, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

    if path == "https://universe.roboflow.com/ram-khlww/bus-emts1-chcch":
        crop_output = dataset.crop_bus_with_number(f"./data_stitched", 16, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])


# Download pure number dataset

number_dataset = "https://universe.roboflow.com/blank-lgcth/led-display-board-detection"
number_dataset_labels = {0: [0], 1: [1], 2: [2], 3: [3], 4: [4], 5: [6], 6: [7], 7: [8], 8: [9], 9: [10], 10: [11], 13: [17], 14: [24]}
number_dataset = download_dataset(number_dataset, output_dir="data_new")
invert_yolo_data(number_dataset)
remap_labels(f"{number_dataset}/labels/train", number_dataset_labels)
remap_labels(f"{number_dataset}/labels/valid", number_dataset_labels)

number_dataset = YOLODataset(number_dataset, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"], ["train", "valid"]) 
number_dataset.crop_images(stitching_crops, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])


stitch_crops_dataset = StitchingCrops(stitching_crops + "/train")

ROBOFLOW_PATHS_UNLABELED = [
    ["https://universe.roboflow.com/nanyang-polytechnic-rskkz/nanyang-poly---block-502-bus-stops", 
     {0:[0,1,2,3]}, "train"],
    ["https://universe.roboflow.com/nanyang-polytechnic/sbs-bus-numbers-uqimt",
     {0: [0, 1, 2,3,4,5,6,7,8,9,10,11,12]}, "train"],
    ["https://universe.roboflow.com/test-project-csgdb/bus-route-number-testdataset",
     {0: [0,1]}, "train"],
    ["https://universe.roboflow.com/nyp/bus-number-dataset",
     {0: [0,1,2,3]}, "train"]
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

# Train model
train_model("number-1.pt.pt", "number_default.yaml", "number_data.yaml")    