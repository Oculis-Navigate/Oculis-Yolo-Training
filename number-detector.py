"""

Classes:

0: 0
1: 1
2: 2
3: 3
4: 4
5: 5
6: 6
7: 7
8: 8
9: 9
10: A
11: E
12:e
13: G
14: M
15: W
"""

from ultralytics import YOLO
from src.roboflow import download_dataset
from src.remap import remap_labels
from src.invert import invert_yolo_data
from src.data_wrangler import YOLODataset
from src.merger import merge_yolo_datasets

from src.ai import train_model

ROBOFLOW_PATHS_UNLABELED = [
    ["https://universe.roboflow.com/ram-khlww/bus-emts1-chcch", 
     {16: [11], 0: [0], 1: [1], 2: [2], 3: [3], 4: [4], 5: [5], 6: [6], 7: [7], 8: [8], 9: [9], 10: [10], 11: [12], 12: [16], 13: [13], 14: [14], 15: [15]}, 
     "train"],
]

output_paths_labeled = []

for path, labels, split in ROBOFLOW_PATHS_UNLABELED:
    dataset_path = download_dataset(path, output_dir="data_new")
    invert_yolo_data(dataset_path)
    remap_labels(f"{dataset_path}/labels/train", labels)
    remap_labels(f"{dataset_path}/labels/valid", labels)

    dataset = YOLODataset(dataset_path, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "E", "e", "G", "M", "W"], ["train", "valid"]) 
    crop_output = dataset.crop_bus_with_number(f"./bus_crops", 16, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])


ROBOFLOW_PATHS_UNLABELED = [
    ["https://universe.roboflow.com/nanyang-polytechnic-rskkz/busdigits", 
     {0: [0], 1: [1], 2: [2], 3: [3], 4: [4], 5: [5], 6: [6], 7: [7], 8: [8], 9: [9]}, 
     "train"],
]

model = YOLO("models/best_16jun_3.pt")

output_paths_unlabeled = []
for path, labels, split in ROBOFLOW_PATHS_UNLABELED:
    dataset_path = download_dataset(path, output_dir="data_new")
    invert_yolo_data(dataset_path)
    remap_labels(f"{dataset_path}/labels/train", labels)
    remap_labels(f"{dataset_path}/labels/valid", labels)

    dataset = YOLODataset(dataset_path, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], ["train", "valid"]) 
    crop_output = dataset.crop_bus_with_number_using_model(f"./bus_crops", model, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    output_paths_unlabeled.append(crop_output)


# # Merge datasets
# merge_yolo_datasets(["./bus_crops_model", "./bus_crops"], "data/", splits=["train","val"])

# Train model 
train_model("yolo11n.pt", "number_default.yaml", "number_data.yaml")    