import os
import shutil
from src.coco import download_coco_dataset
from src.roboflow import download_dataset
from src.remap import remap_labels
from src.invert import invert_yolo_data
from src.merger import merge_yolo_datasets
from src.ai import train_model, test_model
from src.data_wrangler import YOLODataset
import albumentations as A
from snapstitch import Stitcher, PartsLoader, BackgroundLoader, YOLOv8Generator

"""

Training Strategy:

1. Download COCO dataset and remove all buses  
2. Download Singapore Bus Data and crop out all buses
3. Stitch singapore buses onto COCO dataset background images
4. Merge stitched dataset with original singapore bus dataset 
5. Train model on merged dataset

"""

stitch_output = "stitching"
stitch_crops = "stitching/crops"
stitch_backgrounds = "stitching/backgrounds" 

pretrained_model = "models/busDetector_Distilled.pt"

# Define transformations
transform = A.Compose([
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    A.RandomBrightnessContrast(p=0.3),
])

background_transform = A.Compose([
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    A.RandomBrightnessContrast(p=0.3),
    A.RandomGamma(p=0.3),
    A.ShiftScaleRotate(p=0.3),
    A.Rotate(p=0.3),
    A.HorizontalFlip(p=0.3),
    A.VerticalFlip(p=0.3),
])

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

# Move images to backgrounds folder
os.makedirs(stitch_backgrounds, exist_ok=True)
os.rename(f"{coco_dir}/images/train", f"{stitch_backgrounds}/coco/train")
os.rename(f"{coco_dir}/images/val", f"{stitch_backgrounds}/coco/val")

# Download the Singapore Bus Data

ROBOFLOW_PATHS_UNLABELED = [
    ["https://universe.roboflow.com/taku-3grva/bus-detextion", {0: [0, 1, 2, 3, 4]}, "train"],
    ["https://universe.roboflow.com/bafo-ehbsl/bafo-xp3hy", {0: [0]}, "train"],
    ["https://universe.roboflow.com/ram-khlww/bus-emts1-chcch", {0: [11]}, "train"],
    ["https://universe.roboflow.com/test-project-csgdb/bus-route-number-testdataset", {0: [2]}, "val"],
]

output_paths_labeled = []

for path, labels, split in ROBOFLOW_PATHS_UNLABELED:
    dataset_path = download_dataset(path)
    invert_yolo_data(dataset_path)
    remap_labels(f"{dataset_path}/train/labels", labels)
    remap_labels(f"{dataset_path}/valid/labels", labels)
    output_paths_labeled.append(dataset_path)

    dataset = YOLODataset(dataset_path, ["bus"], ["train", "val"]) 
    dataset.crop_images(stitch_crops + "/" + split + "/" + path.split("/")[-1], [0])

# Collect datasets without labels

ROBOFLOW_PATHS_UNLABELED = [
    ["https://universe.roboflow.com/fulgore/sbs-bus-numbers-uqimt-x9n31", "train"],
    ["https://universe.roboflow.com/nanyang-polytechnic-rskkz/nanyang-poly---block-502-bus-stops", "train"],
    ["https://universe.roboflow.com/test-project-csgdb/bus-route-number-only-fyp", "val"],
    ["https://universe.roboflow.com/nanyang-polytechnic/sbs-bus-numbers-uqimt", "train"],
    ["https://universe.roboflow.com/nanyang-polytechnic-rskkz/busdigits", "train"],
    ["https://universe.roboflow.com/nyp/bus-number-dataset", "train"],
    ["https://universe.roboflow.com/fyp-2zcsf/bus-detection-qzyuz", "train"],
    ["https://universe.roboflow.com/school-yghmi/bus-detection-utpws", "train"],
]

output_paths_no_labels = []

for path, split in ROBOFLOW_PATHS_UNLABELED:
    # Download and extract the dataset (returns directory path, not zip path)
    dataset_dir = download_dataset(path, output_dir="data", extract=True)
    
    print(f"Dataset extracted to: {dataset_dir}")
    
    # Now run invert on the extracted directory
    invert_yolo_data(dataset_dir)
    
    # Extract crops of buses
    dataset = YOLODataset(dataset_dir, ["bus"], ["train", "val"]) 
    dataset.crop_using_model(pretrained_model, split, stitch_crops + "/" + path.split("/")[-1], "bus")


# Perform Stitching

train_background = BackgroundLoader(f"{stitch_backgrounds}/coco/train", target_size=(1280, 720), max_cache_size=200, transform=background_transform)
val_background = BackgroundLoader(f"{stitch_backgrounds}/coco/val", target_size=(1280, 720), max_cache_size=200, transform=background_transform)

bus_part_train = PartsLoader(stitch_crops + "/train", scale=0.3, transform=transform, scaling_variation=0.7, max_cache_size=1000)
bus_part_val = PartsLoader(stitch_crops + "/val", scale=0.3, transform=transform, scaling_variation=0.7, max_cache_size=1000)

generator = YOLOv8Generator(overlap_ratio=0.05)

train_stitcher = Stitcher(generator, train_background, {
    "0": [bus_part_train, 0.1],
}, parts_per_image=30)

val_stitcher = Stitcher(generator, val_background, {
    "0": [bus_part_val, 0.1],
}, parts_per_image=100)

# Remove exisint data folder
if os.path.exists("data"):
    shutil.rmtree("data")

os.makedirs("data")

# Generate datasets
train_stitcher.execute(
    7000, 
    "data", 
    "train_1", 
    train_or_val=True,
    perimeter_end=(1280, 720)
)

val_stitcher.execute(
    3000, 
    "data", 
    "val_1", 
    train_or_val=False,
    perimeter_end=(1280, 720)
)