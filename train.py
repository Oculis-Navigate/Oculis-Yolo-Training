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
from ultralytics import YOLO
from src.utils import copy_random_half_files


train_model(
    "yolo11l.pt",
    "default.yaml",
    "data.yaml",
)