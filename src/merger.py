from typing import List
import os
import shutil
import logging
from tqdm import tqdm

"""

Merge X yolo datasets into one

"""

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def merge_yolo_datasets(datasets: List[str], output_path: str, splits = ["train", "valid", "test"]):
    """
    Merge X yolo datasets into one
    """
    logger.info(f"Starting merge of {len(datasets)} datasets")
    logger.info(f"Output path: {output_path}")

    # Create output folders in YOLO format
    logger.info("Creating output directory structure...")
    
    os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "labels"), exist_ok=True)

    
    for split in splits:
        os.makedirs(os.path.join(output_path, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_path, "labels", split), exist_ok=True)

    # Move images and labels from each dataset
    total_files_moved = 0
    
    for dataset in tqdm(datasets, desc="Processing datasets"):
        dataset_name = os.path.basename(dataset)
        logger.info(f"Processing dataset: {dataset_name}")
        
        for split in splits:
            images_path = os.path.join(dataset, "images", split)
            labels_path = os.path.join(dataset, "labels", split)
            
            if not os.path.exists(images_path) or not os.path.exists(labels_path):
                logger.warning(f"Missing {split} split in {dataset_name}")
                continue

            # Create dataset subdirectories
            output_images_dir = os.path.join(output_path, "images", split, dataset_name)
            output_labels_dir = os.path.join(output_path, "labels", split, dataset_name)
            os.makedirs(output_images_dir, exist_ok=True)
            os.makedirs(output_labels_dir, exist_ok=True)

            # Move images
            if os.path.exists(images_path):
                image_files = os.listdir(images_path)
                for image in tqdm(image_files, desc=f"Moving {split} images", leave=False):
                    src = os.path.join(images_path, image)
                    dst = os.path.join(output_images_dir, image)
                    shutil.move(src, dst)
                    total_files_moved += 1

            # Move labels
            if os.path.exists(labels_path):
                label_files = os.listdir(labels_path)
                for label in tqdm(label_files, desc=f"Moving {split} labels", leave=False):
                    src = os.path.join(labels_path, label)
                    dst = os.path.join(output_labels_dir, label)
                    shutil.move(src, dst)
                    total_files_moved += 1

    logger.info(f"Merge completed! Moved {total_files_moved} files from {len(datasets)} datasets to {output_path}")
    
    