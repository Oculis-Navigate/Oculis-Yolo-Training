from typing import List
import os
import shutil

"""

Merge X yolo datasets into one

"""

def merge_yolo_datasets(datasets: List[str], output_path: str):
    """
    Merge X yolo datasets into one
    """

    # Create output folders in YOLO format

    os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "labels"), exist_ok=True)

    os.makedirs(os.path.join(output_path, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "images", "valid"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "images", "test"), exist_ok=True)

    os.makedirs(os.path.join(output_path, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "labels", "valid"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "labels", "test"), exist_ok=True)

    # Move images and labels from each dataset
    for dataset in datasets:
        for split in ["train", "valid", "test"]:
            images_path = os.path.join(dataset, "images", split)
            labels_path = os.path.join(dataset, "labels", split)

            dataset_name = os.path.basename(dataset)

            for image in os.listdir(images_path):
                shutil.move(os.path.join(images_path, image), os.path.join(output_path, "images", split, dataset_name, image))    

            for label in os.listdir(labels_path):
                shutil.move(os.path.join(labels_path, label), os.path.join(output_path, "labels", split, dataset_name, label))

    print(f"Merged {len(datasets)} datasets into {output_path}")
    
    