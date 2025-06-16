import os
from pathlib import Path
import shutil 
import yaml
import logging
from tqdm import tqdm

"""

Turn YOLO data organised by train,test,val into being organised by images,labels

"""

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def invert_yolo_data(data_path: str):
    """
    Invert YOLO data organised by train, val into being organised by images,labels
    """
    logger.info(f"Starting YOLO data inversion for: {data_path}")
    
    # Check if train,test,val folders exist
    required_folders = ["train", "valid", "test"]
    for folder in required_folders:
        if not os.path.exists(os.path.join(data_path, folder)):
            logger.warning(f"{folder} folder does not exist")
    
    # Create images and labels folders
    images_path = os.path.join(data_path, "images")
    labels_path = os.path.join(data_path, "labels")
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)
    logger.info("Created images and labels directories")
    
    # Create train,val folders in images and labels
    splits = ["train", "valid", "test"]
    for split in tqdm(splits, desc="Creating split directories"):
        split_images_path = os.path.join(images_path, split)
        split_labels_path = os.path.join(labels_path, split)
        os.makedirs(split_images_path, exist_ok=True)
        os.makedirs(split_labels_path, exist_ok=True)
    
    # Rename folders to shift them
    logger.info("Moving directories to new structure...")
    for split in tqdm(splits, desc="Moving split directories"):
        old_images = os.path.join(data_path, split, "images")
        old_labels = os.path.join(data_path, split, "labels")
        new_images = os.path.join(data_path, "images", split)
        new_labels = os.path.join(data_path, "labels", split)
        
        if os.path.exists(old_images):
            os.rename(old_images, new_images)
            logger.info(f"Moved {old_images} -> {new_images}")
        
        if os.path.exists(old_labels):
            os.rename(old_labels, new_labels)
            logger.info(f"Moved {old_labels} -> {new_labels}")
    
    # remove old train,test,val folders
    for folder in required_folders:
        os.rmdir(os.path.join(data_path, folder))
    logger.info("Data inversion completed successfully")

def invert_yolo_data_from_roboflow(input_filepath: str):
    """
    Transforms a dataset YAML file by changing image paths and reformatting 'names'.

    Args:
        input_filepath (str): The path to the original YAML file.
        output_filepath (str, optional): The path where the modified YAML will be saved.
                                         If None, the output will be printed to console.
    """
    logger.info(f"Processing YAML file: {input_filepath}")
    
    try:
        with open(input_filepath, 'r') as file:
            data = yaml.safe_load(file)

        # 1. Change the image paths
        path_changes = []
        if 'train' in data:
            data['train'] = './images/train'
            path_changes.append('train')
        if 'val' in data:
            data['val'] = './images/valid'
            path_changes.append('val')
        if 'test' in data:
            data['test'] = './images/test'
            path_changes.append('test')
        
        logger.info(f"Updated paths for: {', '.join(path_changes)}")

        # 2. Reformat 'names' from a list to a dictionary
        if 'names' in data and isinstance(data['names'], list):
            new_names = {i: name for i, name in enumerate(data['names'])}
            data['names'] = new_names
            logger.info(f"Converted names list to dictionary with {len(new_names)} classes")
        
        # Rename to data.yaml to dataset.yaml
        input_filepath = Path(input_filepath)
        output_filepath = input_filepath.parent / "data.yaml"

        with open(output_filepath, 'w') as file:
            yaml.dump(data, file, default_flow_style=False, sort_keys=False)
        logger.info(f"Transformed YAML saved to: {output_filepath}")

    except FileNotFoundError:
        logger.error(f"File not found: {input_filepath}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    # invert_yolo_data("data/bus detextion.v2i.yolov11")
    invert_yolo_data_from_roboflow("data/data.yaml")
