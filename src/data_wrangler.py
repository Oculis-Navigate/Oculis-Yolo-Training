import os
import random
import shutil
import cv2
import logging
from tqdm import tqdm
from ultralytics import YOLO

"""

Data wrangler

Key Functions:
- Crop out images based on class
- Remove a certain class and replace with color
- Crop images using a trained model 

Should be in this format:

Folder
    - images
        - train
        - valid
        - test
    - labels
        - train
        - valid
        - test

"""

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YOLODataset:
    def __init__(self, path: str, classes: list[str], splits_use: list[str] = ["train", "valid", "test"]):
        self.path = path
        self.classes = classes
        self.splits_use = splits_use
        
        # No need load images
        self.labels = []

        logger.info(f"Initializing YOLO dataset from: {path}")
        logger.info(f"Classes: {classes}")
        logger.info(f"Splits: {splits_use}")
        self.load_data()

    def load_data(self):
        logger.info("Loading dataset metadata...")
        total_files = 0
        
        for split in self.splits_use:
            labels_path = os.path.join(self.path, "labels", split) 
            if not os.path.exists(labels_path):
                logger.warning(f"Labels path not found: {labels_path}")
                continue

            # Filter only .txt files for efficiency
            label_files = [f for f in os.listdir(labels_path) if f.endswith('.txt')]
            logger.info(f"Found {len(label_files)} label files in {split} split")
            
            # walk through all label files
            for file in tqdm(label_files, desc=f"Loading {split} labels"):
                file_path = os.path.join(labels_path, file)
                
                # Skip empty files
                if os.path.getsize(file_path) == 0:
                    continue
                    
                with open(file_path, "r") as f:
                    # Read the file line by line
                    overall_dict = {}

                    label_path = file_path
                    image_path = label_path.replace("labels", "images", 1)
                    
                    # Change extension from .txt to common image formats
                    base_name = os.path.splitext(image_path)[0]
                    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                        potential_image = base_name + ext
                        if os.path.exists(potential_image):
                            image_path = potential_image
                            break

                    overall_dict["image_path"] = image_path
                    overall_dict["label_path"] = label_path

                    overall_dict["labels"] = []
                    overall_dict["split"] = split

                    for line in f:
                        line = line.strip()
                        if not line:  # Skip empty lines
                            continue
                            
                        # Split the line into parts
                        parts = line.split()
                        if len(parts) != 5:  # Skip malformed lines
                            continue
                            
                        try:
                            # The first part is the class
                            obj_class = int(parts[0]) 
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])

                            overall_dict["labels"].append({
                                "obj_class": obj_class,
                                "x_center": x_center,
                                "y_center": y_center,
                                "width": width,
                                "height": height
                            })
                        except ValueError:
                            # Skip lines with invalid data
                            continue
                    
                    if overall_dict["labels"]:  # Only add if there are valid labels
                        available_classes = set([label["obj_class"] for label in overall_dict["labels"]])
                        overall_dict["available_classes"] = available_classes
                        self.labels.append(overall_dict)
                        total_files += 1
        
        logger.info(f"Loaded {total_files} files with {len(self.labels)} valid samples")
                        
    def crop_images(self, output_path: str, crop_filter: list[int]): 
        logger.info(f"Starting image cropping. Output: {output_path}")
        logger.info(f"Crop filter classes: {crop_filter}")
        
        # Pre-create all needed directories once
        for split in self.splits_use:
            for class_idx in crop_filter:
                if class_idx < len(self.classes):  # Safety check
                    class_name = self.classes[class_idx]
                    os.makedirs(os.path.join(output_path, split, class_name), exist_ok=True)
        
        total_crops = 0
        filtered_labels = [label for label in self.labels 
                          if len(label["available_classes"].intersection(crop_filter)) > 0]
        
        logger.info(f"Processing {len(filtered_labels)} images for cropping")
        
        for label in tqdm(filtered_labels, desc="Cropping images"):
            output_path_image = os.path.join(output_path, label["split"])
            count = 0

            # Load original image once per file
            original_image = cv2.imread(label["image_path"])
            if original_image is None:
                logger.warning(f"Could not load image: {label['image_path']}")
                continue
                
            # Get dimensions from the loaded image
            image_height, image_width = original_image.shape[:2]

            for obj in label["labels"]:
                if obj["obj_class"] in crop_filter:
                    # De normalise the bounding box 
                    try:
                        x_center = obj["x_center"] * image_width
                        y_center = obj["y_center"] * image_height
                        width = obj["width"] * image_width
                        height = obj["height"] * image_height
    
                        # Crop the image
                        cropped_image = original_image[int(y_center - height / 2):int(y_center + height / 2), 
                                      int(x_center - width / 2):int(x_center + width / 2)]
    
                        # Save the image
                        class_name = self.classes[obj["obj_class"]]
    
                        # No need to create directory - already done
                        class_output_dir = os.path.join(output_path_image, class_name)
                        output_path_full = os.path.join(class_output_dir, os.path.basename(label['image_path']).replace(".", f"_{count}."))
                        cv2.imwrite(output_path_full, cropped_image)   
    
                        count += 1
                        total_crops += 1
                    except:
                        print("Error cropping images")
        logger.info(f"Cropping completed. Total crops saved: {total_crops}")
                        
    def remove_classes_inplace(self, remove_filter: list[int]): 
        logger.info(f"Starting class removal. Classes to remove: {remove_filter}")
        
        processed_images = 0
        total_removed = 0
        
        for label in tqdm(self.labels, desc="Removing classes"):
            old_labels = label["labels"]
            label["labels"] = [obj for obj in label["labels"] if obj["obj_class"] not in remove_filter]

            label["available_classes"] = label["available_classes"].difference(remove_filter)

            # Remove from image
            to_remove = set(old_labels).difference(set(label["labels"])) 

            if len(to_remove) == 0:
                continue

            image = cv2.imread(label["image_path"]) 
            if image is None:
                logger.warning(f"Could not load image: {label['image_path']}")
                continue

            image_height, image_width = image.shape[:2]

            for obj in to_remove:
                y_center = obj["y_center"] * image_height
                x_center = obj["x_center"] * image_width
                height = obj["height"] * image_height
                width = obj["width"] * image_width

                # Add bounds checking to prevent array out of bounds
                y1 = max(0, int(y_center - height / 2))
                y2 = min(image_height, int(y_center + height / 2))
                x1 = max(0, int(x_center - width / 2))
                x2 = min(image_width, int(x_center + width / 2))

                image[y1:y2, x1:x2] = [0, 0, 0]
                total_removed += 1

            # Remove redundant makedirs - the directory should already exist
            cv2.imwrite(label["image_path"], image)
            processed_images += 1
        
        logger.info(f"Class removal completed. Processed {processed_images} images, removed {total_removed} objects")

    def crop_using_model(self, model: YOLO, split: str, output_path: str, class_name: str):
        logger.info(f"Starting image cropping using model. Output: {output_path}") 

        # os.makedirs(output_path, exist_ok=True)

        # Call the model.predict method
        results = model.predict(self.path + "/images/**/*.*", save=True, save_crop=True) 

        saved_dir = results[0].save_dir + "/crops/" + class_name

        # Copy entire saved directory to output path 
        shutil.copytree(saved_dir, output_path)

        logger.info(f"Image cropping completed. Total crops saved: {len(os.listdir(output_path))}")

    def crop_bus_with_number(self, output_path: str, bus_index: int, number_indices: list[int]):
        # Init dataset at output path
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "labels"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "images", "train"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "images", "val"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "labels", "train"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "labels", "val"), exist_ok=True)

        # Loop thru labels
        for label in tqdm(self.labels, desc="Cropping bus with number"):
            split = label["split"] 

            if not bus_index in label["available_classes"]:
                continue

            image_labels = label["labels"] 

            # Get labels with bus index
            bus_labels = [obj for obj in image_labels if obj["obj_class"] == bus_index]

            # Get labels with number indices
            number_labels = [obj for obj in image_labels if obj["obj_class"] in number_indices]
            
            # Loop thru bus labels
            for index, bus_label in enumerate(bus_labels):

                bus_crop_output_path = os.path.join(output_path, "images", split, f"{os.path.basename(label['image_path'])}_{index}.jpg")
                bus_crop_output_label_path = os.path.join(output_path, "labels", split, f"{os.path.basename(label['image_path'])}_{index}.txt")

                # Get bus image
                bus_image = cv2.imread(label["image_path"])
                bus_image_height, bus_image_width = bus_image.shape[:2]

                # Get bus bounding box
                x_center = bus_label["x_center"] * bus_image_width
                y_center = bus_label["y_center"] * bus_image_height
                width = bus_label["width"] * bus_image_width
                height = bus_label["height"] * bus_image_height

                bus_x1 = int(x_center - width / 2)
                bus_x2 = int(x_center + width / 2)
                bus_y1 = int(y_center - height / 2)
                bus_y2 = int(y_center + height / 2)

                # Crop bus image
                bus_crop = bus_image[bus_y1:bus_y2, bus_x1:bus_x2]

                bus_width = bus_x2 - bus_x1
                bus_height = bus_y2 - bus_y1

                # Loop thru number labels and get labels within the bus bounding box
                matching_number_labels = []
                for number_label in number_labels:
                    number_x_center = number_label["x_center"] * bus_image_width
                    number_y_center = number_label["y_center"] * bus_image_height
                    number_width = number_label["width"] * bus_image_width
                    number_height = number_label["height"] * bus_image_height 

                    number_x1 = number_x_center - number_width / 2
                    number_x2 = number_x_center + number_width / 2
                    number_y1 = number_y_center - number_height / 2
                    number_y2 = number_y_center + number_height / 2

                    # Check if number label is within bus bounding box
                    # If it is, normalise the label to the bus crop
                    if number_x1 > bus_x1 and number_x2 < bus_x2 and number_y1 > bus_y1 and number_y2 < bus_y2:
                        # Normalise the label to the bus crop
                        number_label["x_center"] = (number_x_center - bus_x1) / bus_width
                        number_label["y_center"] = (number_y_center - bus_y1) / bus_height
                        number_label["width"] = number_width / bus_width
                        number_label["height"] = number_height / bus_height
                        
                        # Save the label
                        matching_number_labels.append(number_label) 

                # Save the bus crop
                try:
                    cv2.imwrite(bus_crop_output_path, bus_crop)
                except:
                    print("Error saving bus crop")
                    continue

                # Save the labels
                with open(bus_crop_output_label_path, "w") as f:
                    for number_label in matching_number_labels:
                        f.write(f"{number_label['obj_class']} {number_label['x_center']} {number_label['y_center']} {number_label['width']} {number_label['height']}\n")

        return output_path