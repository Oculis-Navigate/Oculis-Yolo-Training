import os
import random
import cv2
import numpy as np
import logging

from tqdm import tqdm
from .stitching import StitchingCrops
from .data_wrangler import YOLODataset
from ultralytics import YOLO

# Set up logger
logger = logging.getLogger(__name__)

class BusNumberStitcher:
    def __init__(self, background_dataset: YOLODataset, bus_number_dataset: YOLODataset, bus_detector: YOLO):
        self.background_dataset = background_dataset.labels
        self.bus_number_dataset = bus_number_dataset
        self.bus_detector = bus_detector
        logger.info("Initialized BusNumberStitcher")

        # Assume class 0 in background dataset is the joined bus number

    def stitch(self, output_path: str):
        logger.info(f"Starting stitching process. Output path: {output_path}")

        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "labels"), exist_ok=True)

        os.makedirs(os.path.join(output_path, "images", "train"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "labels", "train"), exist_ok=True)

        os.makedirs(os.path.join(output_path, "images", "valid"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "labels", "valid"), exist_ok=True)
        
        logger.info("Created output directory structure")

        # Iterate over background dataset
        for background in tqdm(self.background_dataset):
            
            if not background["available_classes"]: 
                continue
            
            split = background["split"]
            labels = background["labels"]

            # Find bus using model
            image = cv2.imread(background["image_path"])
            if image is None:
                logger.warning(f"Could not load image: {background['image_path']}")
                continue
                
            bus_image_width = image.shape[1]
            bus_image_height = image.shape[0]

            results = self.bus_detector.predict(image, save=True)

            # Get bus crop
            bus_coords = results[0].boxes.xyxy 
            logger.debug(f"Detected {len(bus_coords)} buses in image")
            
            for index, coord in enumerate(bus_coords):
                x1 = int(coord[0])
                y1 = int(coord[1])
                x2 = int(coord[2])
                y2 = int(coord[3])

                print(f"Bus crop: {x1}:{x2} {y1}:{y2}")

                bus_crop_output_path = os.path.join(output_path, "images", split, f"{os.path.basename(background['image_path'])}_{index}.jpg")
                bus_crop_output_label_path = os.path.join(output_path, "labels", split, f"{os.path.basename(background['image_path'])}_{index}.txt")

                # Crop the bus
                bus_crop = image[y1:y2, x1:x2]
                print(f"Bus crop shape: {bus_crop.shape}")
                new_labels = []

                # Find labels within bus crop area
                for number_label in labels:
                    number_x_center = number_label["x_center"] * bus_image_width
                    number_y_center = number_label["y_center"] * bus_image_height
                    number_width = number_label["width"] * bus_image_width
                    number_height = number_label["height"] * bus_image_height

                    number_x1 = number_x_center - number_width / 2
                    number_x2 = number_x_center + number_width / 2
                    number_y1 = number_y_center - number_height / 2
                    number_y2 = number_y_center + number_height / 2

                    print(f"Number label: {number_x1}:{number_x2} {number_y1}:{number_y2}")

                    if number_x1 >= x1 and number_x2 <= x2 and number_y1 >= y1 and number_y2 <= y2:
                        # Label is within bus crop area so black out this area and stitch new numbers

                        # Re-nomralise to crop 
                        new_number_x1 = int(number_x1 - x1) 
                        new_number_x2 = int(number_x2 - x1) 
                        new_number_y1 = int(number_y1 - y1)
                        new_number_y2 = int(number_y2 - y1)

                        print(f"New number crop: {new_number_x1}:{new_number_x2} {new_number_y1}:{new_number_y2}")

                        bus_crop[new_number_y1:new_number_y2, new_number_x1:new_number_x2] = 0 

                        

                        crop_width = new_number_x2 - new_number_x1
                        crop_height = new_number_y2 - new_number_y1

                        # Stitch number images
                        number_images_to_stitch = random.choice([1,2,3,4])

                        to_stitch = []
                        for _ in range(number_images_to_stitch):
                            stitched_number_image, chosen_class = self.bus_number_dataset.get_random_crop()
                            # reszie to crop height without changing aspect ratio
                            new_height = int(number_height)
                            ratio = new_height / stitched_number_image.shape[0]
                            new_width = int(stitched_number_image.shape[1] * ratio)
                            print(f"Resizing to {new_width}x{new_height}")
                            stitched_number_image = cv2.resize(stitched_number_image, (new_width, new_height))

                            to_stitch.append((stitched_number_image, chosen_class))

                        # Stitch side by side
                        previous_x_end = 0
                        for index, (stitched_number_image, chosen_class) in enumerate(to_stitch):
                            starting_x1 = int(new_number_x1 + previous_x_end)
                            starting_y1 = int(new_number_y1)
                            ending_x2 = int(starting_x1 + stitched_number_image.shape[1])
                            ending_y2 = int(starting_y1 + stitched_number_image.shape[0])

                            print(f"Stitching at {starting_x1}:{ending_x2} {starting_y1}:{ending_y2}, bus crop shape: {bus_crop.shape}")
                            
                            # check if the stitched number image is within the bus crop
                            if starting_x1 >= 0 and ending_x2 <= bus_crop.shape[1] and starting_y1 >= 0 and ending_y2 <= bus_crop.shape[0]:
                                bus_crop[starting_y1:ending_y2, starting_x1:ending_x2] = stitched_number_image
                                previous_x_end += stitched_number_image.shape[1] 

                                # Add label
                                new_labels.append({
                                    "x_center": (starting_x1 + ending_x2) / 2 / bus_crop.shape[1],
                                    "y_center": (starting_y1 + ending_y2) / 2 / bus_crop.shape[0],
                                    "width": stitched_number_image.shape[1] / bus_crop.shape[1],
                                    "height": stitched_number_image.shape[0] / bus_crop.shape[0],
                                    "class": chosen_class
                                })
                            else:
                                print(f"Stitched number image is out of bounds: {starting_x1}:{ending_x2} {starting_y1}:{ending_y2}, bus crop shape: {bus_crop.shape}")
                                continue

                # Save bus crop
                cv2.imwrite(bus_crop_output_path, bus_crop)
                with open(bus_crop_output_label_path, "w") as f:
                    for label in new_labels:
                        f.write(f"{label['class']} {label['x_center']} {label['y_center']} {label['width']} {label['height']}\n")
                
                # Save a copy of bus crop for label-less background
                # Loop thru labels and de-normalise to bus crop
                # Black out the labels
                temp_bus_crop = bus_crop.copy()
                for label in new_labels:
                    label["x_center"] = label["x_center"] * bus_crop.shape[1]
                    label["y_center"] = label["y_center"] * bus_crop.shape[0]
                    label["width"] = label["width"] * bus_crop.shape[1]
                    label["height"] = label["height"] * bus_crop.shape[0]

                    temp_bus_crop[int(label["y_center"] - label["height"] / 2):int(label["y_center"] + label["height"] / 2), int(label["x_center"] - label["width"] / 2):int(label["x_center"] + label["width"] / 2)] = 0

                bus_crop_output_path_no_labels = os.path.join(output_path, "images", split, f"{os.path.basename(background['image_path'])}_{index}_no_labels.jpg")
                cv2.imwrite(bus_crop_output_path_no_labels, temp_bus_crop)
                logger.debug(f"Saved bus crop: {bus_crop_output_path}")
        
        logger.info("Stitching process completed successfully")
                            