import os
import random
import cv2

"""

Data wrangler

Key Functions:
- Crop out images based on class
- Remove a certain class and replace with color

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


class YOLODataset:
    def __init__(self, path: str, classes: list[str], splits_use: list[str] = ["train", "valid", "test"]):
        self.path = path
        self.classes = classes
        self.splits_use = splits_use
        
        # No need load images
        self.labels = []

        self.load_data()

    def load_data(self):
        for split in self.splits_use:
            labels_path = os.path.join(self.path, "labels", split) 

            # walk through all files in the labels folder
            for file in os.listdir(labels_path):
                with open(os.path.join(labels_path, file), "r") as f:
                    # Read the file line by line
                    overall_dict = {}

                    label_path = os.path.join(labels_path, file)
                    image_path = label_path.replace("labels", "images", 1)

                    overall_dict["image_path"] = image_path
                    overall_dict["label_path"] = label_path

                    overall_dict["labels"] = []
                    overall_dict["split"] = split

                    for line in f:
                        # Split the line into parts
                        parts = line.split()
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
                    
                    available_classes = set([label["obj_class"] for label in overall_dict["labels"]])

                    overall_dict["available_classes"] = available_classes

                    self.labels.append(overall_dict)
                        
    def crop_images(self, output_path: str, crop_filter: list[int]): 
        for label in self.labels:
            if len(label["available_classes"].intersection(crop_filter)) > 0:
                image = cv2.imread(label["image_path"])

                image_width = image.shape[1]
                image_height = image.shape[0]

                output_path_image = os.path.join(output_path, label["split"])

                count = 0

                for obj in label["labels"]:
                    if obj["obj_class"] in crop_filter:
                        
                        # De normalise the bounding box 
                        x_center = obj["x_center"] * image_width
                        y_center = obj["y_center"] * image_height
                        width = obj["width"] * image_width
                        height = obj["height"] * image_height

                        # Crop the image
                        image = image[int(y_center - height / 2):int(y_center + height / 2), 
                                      int(x_center - width / 2):int(x_center + width / 2)]

                        # Save the image
                        class_name = self.classes[obj["obj_class"]]

                        output_path_full = os.path.join(output_path_image, class_name, os.path.basename(label['image_path']).replace(".", f"_{count}."))
                        cv2.imwrite(output_path_full, image)   

                        count += 1
                        
    def remove_classes_inplace(self, remove_filter: list[int]): 
        for label in self.labels:
            old_labels = label["labels"]
            label["labels"] = [obj for obj in label["labels"] if obj["obj_class"] not in remove_filter]

            label["available_classes"] = label["available_classes"].difference(remove_filter)

            # Remove from image
            to_remove = set(old_labels).difference(set(label["labels"])) 

            if len(to_remove) == 0:
                continue

            image = cv2.imread(label["image_path"]) 

            image_width = image.shape[1]    
            image_height = image.shape[0]

            for obj in to_remove:
                y_center = obj["y_center"] * image_height
                x_center = obj["x_center"] * image_width
                height = obj["height"] * image_height
                width = obj["width"] * image_width

                image[int(y_center - height / 2):int(y_center + height / 2), 
                      int(x_center - width / 2):int(x_center + width / 2)] = [0, 0, 0]

            cv2.imwrite(label["image_path"], image)
