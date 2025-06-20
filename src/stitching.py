import os
import random
import logging

import cv2

# Set up logger
logger = logging.getLogger(__name__)

class StitchingCrops:
    def __init__(self, path):
        self.path = path
        self.crops = {} 
        self.classes = []
        self.load_crops()
        logger.info(f"Initialized StitchingCrops with path: {path}")

    def load_crops(self):
        logger.info(f"Loading crops from path: {self.path}")
        for folder in os.listdir(self.path):
            self.classes.append(folder)
            self.crops[folder] = []
            for file in os.listdir(os.path.join(self.path, folder)):
                self.crops[folder].append(os.path.join(self.path, folder, file))
            logger.debug(f"Loaded {len(self.crops[folder])} crops for class '{folder}'")
        logger.info(f"Finished loading crops. Total classes: {len(self.classes)}")

    def get_random_crop(self):
        # Get random class
        chosen_class = random.choice(self.classes)

        # Get random crop
        chosen_crop = random.choice(self.crops[chosen_class])

        image = cv2.imread(chosen_crop)
        if image is None:
            logger.warning(f"Could not load image: {chosen_crop}")
            return self.get_random_crop()
        
        # Return crop
        return image, chosen_class

    def get_crops(self):
        return self.crops

    def get_classes(self):
        return self.classes