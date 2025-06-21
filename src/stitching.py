import os
import random
import logging

import cv2

# Set up logger
logger = logging.getLogger(__name__)

"""
Include augmentations like partial block out, blur, etc.
"""


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

        # Randomly choose 2 augmentations
        augmentations = [self.partial_block_out, self.blur, self.saturation, self.brightness, self.contrast, self.random_erase_top_or_bottom, self.random_erase_left_or_right, self.random_erase_middle]
        chosen_augmentations = random.sample(augmentations, 2)
        for augmentation in chosen_augmentations:
            image = augmentation(image)
        
        # Return crop
        return image, chosen_class
    
    def partial_block_out(self, image):
        #  Randomly block out a part of the image
        # Get random x, y, width, height
        # make block out horzontal line rather than vertical
        x = random.randint(0, image.shape[1])
        y = random.randint(0, image.shape[0]//5)
        width = random.randint(0, image.shape[1] - x)
        height = random.randint(0, image.shape[0] - y)
        image[y:y+height, x:x+width] = 0
        return image
    
    def blur(self, image):
        # Blur the image
        image = cv2.GaussianBlur(image, (5, 5), 0)
        return image

    def saturation(self, image):
        # Saturate the image
        image = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
        return image
    
    def brightness(self, image):
        # Brighten the image
        image = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
        return image
    
    def contrast(self, image):
        # Contrast the image
        image = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
        return image
    
    def random_erase_top_or_bottom(self, image):
        # Randomly erase the top or bottom of the image
        if random.random() < 0.5:
            image[:image.shape[0]//5, :] = 0
        else:
            image[image.shape[0]//5:, :] = 0
        return image
    
    def random_erase_left_or_right(self, image):
        # Randomly erase the left or right of the image
        if random.random() < 0.5:
            image[:, :image.shape[1]//5] = 0
        else:
            image[:, image.shape[1]//5:] = 0
        return image
    
    def random_erase_middle(self, image):
        # Randomly erase the middle of the image
        image[image.shape[0]//5*2:image.shape[0]//5*3, :] = 0
        return image
    
    def get_crops(self):
        return self.crops

    def get_classes(self):
        return self.classes