from snapstitch import Stitcher, PartsLoader, BackgroundLoader, YOLOv8Generator
import albumentations as A

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

# Load backgrounds
background = BackgroundLoader("stitching/batch_2/train/background", target_size=(1280, 720), max_cache_size=200, transform=background_transform)
val_background = BackgroundLoader("stitching/batch_2/val/background", target_size=(1280, 720), max_cache_size=200, transform=background_transform)

# Load parts for each class with transformations
class_0 = PartsLoader("stitching/batch_2/train/0", scale=0.3, transform=transform, scaling_variation=0.7, max_cache_size=1000)
class_1 = PartsLoader("stitching/batch_2/train/1", scale=0.3, transform=transform, scaling_variation=0.7, max_cache_size=1000)
class_2 = PartsLoader("stitching/batch_2/train/2", scale=0.3, transform=transform, scaling_variation=0.7, max_cache_size=1000)
class_3 = PartsLoader("stitching/batch_2/train/3", scale=0.3, transform=transform, scaling_variation=0.7, max_cache_size=1000)
class_4 = PartsLoader("stitching/batch_2/train/4", scale=0.3, transform=transform, scaling_variation=0.7, max_cache_size=1000)
class_5 = PartsLoader("stitching/batch_2/train/5", scale=0.3, transform=transform, scaling_variation=0.7, max_cache_size=1000)
class_6 = PartsLoader("stitching/batch_2/train/6", scale=0.3, transform=transform, scaling_variation=0.7, max_cache_size=1000)
class_7 = PartsLoader("stitching/batch_2/train/7", scale=0.3, transform=transform, scaling_variation=0.7, max_cache_size=1000)
class_8 = PartsLoader("stitching/batch_2/train/8", scale=0.3, transform=transform, scaling_variation=0.7, max_cache_size=1000)
class_9 = PartsLoader("stitching/batch_2/train/9", scale=0.3, transform=transform, scaling_variation=0.7, max_cache_size=1000)
negative_samples = PartsLoader(
    "stitching/batch_2/train/misled", 
    scale=0.3,
    transform=transform, 
    scaling_variation=0.9
)

# Val
val_class_0 = PartsLoader("stitching/batch_2/val/0", scale=0.3, transform=transform, scaling_variation=0.7, max_cache_size=1000)
val_class_1 = PartsLoader("stitching/batch_2/val/1", scale=0.3, transform=transform, scaling_variation=0.7, max_cache_size=1000)
val_class_2 = PartsLoader("stitching/batch_2/val/2", scale=0.3, transform=transform, scaling_variation=0.7, max_cache_size=1000)
val_class_3 = PartsLoader("stitching/batch_2/val/3", scale=0.3, transform=transform, scaling_variation=0.7, max_cache_size=1000)
val_class_4 = PartsLoader("stitching/batch_2/val/4", scale=0.3, transform=transform, scaling_variation=0.7, max_cache_size=1000)
val_class_5 = PartsLoader("stitching/batch_2/val/5", scale=0.3, transform=transform, scaling_variation=0.7, max_cache_size=1000)
val_class_6 = PartsLoader("stitching/batch_2/val/6", scale=0.3, transform=transform, scaling_variation=0.7, max_cache_size=1000)
val_class_7 = PartsLoader("stitching/batch_2/val/7", scale=0.3, transform=transform, scaling_variation=0.7, max_cache_size=1000)
val_class_8 = PartsLoader("stitching/batch_2/val/8", scale=0.3, transform=transform, scaling_variation=0.7, max_cache_size=1000)
val_class_9 = PartsLoader("stitching/batch_2/val/9", scale=0.3, transform=transform, scaling_variation=0.7, max_cache_size=1000)
val_negative_samples = PartsLoader(
    "stitching/batch_2/val/misled", 
    scale=0.3,
    transform=transform, 
    scaling_variation=0.9
)


# Initialize YOLOv8 generator
generator = YOLOv8Generator(overlap_ratio=0.05)

# Define Stitcher with class proportions
stitcher = Stitcher(
    generator, 
    background, 
    {
        "0": [class_0, 0.1],
        "1": [class_1, 0.1],
        "2": [class_2, 0.1],
        "3": [class_3, 0.1],
        "4": [class_4, 0.1],
        "5": [class_5, 0.1],
        "6": [class_6, 0.1],
        "7": [class_7, 0.1],
        "8": [class_8, 0.1],
        "9": [class_9, 0.1],
        "_": [negative_samples, 0.4],
    }, 
    parts_per_image=30
)

val_stitcher = Stitcher(
    generator, 
    val_background, 
    {
        "0": [val_class_0, 0.1],
        "1": [val_class_1, 0.1],
        "2": [val_class_2, 0.1],
        "3": [val_class_3, 0.1],
        "4": [val_class_4, 0.1],
        "5": [val_class_5, 0.1],
        "6": [val_class_6, 0.1],
        "7": [val_class_7, 0.1],
        "8": [val_class_8, 0.1],
        "9": [val_class_9, 0.1],
        "_": [val_negative_samples, 0.7],
    }, 
    parts_per_image=100
)

empty_stitcher = Stitcher(
    generator, 
    background, 
    {
        "_": [negative_samples, 1.0],
    }, 
    parts_per_image=100
)

empty_val_stitcher = Stitcher(
    generator, 
    val_background, 
    {
        "_": [val_negative_samples, 1.0],
    }, 
    parts_per_image=100
)

# Generate datasets
stitcher.execute(
    7000, 
    "stitched/batch_2", 
    "train_1", 
    train_or_val=True,
    perimeter_end=(1280, 720)
)

empty_stitcher.execute(
    3000, 
    "stitched/batch_2", 
    "train_empty", 
    train_or_val=True,
    perimeter_end=(1280, 720)
)

# Generate val
val_stitcher.execute(
    3000, 
    "stitched/batch_2", 
    "val_1", 
    train_or_val=False,
    perimeter_end=(1280, 720)
)

empty_val_stitcher.execute(
    2000, 
    "stitched/batch_2", 
    "val_empty", 
    train_or_val=False,
    perimeter_end=(1280, 720)
)
