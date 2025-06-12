import os
import shutil 


"""

Turn YOLO data organised by train,test,val into being organised by images,labels

"""

def invert_yolo_data(data_path: str):
    """
    Invert YOLO data organised by train, val into being organised by images,labels
    """
    
    # Check if train,test,val folders exist
    if not os.path.exists(os.path.join(data_path, "train")):
        raise ValueError("train folder does not exist")
    if not os.path.exists(os.path.join(data_path, "valid")):
        raise ValueError("valid folder does not exist")
    if not os.path.exists(os.path.join(data_path, "test")):
        raise ValueError("test folder does not exist")
    
    # Create images and labels folders
    images_path = os.path.join(data_path, "images")
    labels_path = os.path.join(data_path, "labels")
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)
    
    # Create train,val folders in images and labels
    train_images_path = os.path.join(images_path, "train")
    train_labels_path = os.path.join(labels_path, "train")
    os.makedirs(train_images_path, exist_ok=True)
    os.makedirs(train_labels_path, exist_ok=True)
    
    valid_images_path = os.path.join(images_path, "valid")
    valid_labels_path = os.path.join(labels_path, "valid")
    os.makedirs(valid_images_path, exist_ok=True)
    os.makedirs(valid_labels_path, exist_ok=True)

    test_images_path = os.path.join(images_path, "test")
    test_labels_path = os.path.join(labels_path, "test")
    os.makedirs(test_images_path, exist_ok=True)
    os.makedirs(test_labels_path, exist_ok=True)
    
    # Rename folders to shift them
    os.rename(os.path.join(data_path, "train", "images"), os.path.join(data_path, "images", "train"))
    os.rename(os.path.join(data_path, "train", "labels"), os.path.join(data_path, "labels", "train"))
    os.rename(os.path.join(data_path, "valid", "images"), os.path.join(data_path, "images", "valid"))
    os.rename(os.path.join(data_path, "valid", "labels"), os.path.join(data_path, "labels", "valid"))
    os.rename(os.path.join(data_path, "test", "images"), os.path.join(data_path, "images", "test"))
    os.rename(os.path.join(data_path, "test", "labels"), os.path.join(data_path, "labels", "test"))
    
    print("Data inverted successfully")

if __name__ == "__main__":
    invert_yolo_data("test/dataset")
