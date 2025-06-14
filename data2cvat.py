import os
import shutil

"""
This script move and rename image and labels files from source directory and prepare them in the correct format to be uploaded up to CVAT.

Parameters:
    -----------
    src_dir : Path to you source directory for image and labels.
        
    classes : List of class names representing each object categories. (Should correspond to YOLO's COCO.yaml file input)

    subset : Subset indicating if the files belong to the training set or validation set. Allowed values are 'Train' or 'Validation'.
        
Functionality:
    --------------
    1. Directory Set up:
       - The source directory expects to have 2 subfolders: images - containing image files, labels - corresponding files in .txt format.
       - The destination directory is generated based on the source directory name with and added "_cvat". 
       - Destination directory contains an additional subfolder depending on the subset (Train or Validation), e.g. obj_Train_data or obj_Validation_data 
    
    2. File Renaming and Moving:
        - Checks for subfolders in the images folder
        - If subfolders exists, the images and labels are copied and renamed using the format:
            {subfolders_name}_frame{i}.jpg and {subfolders_name}_frame{i}.txt 
        - If no subfolder exists, images and labels are copied without renaming.
    
    3. Creating .txt Files:
        - A Train.txt or Validation.txt file is created inside the desitnation folder, depending on the subset.
        - The file contains all image and text files, formatted to match the CVAT expected structure 
    
    4. CVAT Config Files:
        - obj.names: A file containing all class names, one per line.
        - obj.data: A file that specifies the number of classes, the path to the training or validation .txt file, the path to obj.names, and a backup directory.

    5. Creating a ZIP Archive:
        - After organizing all files and generating configuration files, the folder is zipped for uploading to CVAT 

Example Usage:
    --------------
    >>> move_and_rename_files(src_dir, None, classes, subset)

Notes:
    -------------
    - Source directory has to be in this format:

        src_dir/
        ├── images/
        │   ├── subfolder1/
        │   │   ├── img1.jpg
        │   │   ├── img2.jpg
        ├── labels/
        │   ├── subfolder1/
        │   │   ├── img1.txt
        │   │   ├── img2.txt
    
    - The output zip'd folder structure will be in this format suitable for CVAT uploading:

        dest_dir/
        ├── obj_Train_data/
        │   ├── image1.jpg
        │   ├── image1.txt
        ├── Train.txt
        ├── obj.names
        ├── obj.data
        ├── backup/
        └── dest_dir.zip
    
    - Input Assumptions:
        - The source directory contains images and labels subdirectories.
        - Each image in the images subdirectory has a corresponding annotation file in the labels subdirectory.

    -  Naming Convention:
        - The files are renamed if there are subfolders within the images folder, otherwise, they retain their original names.
"""

src_dir = 'data/taku-3grva_bus-detextion_dataset'  # Replace with the path to your source directory
classes = ["bus", "bus 151", "bus 154", "bus 184", "bus 74"]  # Replace with your class names
subset = 'Train'  # Choose between 'Train' or 'Validation'

def move_and_rename_files(src_dir, dest_dir, classes, subset):
    images_subfolder = os.path.join(src_dir, 'images')
    labels_subfolder = os.path.join(src_dir, 'labels')

    # Extract the base name of the src_dir and append '_cvat'
    src_dir_name = os.path.basename(src_dir.rstrip('/'))
    dest_dir_name = f"{src_dir_name}_cvat"
    dest_dir = os.path.join(os.path.dirname(src_dir), dest_dir_name)

    # Create destination subfolder if it doesn't exist
    dest_subfolder = os.path.join(dest_dir, f"obj_{subset}_data")
    os.makedirs(dest_subfolder, exist_ok=True)

    # Create Train.txt or Validation.txt file
    txt_file = open(os.path.join(dest_dir, f"{subset}.txt"), "w")

    # Check if there are subfolders in the images_subfolder
    subfolders = [f for f in os.listdir(images_subfolder) if os.path.isdir(os.path.join(images_subfolder, f))]

    if subfolders:
        for folder_name in subfolders:
            src_images_folder = os.path.join(images_subfolder, folder_name)
            src_labels_folder = os.path.join(labels_subfolder, folder_name)

            images = sorted(os.listdir(src_images_folder))
            labels = sorted(os.listdir(src_labels_folder))

            for i, (image, label) in enumerate(zip(images, labels), start=1):
                src_image = os.path.join(src_images_folder, image)
                src_label = os.path.join(src_labels_folder, label)

                dest_image = os.path.join(dest_subfolder, f"{os.path.basename(folder_name)}_frame{i}.jpg")
                dest_label = os.path.join(dest_subfolder, f"{os.path.basename(folder_name)}_frame{i}.txt")

                # Copy and rename the image and label files
                shutil.copy2(src_image, dest_image)
                shutil.copy2(src_label, dest_label)

                # Write the path of the image to Train.txt or Validation.txt
                txt_file.write(f"data/obj_{subset}_data/{os.path.basename(folder_name)}_frame{i}.jpg\n")
    else:
        images = sorted(os.listdir(images_subfolder))
        labels = sorted(os.listdir(labels_subfolder))

        for image, label in zip(images, labels):
            src_image = os.path.join(images_subfolder, image)
            src_label = os.path.join(labels_subfolder, label)

            dest_image = os.path.join(dest_subfolder, image)
            dest_label = os.path.join(dest_subfolder, label)

            # Copy the image and label files without renaming
            shutil.copy2(src_image, dest_image)
            shutil.copy2(src_label, dest_label)

            # Write the path of the image to Train.txt or Validation.txt
            txt_file.write(f"data/obj_{subset}_data/{image}\n")

    # Close Train.txt or Validation.txt file
    txt_file.close()

    # Create obj.names file
    with open(os.path.join(dest_dir, "obj.names"), "w") as f:
        for cls in classes:
            f.write(f"{cls}\n")

    # Create obj.data file
    with open(os.path.join(dest_dir, "obj.data"), "w") as f:
        f.write(f"classes = {len(classes)}\n")
        f.write(f"{subset} = data/{subset}.txt\n")
        f.write(f"names = data/obj.names\n")
        f.write(f"backup = backup/\n")

    # Create a zip archive of the final product
    shutil.make_archive(dest_dir, 'zip', dest_dir)

move_and_rename_files(src_dir, None, classes, subset)