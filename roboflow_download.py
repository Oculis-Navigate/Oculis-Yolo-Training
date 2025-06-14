import os
from src.invert import invert_yolo_data, invert_yolo_data_from_roboflow
from src.roboflow import download_dataset

ROBOFLOW_PATHS = [
    "https://universe.roboflow.com/taku-3grva/bus-detextion",
    "https://universe.roboflow.com/test-project-csgdb/bus-route-number-testdataset",
    "https://universe.roboflow.com/ram-khlww/bus-emts1-chcch",
]

output_paths = []

for path in ROBOFLOW_PATHS:
    # Download and extract the dataset (returns directory path, not zip path)
    dataset_dir = download_dataset(path, output_dir="data", extract=True)
    
    print(f"Dataset extracted to: {dataset_dir}")
    
    # Now run invert on the extracted directory
    invert_yolo_data(dataset_dir)
    
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    invert_yolo_data_from_roboflow(yaml_path)
    output_paths.append(dataset_dir)

print("All datasets processed:")
print(output_paths)

