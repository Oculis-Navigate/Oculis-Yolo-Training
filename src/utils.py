import os
import shutil
import random
from pathlib import Path

def copy_random_half_files(source_dir, destination_dir, ratio=0.5):
    """
    Copies a random half of the files from source_dir to destination_dir.
    Assumes source_dir contains only files (no subdirectories to copy selectively).
    """
    source_path = Path(source_dir)
    destination_path = Path(destination_dir)

    if not source_path.is_dir():
        print(f"Error: Source directory '{source_path}' does not exist or is not a directory.")
        return

    # Create destination directory if it doesn't exist
    destination_path.mkdir(parents=True, exist_ok=True)

    all_files = [f for f in source_path.iterdir() if f.is_file()]
    if not all_files:
        print(f"No files found in '{source_path}'. Nothing to copy.")
        return

    num_files_to_copy = int(len(all_files) * ratio) # Integer division for half
    
    # Shuffle and select half
    selected_files = random.sample(all_files, num_files_to_copy)

    print(f"Copying {len(selected_files)} out of {len(all_files)} files from '{source_path}' to '{destination_path}'...")

    for file_path in selected_files:
        try:
            shutil.copy2(file_path, destination_path / file_path.name)
            # print(f"  Copied: {file_path.name}") # Uncomment for verbose output
        except Exception as e:
            print(f"  Error copying {file_path.name}: {e}")
    print("Copying complete.")
