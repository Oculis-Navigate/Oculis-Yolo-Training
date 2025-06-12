import os
from pathlib import Path
from typing import Dict, List, Union
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def remap_labels(input_folder: Union[str, Path], mapping: Dict[int, List[int]]) -> Dict[str, int]:
    """
    Remap class IDs in YOLO label files.
    
    Args:
        input_folder: Path to folder containing YOLO label .txt files
        mapping: Dict where keys are new class IDs, values are lists of old class IDs to map
                Example: {0: [5], 1: [2, 3]} maps old class 5→0, old classes 2,3→1
    
    Returns:
        Dict with stats: {'files_processed': int, 'labels_remapped': int, 'labels_removed': int}
    """
    input_path = Path(input_folder)
    if not input_path.exists():
        raise ValueError(f"Folder not found: {input_folder}")
    
    # Create reverse mapping for lookup
    reverse_mapping = {}
    for new_class, old_classes in mapping.items():
        for old_class in old_classes:
            reverse_mapping[old_class] = new_class
    
    label_files = list(input_path.glob("*.txt"))
    if not label_files:
        print(f"No .txt files found in {input_folder}")
        return {'files_processed': 0, 'labels_remapped': 0, 'labels_removed': 0}
    
    stats = {'files_processed': 0, 'labels_remapped': 0, 'labels_removed': 0}
    
    for file_path in tqdm(label_files, desc="Processing labels"):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        updated_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            try:
                class_id = int(parts[0])
                if class_id in reverse_mapping:
                    new_class_id = reverse_mapping[class_id]
                    updated_lines.append(f"{new_class_id} {' '.join(parts[1:])}\n")
                    stats['labels_remapped'] += 1
                else:
                    stats['labels_removed'] += 1
            except ValueError:
                continue
        
        with open(file_path, 'w') as f:
            f.writelines(updated_lines)
        
        stats['files_processed'] += 1
    
    print(f"Done! Processed {stats['files_processed']} files, "
          f"remapped {stats['labels_remapped']} labels, "
          f"removed {stats['labels_removed']} labels")
    
    return stats

def validate_yolo_labels(folder_path: Union[str, Path], expected_classes: List[int]) -> Dict[str, List[str]]:
    """
    Validate YOLO label files to check for invalid class IDs.
    
    Args:
        folder_path: Path to folder containing .txt label files
        expected_classes: List of valid class IDs
        
    Returns:
        Dictionary with validation results: {'valid_files': [...], 'invalid_files': [...], 'errors': [...]}
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")
    
    results = {'valid_files': [], 'invalid_files': [], 'errors': []}
    
    for file_path in folder.glob("*.txt"):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            
            file_valid = True
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 5:
                    results['errors'].append(f"{file_path.name}:{line_num} - Invalid format")
                    file_valid = False
                    continue
                
                try:
                    class_id = int(parts[0])
                    if class_id not in expected_classes:
                        results['errors'].append(f"{file_path.name}:{line_num} - Invalid class ID: {class_id}")
                        file_valid = False
                except ValueError:
                    results['errors'].append(f"{file_path.name}:{line_num} - Non-integer class ID: {parts[0]}")
                    file_valid = False
            
            if file_valid:
                results['valid_files'].append(str(file_path))
            else:
                results['invalid_files'].append(str(file_path))
                
        except OSError as e:
            results['errors'].append(f"{file_path.name} - File read error: {e}")
            results['invalid_files'].append(str(file_path))
    
    return results