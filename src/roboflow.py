import os
import requests
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional
import logging
from tqdm import tqdm
import zipfile

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def download_dataset(url: str, output_dir: str = ".", export_format: str = "yolov11", extract: bool = True) -> str:
    """
    Download a dataset from Roboflow Universe.
    
    Args:
        url: The Roboflow Universe URL (e.g., "https://universe.roboflow.com/workspace/project")
        output_dir: Directory to save the dataset (default: current directory)
        export_format: Format to export (default: "yolov11")
        extract: If True, automatically extract the zip file and return directory path
    
    Returns:
        Path to the downloaded dataset file or directory
        
    Raises:
        ValueError: If API key is not found or URL is invalid
        requests.RequestException: If download fails
    """
    logger.info(f"Starting download from: {url}")
    logger.info(f"Export format: {export_format}")
    
    # Validate inputs
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError("ROBOFLOW_API_KEY environment variable not found")
    
    if not url.startswith("https://universe.roboflow.com"):
        raise ValueError("Invalid Roboflow Universe URL")
    
    # Extract project info and create filename
    url_parts = url.rstrip("/").split("/")
    workspace, project = url_parts[-2], url_parts[-1]
    filename = f"{workspace}_{project}_dataset.zip"
    output_path = Path(output_dir) / filename
    
    logger.info(f"Workspace: {workspace}, Project: {project}")
    logger.info(f"Output file: {filename}")
    
    # Convert Universe URL to API URL
    api_url = url.replace("https://universe.roboflow.com", "https://api.roboflow.com")
    
    try:
        # Get project metadata
        logger.info("Fetching project metadata...")
        response = requests.get(f"{api_url}?api_key={api_key}")
        response.raise_for_status()
        project_data = response.json()
        
        # Get latest version info
        versions = project_data.get("versions", [])
        if not versions:
            raise ValueError("No versions found for this project")
        
        latest_version = versions[0]
        version_number = len(versions)
        logger.info(f"Found {len(versions)} versions, using version {version_number}")
        
        # Check if requested format is available, with fallback logic
        available_exports = latest_version.get("exports", [])
        
        # Try the requested format first
        if export_format in available_exports:
            final_format = export_format
            logger.info(f"Export format '{export_format}' is available")
        # Fallback to yolov8 if yolov11 is not available
        elif export_format == "yolov11" and "yolov8" in available_exports:
            final_format = "yolov8"
            logger.warning(f"Export format '{export_format}' not available, falling back to 'yolov8'")
        else:
            raise ValueError(f"Export format '{export_format}' not available and no suitable fallback found. "
                           f"Available formats: {', '.join(available_exports)}")
        
        # Get download link
        export_url = f"{api_url}/{version_number}/{final_format}?api_key={api_key}"
        logger.info("Getting download link...")
        
        response = requests.get(export_url)
        response.raise_for_status()
        export_data = response.json()
        
        download_url = export_data["export"]["link"]
        logger.info("Starting dataset download...")
        
        # Download and save the dataset with progress bar
        with requests.get(download_url, stream=True) as response:
            response.raise_for_status()
            
            # Get file size for progress bar
            total_size = int(response.headers.get('content-length', 0))
            
            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, "wb") as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        logger.info(f"Dataset downloaded successfully to: {output_path}")
        
        if extract:
            # Extract the zip file
            extract_dir = str(output_path).replace('.zip', '')
            
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            logger.info(f"Dataset extracted to: {extract_dir}")
            return extract_dir
        
        return str(output_path)
        
    except requests.RequestException as e:
        logger.error(f"Download failed: {e}")
        raise requests.RequestException(f"Failed to download dataset: {e}")
    except KeyError as e:
        logger.error(f"API response error: missing key {e}")
        raise ValueError(f"Unexpected API response format: missing key {e}")
