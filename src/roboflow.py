
import os
import requests
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional

# Load environment variables
load_dotenv()

def download_dataset(url: str, output_dir: str = ".", export_format: str = "yolov11") -> str:
    """
    Download a dataset from Roboflow Universe.
    
    Args:
        url: The Roboflow Universe URL (e.g., "https://universe.roboflow.com/workspace/project")
        output_dir: Directory to save the dataset (default: current directory)
        export_format: Format to export (default: "yolov11")
    
    Returns:
        Path to the downloaded dataset file
        
    Raises:
        ValueError: If API key is not found or URL is invalid
        requests.RequestException: If download fails
    """
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
    
    # Convert Universe URL to API URL
    api_url = url.replace("https://universe.roboflow.com", "https://api.roboflow.com")
    
    try:
        # Get project metadata
        print(f"Fetching project metadata for {workspace}/{project}...")
        response = requests.get(f"{api_url}?api_key={api_key}")
        response.raise_for_status()
        project_data = response.json()
        
        # Get latest version info
        versions = project_data.get("versions", [])
        if not versions:
            raise ValueError("No versions found for this project")
        
        latest_version = versions[0]
        version_number = len(versions)
        
        # Check if requested format is available
        available_exports = latest_version.get("exports", [])
        if export_format not in available_exports:
            raise ValueError(f"Export format '{export_format}' not available. "
                           f"Available formats: {', '.join(available_exports)}")
        
        print(f"Found export format '{export_format}' in version {version_number}")
        
        # Get download link
        export_url = f"{api_url}/{version_number}/{export_format}?api_key={api_key}"
        print(f"Getting download link...")
        
        response = requests.get(export_url)
        response.raise_for_status()
        export_data = response.json()
        
        download_url = export_data["export"]["link"]
        print(f"Downloading dataset from: {download_url}")
        
        # Download and save the dataset
        with requests.get(download_url, stream=True) as response:
            response.raise_for_status()
            
            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        print(f"Dataset downloaded successfully to: {output_path}")
        return str(output_path)
        
    except requests.RequestException as e:
        raise requests.RequestException(f"Failed to download dataset: {e}")
    except KeyError as e:
        raise ValueError(f"Unexpected API response format: missing key {e}")
