# Download the COCO dataset
from pathlib import Path
import logging
from tqdm import tqdm

from ultralytics.utils.downloads import download

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_coco_dataset(dir: str):
    logger.info(f"Starting COCO dataset download to: {dir}")
    
    # Download labels
    segments = True  # segment or box labels
    dir = Path(dir)  # dataset root dir
    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
    urls = [url + ("coco2017labels-segments.zip" if segments else "coco2017labels.zip")]  # labels
    
    logger.info("Downloading COCO labels...")
    download(urls, dir=dir.parent)

    # Download data
    urls = [
        "http://images.cocodataset.org/zips/train2017.zip",  # 19G, 118k images
        "http://images.cocodataset.org/zips/val2017.zip",  # 1G, 5k images
        "http://images.cocodataset.org/zips/test2017.zip",  # 7G, 41k images (optional)
    ]
    
    logger.info("Downloading COCO images (this may take a while)...")
    logger.info("Files to download: train2017.zip (~19GB), val2017.zip (~1GB), test2017.zip (~7GB)")
    
    download(urls, dir=dir / "images", threads=3)
    logger.info("COCO dataset download completed!")