from ultralytics import YOLO
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(model_name: str, config_path: str, data_path: str):
    logger.info(f"Starting training with model: {model_name}")
    logger.info(f"Config: {config_path}, Data: {data_path}")
    
    model = YOLO(model_name)
    
    # Training progress is handled by ultralytics internally
    model.train(data=data_path, cfg=config_path)
    
    output_path = f"{model_name}.pt"
    model.save(output_path)
    logger.info(f"Model saved to: {output_path}")
    
    return model    

def test_model(model_name: str, data_path: str, save_crops: bool = False):
    logger.info(f"Starting prediction with model: {model_name}")
    logger.info(f"Data: {data_path}, Save crops: {save_crops}")
    
    model = YOLO(model_name)
    
    # Prediction progress is handled by ultralytics internally
    results = model.predict(data_path, save=True, vid_stride=10, save_crop=save_crops)
    logger.info(f"Prediction completed. Results saved.")
    
    return model
