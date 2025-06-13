from ultralytics import YOLO

def train_model(model_name: str, config_path: str, data_path: str):
    model = YOLO(model_name)

    model.train(data=data_path, cfg=config_path)

    model.save(f"{model_name}.pt")

    return model    

def test_model(model_name: str, data_path: str, save_crops: bool = False):
    model = YOLO(model_name)

    model.predict(data_path, save=True, vid_stride=10, save_crop=save_crops)

    return model
