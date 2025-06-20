import cv2
import os
from ultralytics import YOLO

bus_model = YOLO("models/best_16jun_3.pt")
number_model = YOLO("models/number-1.pt")

def inference(image_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    image = cv2.imread(image_path)
    bus_results = bus_model.predict(image) 
    count = 0
    for box in bus_results[0].boxes.xyxy:
        x1, y1, x2, y2 = box.int()
        crop = image[y1:y2, x1:x2]
        number_results = number_model.predict(crop)
        for i, box in enumerate(number_results[0].boxes.xyxy):
            x1, y1, x2, y2 = box.int()
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            # Draw bounding box
            cv2.rectangle(crop, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # Draw number
            cv2.putText(crop, str(int(number_results[0].boxes.cls[i].item())), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imwrite(output_folder + f"/{count}.jpg", crop)
        count += 1

inference("data/Sample_Bus/386.jpg", "data/test")
