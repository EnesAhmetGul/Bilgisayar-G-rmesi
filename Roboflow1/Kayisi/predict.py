
import torch
import cv2
from pathlib import Path

def predict_images(image_dir, model_path='best.pt'):
    # Load YOLOv5 model from the specified model path
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
    
    # Go through each image in the directory and predict
    for image_path in Path(image_dir).glob("*.jpg"):
        img = cv2.imread(str(image_path))
        results = model(img)
        results.show()  # Display the results
        print(f"Predictions for {image_path.name}:")
        print(results.pandas().xyxy[0])  # Print predictions as DataFrame

if __name__ == "__main__":
    image_dir = './Odev1_dataset/test/images'  # Replace with your directory of test images
    model_path = './Odev1_dataset/weights/best.pt'  # Path to the trained model
    predict_images(image_dir=image_dir, model_path=model_path)
