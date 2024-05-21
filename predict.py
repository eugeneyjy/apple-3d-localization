import os
import json
import random

from pathlib import Path
from PIL import Image, ImageDraw 
from ultralytics import YOLO

if __name__ == "__main__":
    with open("./predict_config.json") as f:
        config = json.load(f)

    pred_dir = f"pred_{config['model'].split('/')[-3]}_{config['folder'].split('/')[-3]}"
    save_dir = Path(f"{config['project']}/{pred_dir}")

    if not save_dir.exists():
        os.makedirs(save_dir)

    # Load a model
    model = YOLO(config["model"])

    image_folder = config["folder"]
    images = [f'{image_folder}/{file}' for file in os.listdir(image_folder)]

    if config["random"]:
        random.shuffle(images)
    
    if config["n_images"] < len(images):
        images = images[:config["n_images"]]

    # Run batched inference on a list of images
    results = model(images, iou=config["iou"], conf=config["conf"])  # return a list of Results objects

    # Process results list
    for i, result in enumerate(results):
        # draw bounding box for each detected apples 
        img = Image.open(images[i])
        img1 = ImageDraw.Draw(img)   
        for box in result.boxes.xyxy:
            img1.rectangle(box.tolist(), fill =None, outline ="blue", width=5)
        img.save(f'{save_dir}/result_{i}.jpg')
