import json
from ultralytics import YOLO

if __name__ == "__main__":
    with open("./train_config.json") as f:
        config = json.load(f)

    # Load a model
    model = YOLO(config["model"])

    # Train the model
    results = model.train(data=config["data"],
                          epochs=config["epochs"], 
                          imgsz=640, 
                          workers=2,
                          name=config["name"])