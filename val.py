import json

from ultralytics import YOLO

if __name__ == "__main__":
    with open("./val_config.json") as f:
        config = json.load(f)

    # Load a model
    model = YOLO(config["model"])

    val_dir = f"val_{config['model'].split('/')[-3]}_{config['data'].split('/')[-2]}"
    save_dir = f"{config['project']}/{val_dir}"

    # Customize validation settings
    validation_results = model.val(data=config["data"],
                                   batch=config["batch"],
                                   conf=config["conf"],
                                   iou=config["iou"],
                                   split=config["split"],
                                   project=config["project"],
                                   name=val_dir)
    
    result_metrics = validation_results.box
    with open(f"{save_dir}/performance.txt", "w") as f:
        f.write("precision,recall,map50,map50-95\n")
        f.write(f"{result_metrics.p[0]},{result_metrics.r[0]},{result_metrics.map50},{result_metrics.map}")