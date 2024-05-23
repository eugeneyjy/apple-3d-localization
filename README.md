# Apple 3D Detection & Localization
Project that train YOLO detection models on depth thresholded images to obtain rough 3D location of each foreground apples.
## Results

| Model  | Speed<br><sup>GTX 1060<br>(ms) | Precision | Recall | F1 | mAP50 | mAP50-95 |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| YOLOv8n | 19.1  | 0.937 | 0.847 | 0.890 | 0.913 | 0.696 |
| YOLOv8s | 22.8  | 0.912 | 0.875 | 0.893 | 0.925 | 0.711 |
| YOLOv8m | 44.8 | 0.919 | 0.866 | 0.892 | 0.927 | 0.707 |
| YOLOv8l | 66.0 | 0.927 | 0.853 | 0.888 | 0.922 | 0.708 |
| YOLOv8x | 96.8 | 0.919 | 0.853 | 0.885 | 0.921 | 0.714 |
| YOLOv9c | 58.0 | 0.927 | 0.867 | 0.896 | 0.929 | 0.718 |
| YOLOv9e | 103.0 | 0.937 | 0.865 | 0.900 | 0.928 | 0.720 |

## Approach
![approach](https://github.com/eugeneyjy/apple-3d-localization/assets/46506744/318976e2-3cd1-463c-98e4-55dd4880cc47)

## Example
![demo](https://github.com/eugeneyjy/apple-3d-localization/assets/46506744/aed62b9a-3c05-47c5-bbe4-a0f36ae731bb)

## Dataset
https://universe.roboflow.com/apple-detection-localization

## Models
https://drive.google.com/drive/folders/19eW2avN8TmLwfF5wTqMihWmJXaJe1ctT?usp=drive_link


