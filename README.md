# Apple 3D Detection & Localization
Project that train YOLO detection models on depth thresholded images to obtain rough 3D location of each foreground apples.
## Results

| Model  | Speed<br><sup>GTX 1060<br>(ms) | Precision | Recall | F1 | mAP50 | mAP50-95 |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| YOLOv8n | 19.1  | 0.925 | 0.864 | 0.893 | 0.916 | 0.69 |
| YOLOv8s | 22.8  | 0.921 | 0.864 | 0.892 | 0.926 | 0.704 |
| YOLOv8m | 44.8 | 0.931 | 0.852 | 0.890 | 0.924 | 0.711 |
| YOLOv8l | 66.0 | 0.919 | 0.871 | 0.894 | 0.925 | 0.711 |
| YOLOv8x | 96.8 | 0.914 | 0.856 | 0.884 | 0.919 | 0.711 |
| YOLOv9c | 58.0 | 0.927 | 0.867 | 0.896 | 0.929 | 0.718 |
| YOLOv9e | 103.0 | 0.937 | 0.865 | 0.900 | 0.928 | 0.720 |

## Approach
![approach](https://github.com/eugeneyjy/apple-3d-localization/assets/46506744/318976e2-3cd1-463c-98e4-55dd4880cc47)

## Example
![demo](https://github.com/eugeneyjy/apple-3d-localization/assets/46506744/aed62b9a-3c05-47c5-bbe4-a0f36ae731bb)

## Install Prerequisites
Install [PyTorch>=1.8](https://pytorch.org/get-started/locally/) based on CUDA version.  
Follow guide on [pyk4a](https://github.com/etiennedub/pyk4a) and install [Azure SDK](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/docs/usage.md).

## Dataset
https://universe.roboflow.com/apple-detection-localization

## Models
https://drive.google.com/drive/folders/19eW2avN8TmLwfF5wTqMihWmJXaJe1ctT?usp=drive_link


