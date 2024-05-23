import argparse
import numpy as np
import pathlib
import cv2
import torch

from ultralytics import YOLO
from pyk4a import PyK4APlayback

# Function to get {x,y} index inside a 2D depth arr that associate with the median depth
# i.e. arr[median_y, median_x] == median of arr
def argmedian(arr: np.ndarray):
    # Store original indices of the arr
    ori_inds = np.arange(arr.size)
    
    # Store the indices of depth > 0 in the original array
    filtered_ori_inds = ori_inds[arr.flatten() > 0]
    # Get only elements with depth > 0 from the original array
    filtered_depth = arr[arr > 0]

    # If all depth is <= 0, return None
    if filtered_depth.size == 0:
        return None
    
    mid = filtered_depth.size//2
    # Get median of depth > 0
    filtered_median_idx = np.argpartition(filtered_depth, mid, axis=None)[mid]
    # Get the flatten original index
    ori_median_idx = filtered_ori_inds[filtered_median_idx]
    # Convert flatten index back to {x,y} indicies
    median_y, median_x = np.unravel_index(ori_median_idx, arr.shape)
    return median_x, median_y

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", required=True, help="Path to model for prediction")
    parser.add_argument("-file", required=True, help="Path to mkv file to be inferred")
    parser.add_argument("--nosave", action="store_true", required=False, help="Flag to specify if don't want to save video result")
    args = parser.parse_args()
    model_name = pathlib.Path(args.model).stem
    filename = pathlib.Path(args.file).stem
    box_model = YOLO(args.model)  # load a custom model

    save = not args.nosave
    if save:
        output = cv2.VideoWriter(f"{model_name}_{filename}.avi", cv2.VideoWriter_fourcc(*'XVID'), 30, (1920, 1080))
    playback = PyK4APlayback(args.file)
    playback.open()
    print(f"Record length: {playback.length / 1000000: 0.2f} sec")

    while True:
        try:
            capture = playback.get_next_capture()

            depth = capture.transformed_depth
            if depth is not None:
                depth_3d = np.tile(depth[:, :, np.newaxis], (1, 1, 3))
            else:
                continue

            bgr_image = cv2.imdecode(capture.color, cv2.IMREAD_COLOR)
            bgr_image = np.where((depth_3d <= 1200) & (depth_3d != 0), bgr_image, 0)

            box_results =  box_model(bgr_image, conf=0.4, iou=0.6)
            result_3d = []

            if len(box_results[0].boxes) != 0:
                xyxy_boxes = box_results[0].boxes.xyxy
                for box in xyxy_boxes:
                    x1 = int(box[0])
                    x2 = int(box[2])
                    y1 = int(box[1])
                    y2 = int(box[3])
                    depth_box = depth[y1:y2, x1:x2]
                    median = argmedian(depth_box)
                    if median == None:
                        print(f"depth_box: {box} is all zeros")
                        continue
                    median_x, median_y = median[0], median[1]
                    bgr_image = cv2.rectangle(bgr_image, (x1, y1), (x2, y2), (255, 0, 0), 5) 
                    bgr_image = cv2.circle(bgr_image, (median_x+x1, median_y+y1), 5, (0, 255, 255), -1)
                    result_3d.append((median_x, median_y, depth_box[median_y, median_x]))

            cv2.imshow('Image', bgr_image)
            if save:
                output.write(bgr_image)
            key = cv2.waitKey(10)
            if key != -1:
                break
        except EOFError:
            break
        
    if save:
        output.release() 
    cv2.destroyAllWindows()
    playback.close()