# !/bin/bash

path=src/yolo_bot

# echo rosrun yolo_bot detect_signs.py --weights ./${path}/runs/train/yolov5s_results/weights/best.pt --img 960 --conf 0.8 --source   ./${path}/data/videos/x.mkv --view-img

rosrun yolo_bot detect_signs.py --weights ./${path}/runs/train/yolov5s_results/weights/best.pt --img 960 --conf 0.8 --source   ./${path}/data/videos/right1.mp4 --view-img
