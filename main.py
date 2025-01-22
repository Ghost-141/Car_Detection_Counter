from ultralytics import YOLO
import cv2
import math
import cvzone
import numpy as np
from sort import *
import torch

cap = cv2.VideoCapture("Car_Detection_Counter/car2lane.mp4")
model = YOLO("Car_Detection_Counter/yolov8s.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
mask = cv2.imread("Car_Detection_Counter/mask.png")  

tracker_lane1 = Sort(max_age=20, min_hits=3, iou_threshold=0.4)
tracker_lane2 = Sort(max_age=20, min_hits=3, iou_threshold=0.4)

class_name = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", 
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", 
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", 
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", 
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", 
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", 
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", 
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", 
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", 
    "toothbrush"]

limit1 = [264, 807, 891, 808]
limit2 = [1101, 710, 1620, 711]

id1 = int(0)
id2 = int(0)

total_count1 = []
total_count2 = []    

while True:
    success, img = cap.read()
    if not success:
        break
    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion, stream=False)

    detections1 = np.empty((0, 5))
    detections2 = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            current_cls = class_name[cls]

            if (current_cls == "car" or current_cls == "bus" or current_cls == "truck") and conf > 0.5:
                cx, cy = x1 + w // 2, y1 + h // 2
                currentArray = np.array([x1, y1, x2, y2, conf])
                cvzone.cornerRect(img, (x1,y1,w,h), l=8, rt=3) 
                label = f"{current_cls} {conf:.2f}"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)


                if limit1[0] < cx < limit1[2]:  
                    detections1 = np.vstack((detections1, currentArray))
                elif limit2[0] < cx < limit2[2]:
                    detections2 = np.vstack((detections2, currentArray))

    resultsTrack1 = tracker_lane1.update(detections1)
    resultsTrack2 = tracker_lane2.update(detections2)

    cv2.line(img, (limit1[0], limit1[1]), (limit1[2], limit1[3]), (0, 0, 255), 5)
    cv2.line(img, (limit2[0], limit2[1]), (limit2[2], limit2[3]), (0, 0, 255), 5)

    for result in resultsTrack1:
        x1, y1, x2, y2, id1 = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2

        if limit1[1] - 15 < cy < limit1[1] + 15:
            if total_count1.count(id1) == 0:
                total_count1.append(id1)
                cv2.line(img, (limit1[0], limit1[1]), (limit1[2], limit1[3]), (0, 255, 0), 5)

    for result in resultsTrack2:
        x1, y1, x2, y2, id2 = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2

        if limit2[1] - 15 < cy < limit2[1] + 15:
            if total_count2.count(id2) == 0:
                total_count2.append(id2)
                cv2.line(img, (limit2[0], limit2[1]), (limit2[2], limit2[3]), (0, 255, 0), 5)

    cv2.putText(img,  f'Lane 1 Count: {len(total_count1)}', (20, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (20, 50, 255), 3)
    cv2.putText(img, f'Lane 2 Count: {len(total_count2)}', (1460, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (50, 50, 255), 3)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()