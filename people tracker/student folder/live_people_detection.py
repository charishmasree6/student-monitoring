import torch
import cv2
import numpy as np
import time
import sys

# Add yolov5 directory to path
sys.path.append(r"C:\yolov5")
from models.common import DetectMultiBackend
from utils.datasets import LoadStreams
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

# Load model
device = select_device('')
model = DetectMultiBackend(r"C:\yolov5\yolov5s.pt", device=device)
model.eval()

# Load webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (640, 640))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img)
    pred = non_max_suppression(pred)[0]

    if pred is not None and len(pred):
        for *xyxy, conf, cls in pred:
            label = f'Person {conf:.2f}'
            frame = cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
            frame = cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Live People Detection', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
