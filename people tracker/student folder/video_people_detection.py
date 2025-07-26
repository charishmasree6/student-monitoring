import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv5 model
model = YOLO('yolov5s.pt')

# Open the video
cap = cv2.VideoCapture('people_vd.mp4')

# Get video dimensions
ret, frame = cap.read()
height, width = frame.shape[:2]

# Create a blank heatmap (grayscale)
heatmap = np.zeros((height, width), dtype=np.float32)

# Reset video to start
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    # Draw detections and accumulate to heatmap
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if model.names[cls] == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Increase heatmap intensity in the bounding box area
                heatmap[y1:y2, x1:x2] += 1

    # Normalize heatmap to range 0-255 and convert to uint8
    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_uint8 = heatmap_norm.astype(np.uint8)

    # Apply a color map (COLORMAP_JET gives a good heat effect)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Overlay heatmap on frame
    overlay = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)

    # Show the result
    cv2.imshow("People Detection + Heatmap", overlay)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
