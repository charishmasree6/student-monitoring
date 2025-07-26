import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(layout="wide")
st.title("📊 People Detection + Heatmap Overlay")

# Upload video
video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if video_file is not None:
    tfile = open("temp_video.mp4", "wb")
    tfile.write(video_file.read())

    # Load model
    model = YOLO("yolov5su.pt")

    # Load video
    cap = cv2.VideoCapture("temp_video.mp4")

    # Get frame size
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    heatmap = np.zeros((height, width), dtype=np.float32)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Streamlit placeholders
    video_placeholder = st.empty()
    heatmap_placeholder = st.empty()
    count_placeholder = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        people_count = 0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if model.names[cls] == "person":
                    people_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    heatmap[y1:y2, x1:x2] += 1

        # Normalize heatmap
        heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_uint8 = heatmap_norm.astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Overlay
        overlay = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)

        # Convert for Streamlit display
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        overlay_img = Image.fromarray(overlay_rgb)

        # Update dashboard
        video_placeholder.image(overlay_img, caption="Live Detection + Heatmap", use_column_width=True)
        count_placeholder.markdown(f"### 👥 People detected in frame: {people_count}")

    cap.release()
    st.success("✅ Video processing finished!")
