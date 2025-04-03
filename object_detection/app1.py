import streamlit as st
import cv2
import os
import time
import shutil
import torch
from ultralytics import YOLO
import numpy as np

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Use "yolov8s.pt" for better accuracy

# Streamlit UI
st.set_page_config(
    page_title="Accident Detection Video Analyzer",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Accident Detection & Object Tracking")
st.markdown("Upload a video to detect objects and track accidents.")

# Folder where the video and output will be saved
project_folder = r"C:\AI Agents\AI_Agents\object_detection"
temp_folder = os.path.join(project_folder, "temp")

# Ensure that the temp folder exists
if not os.path.exists(temp_folder):
    os.makedirs(temp_folder)

# File Uploader
video_file = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov'])

if video_file:
    # Save the uploaded video
    video_path = os.path.join(temp_folder, "input_video.mp4")
    with open(video_path, 'wb') as f:
        f.write(video_file.read())

    st.video(video_path)  # Show uploaded video

    if st.button("🔍 Detect Objects"):
        st.write("Processing video... This may take some time ⏳")

        # OpenCV Video Processing
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Use 'avc1' for H.264 encoding

        # Output video path
        output_path = os.path.join(temp_folder, "output_video.mp4")
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height), isColor=True)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform object detection
            results = model(frame)

            # Draw bounding boxes
            for r in results:
                for box in r.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box)  # Bounding box coordinates
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            out.write(frame)
            cv2.waitKey(1)

        cap.release()
        out.release()

        # Ensure video file is written before displaying
        time.sleep(2)  

        # Copy the output video to a new file before displaying
        final_output_path = os.path.join(temp_folder, "final_output_video.mp4")
        shutil.copy(output_path, final_output_path)

        # Verify file existence and size
        if os.path.exists(final_output_path) and os.path.getsize(final_output_path) > 0:
            st.success("Object detection completed! 🎯")
            st.video(final_output_path)  # Display processed video
        else:
            st.error("Error: Output video file is empty or not found!")

        # Cleanup
        os.remove(video_path)
