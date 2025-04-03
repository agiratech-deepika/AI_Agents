import streamlit as st
import cv2
import tempfile
import torch
from ultralytics import YOLO
import numpy as np
import os
import asyncio

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
    # # Save file temporarily
    # with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
    #     temp_video.write(video_file.read())
    #     video_path = temp_video.name
    # Save the uploaded video in the temp directory
    video_path = os.path.join(temp_folder, "input_video.mp4")
    with open(video_path, 'wb') as f:
        f.write(video_file.read())

    st.video(video_path)

    if st.button("🔍 Detect Objects"):
        st.write("Processing video... This may take some time ⏳")

        # OpenCV Video Processing
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # output_path = "output1.mp4"
        # out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

        # # Save the processed video to a temporary location
        # with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_output:
        #     output_path = temp_output.name
        #     out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

        # Save the processed video to a specific location in 'temp' folder
        output_path = os.path.join(temp_folder, "output_video.mp4")
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

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
            cv2.waitKey(1)  # Fix for frame processing issue

        cap.release()
        out.release()

        # st.success("Object detection completed! 🎯")
        # st.video(output_path)  # Display the processed video

        # Debugging: Check if the output file exists
        if os.path.exists(output_path):
            st.write(f"Processed video saved to: {output_path}")
        else:
            st.error("Output video file not found!")

        st.success("Object detection completed! 🎯")
        st.video(output_path)  # Display the processed video

        # Display video
        # Open the processed video in binary mode and pass it to Streamlit
        # with open(output_path, "rb") as video_file:
        #     st.video(video_file)

        #  # Embed the video directly using HTML
        # video_html = f"""
        # <video width="600" height="400" controls>
        #     <source src="file:///{output_path}" type="video/mp4">
        #     Your browser does not support the video tag.
        # </video>
        # """
        # st.markdown(video_html, unsafe_allow_html=True)

        # Cleanup
        os.remove(video_path)
        # os.remove(output_path)