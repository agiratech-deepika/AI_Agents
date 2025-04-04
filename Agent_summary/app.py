import streamlit as st
import cv2
import os
import time
import shutil
import torch
import tempfile
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file, get_file
import google.generativeai as genai
from dotenv import load_dotenv

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

# Load YOLO model
model = YOLO("yolov8n.pt")
CLASS_NAMES = model.names  # COCO class names

# Streamlit UI setup
st.set_page_config(page_title="AI Video Analyzer", page_icon="🎥", layout="wide")
st.title("🚗 AI-Powered Video Analyzer")
st.markdown("Upload a video to detect moving objects, analyze accidents, or summarize content with AI.")

# Initialize AI Agent
@st.cache_resource
def initialize_agent():
    return Agent(name="Video AI Summarizer", model=Gemini(id="gemini-2.0-flash-exp"), tools=[DuckDuckGo()], markdown=True)

multimodal_Agent = initialize_agent()

# File uploader
video_file = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov'])
if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name
    st.video(video_path)

    # Buttons for different functionalities
    col1, col2 = st.columns(2)
    with col1:
        detect_objects = st.button("🔍 Detect Moving Objects & Accidents")
    with col2:
        analyze_video = st.button("🎥 Analyze Video with AI")

    if detect_objects:
        st.write("Processing video for object detection and accident analysis... ⏳")
        cap = cv2.VideoCapture(video_path)
        frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        output_path = os.path.join(tempfile.gettempdir(), "output_video.mp4")
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height), isColor=True)

        vehicle_positions = {}
        def calculate_velocity(prev_box, curr_box):
            if prev_box is None or curr_box is None:
                return 0
            prev_center = ((prev_box[0] + prev_box[2]) // 2, (prev_box[1] + prev_box[3]) // 2)
            curr_center = ((curr_box[0] + curr_box[2]) // 2, (curr_box[1] + curr_box[3]) // 2)
            return np.linalg.norm(np.array(curr_center) - np.array(prev_center))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            current_positions = {}
            accident_detected = False

            for r in results:
                for i, box in enumerate(r.boxes.xyxy):
                    x1, y1, x2, y2 = map(int, box)
                    class_id = int(r.boxes.cls[i])
                    label = CLASS_NAMES[class_id]
                    vehicle_id = f"{x1}_{y1}_{x2}_{y2}"
                    current_positions[vehicle_id] = (x1, y1, x2, y2)
                    prev_pos = vehicle_positions.get(vehicle_id)
                    velocity = calculate_velocity(prev_pos, (x1, y1, x2, y2))
                    color = (0, 255, 0)

                    if label in ["car", "truck", "bus", "motorbike", "auto", "rickshaw"] and velocity > 2:
                        accident_detected = True
                        color = (0, 0, 255)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            vehicle_positions = current_positions
            if accident_detected:
                cv2.putText(frame, "⚠️ Accident Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            out.write(frame)

        cap.release()
        out.release()
        st.success("Detection completed!")
        st.video(output_path)

    if analyze_video:
        user_query = st.text_area("What insights are you seeking from the video?", 
        placeholder="Ask anything about the video content.The AI agent will analyze and gather additional context if needed",
        help="Provide specific questions or insights you want from the video."
        )
        if user_query:
            try:
                with st.spinner("Processing video for AI analysis..."):
                    processed_video = upload_file(video_path)
                    while processed_video.state.name == "PROCESSING":
                        time.sleep(1)
                        processed_video = get_file(processed_video.name)

                    analysis_prompt = f"Analyze the uploaded video. Respond to: {user_query}" 
                    response = multimodal_Agent.run(analysis_prompt, videos=[processed_video])

                st.subheader("Analysis Result")
                st.markdown(response.content)

            except Exception as error:
                st.error(f"Error during analysis: {error}")
            finally:
                Path(video_path).unlink(missing_ok=True)
        else:
            st.warning("Please enter a query before analyzing.")
else:
    st.info("Upload a video file to begin analysis.")
