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

# Class names for YOLO (specific to COCO dataset)
CLASS_NAMES = model.names  # Mapping class IDs to names

# Streamlit UI
st.set_page_config(
    page_title="Accident Detection Video Analyzer",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Accident Detection & Object Tracking")
st.markdown("Upload a video to detect objects, track movement, and identify accidents.")

# Folder setup
project_folder = r"C:\AI Agents\AI_Agents\object_detection"
temp_folder = os.path.join(project_folder, "temp")
os.makedirs(temp_folder, exist_ok=True)

# File uploader
video_file = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov'])

if video_file:
    video_path = os.path.join(temp_folder, "input_video.mp4")
    with open(video_path, 'wb') as f:
        f.write(video_file.read())
    st.video(video_path)

    if st.button("🔍 Detect Objects & Accidents"):
        st.write("Processing video... This may take some time ⏳")

        cap = cv2.VideoCapture(video_path)
        frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        output_path = os.path.join(temp_folder, "output_video.mp4")
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height), isColor=True)

        vehicle_positions = {}

        def calculate_velocity(prev_box, curr_box):
            if prev_box is None or curr_box is None:
                return 0
            prev_x1, prev_y1, prev_x2, prev_y2 = prev_box
            curr_x1, curr_y1, curr_x2, curr_y2 = curr_box
            prev_center = ((prev_x1 + prev_x2) // 2, (prev_y1 + prev_y2) // 2)
            curr_center = ((curr_x1 + curr_x2) // 2, (curr_y1 + curr_y2) // 2)
            return np.linalg.norm(np.array(curr_center) - np.array(prev_center))

        def iou(box1, box2):
            x1, y1, x2, y2 = box1
            x1_p, y1_p, x2_p, y2_p = box2
            xi1, yi1 = max(x1, x1_p), max(y1, y1_p)
            xi2, yi2 = min(x2, x2_p), min(y2, y2_p)
            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            union_area = (x2 - x1) * (y2 - y1) + (x2_p - x1_p) * (y2_p - y1_p) - inter_area
            return inter_area / union_area if union_area > 0 else 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            accident_detected = False
            collisions = []
            current_positions = {}

            for r in results:
                for i, box in enumerate(r.boxes.xyxy):
                    x1, y1, x2, y2 = map(int, box)
                    class_id = int(r.boxes.cls[i])
                    label = CLASS_NAMES[class_id]

                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    vehicle_id = f"{center_x}_{center_y}"
                    current_positions[vehicle_id] = (x1, y1, x2, y2)
                    prev_pos = vehicle_positions.get(vehicle_id)
                    velocity = calculate_velocity(prev_pos, (x1, y1, x2, y2))
                    color = (0, 255, 0)

                    if label in ["car", "truck", "bus", "motorbike", "auto", "rickshaw"]:
                        for other_id, other_box in current_positions.items():
                            if other_id == vehicle_id:
                                continue
                            if iou((x1, y1, x2, y2), other_box) > 0.5:
                                prev_other_pos = vehicle_positions.get(other_id)
                                other_velocity = calculate_velocity(prev_other_pos, other_box)
                                if velocity > 2 and other_velocity > 2:
                                    accident_detected = True
                                    collisions.append((label, CLASS_NAMES[int(r.boxes.cls[i])]))
                                    color = (0, 0, 255)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            vehicle_positions = current_positions

            if accident_detected:
                cv2.putText(frame, "⚠️ Accident Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                for v1, v2 in collisions:
                    cv2.putText(frame, f"Collision: {v1} & {v2}", (50, 100 + 30 * collisions.index((v1, v2))), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            out.write(frame)
            cv2.waitKey(1)

        cap.release()
        out.release()
        time.sleep(2)
        final_output_path = os.path.join(temp_folder, "final_output_video.mp4")
        shutil.copy(output_path, final_output_path)

        if os.path.exists(final_output_path) and os.path.getsize(final_output_path) > 0:
            st.success("Object detection & accident detection completed! 🎯")
            st.video(final_output_path)
        else:
            st.error("Error: Output video file is empty or not found!")

        os.remove(video_path)




