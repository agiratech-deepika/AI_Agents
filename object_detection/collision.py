# import streamlit as st
# import cv2
# import os
# import time
# import shutil
# import torch
# from ultralytics import YOLO
# import numpy as np

# torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

# # Load YOLOv8 model
# model = YOLO("yolov8n.pt")  # Use "yolov8s.pt" for better accuracy

# # Streamlit UI
# st.set_page_config(
#     page_title="Accident Detection Video Analyzer",
#     page_icon="🚗",
#     layout="wide"
# )

# st.title("🚗 Accident Detection & Object Tracking")
# st.markdown("Upload a video to detect objects and track accidents.")

# # Folder where the video and output will be saved
# project_folder = r"C:\AI Agents\AI_Agents\object_detection"
# temp_folder = os.path.join(project_folder, "temp")

# # Ensure that the temp folder exists
# if not os.path.exists(temp_folder):
#     os.makedirs(temp_folder)

# # File Uploader
# video_file = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov'])

# if video_file:
#     # Save the uploaded video
#     video_path = os.path.join(temp_folder, "input_video.mp4")
#     with open(video_path, 'wb') as f:
#         f.write(video_file.read())

#     st.video(video_path)  # Show uploaded video

#     if st.button("🔍 Detect Objects & Accidents"):
#         st.write("Processing video... This may take some time ⏳")

#         # OpenCV Video Processing
#         cap = cv2.VideoCapture(video_path)
#         frame_width = int(cap.get(3))
#         frame_height = int(cap.get(4))
#         fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Use 'avc1' for H.264 encoding

#         # Output video path
#         output_path = os.path.join(temp_folder, "output_video.mp4")
#         out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height), isColor=True)

#         # Dictionary to store previous frame vehicle positions
#         vehicle_positions = {}

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # Perform object detection
#             results = model(frame)

#             accident_detected = False  # Flag to indicate if an accident has occurred
#             collisions = []  # Store which vehicles collided

#             # Store vehicle positions for collision detection
#             current_positions = {}

#             for r in results:
#                 for box in r.boxes.xyxy:
#                     x1, y1, x2, y2 = map(int, box)  # Bounding box coordinates
#                     center_x = (x1 + x2) // 2
#                     center_y = (y1 + y2) // 2

#                     # Assign an ID based on center position
#                     vehicle_id = f"{center_x}_{center_y}"
#                     current_positions[vehicle_id] = (x1, y1, x2, y2)

#                     # Check for collisions (if previous frame had the same vehicle in a different position)
#                     if vehicle_id in vehicle_positions:
#                         prev_x1, prev_y1, prev_x2, prev_y2 = vehicle_positions[vehicle_id]

#                         # Simple collision detection: Check if bounding boxes overlap significantly
#                         if abs(prev_x1 - x1) < 30 and abs(prev_y1 - y1) < 30:
#                             accident_detected = True
#                             collisions.append(vehicle_id)

#                     # Choose bounding box color: Green if normal, Red if accident detected
#                     color = (0, 255, 0)  # Default Green
#                     if vehicle_id in collisions:
#                         color = (0, 0, 255)  # Red for accident

#                     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

#             # Update previous frame vehicle positions
#             vehicle_positions = current_positions

#             # Display accident alert
#             if accident_detected:
#                 cv2.putText(frame, "⚠️ Accident Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
#                 for v_id in collisions:
#                     cv2.putText(frame, f"Collision: {v_id}", (50, 100 + 30 * collisions.index(v_id)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#             out.write(frame)
#             cv2.waitKey(1)

#         cap.release()
#         out.release()

#         # Ensure video file is written before displaying
#         time.sleep(2)  

#         # Copy the output video to a new file before displaying
#         final_output_path = os.path.join(temp_folder, "final_output_video.mp4")
#         shutil.copy(output_path, final_output_path)

#         # Verify file existence and size
#         if os.path.exists(final_output_path) and os.path.getsize(final_output_path) > 0:
#             st.success("Object detection & accident detection completed! 🎯")
#             st.video(final_output_path)  # Display processed video
#         else:
#             st.error("Error: Output video file is empty or not found!")

#         # Cleanup
#         os.remove(video_path)


# import streamlit as st
# import cv2
# import os
# import time
# import shutil
# import torch
# from ultralytics import YOLO
# import numpy as np

# torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

# # Load YOLOv8 model
# model = YOLO("yolov8n.pt")  # Use "yolov8s.pt" for better accuracy

# # Streamlit UI
# st.set_page_config(
#     page_title="Accident Detection Video Analyzer",
#     page_icon="🚗",
#     layout="wide"
# )

# st.title("🚗 Accident Detection & Object Tracking")
# st.markdown("Upload a video to detect objects and track accidents.")

# # Folder where the video and output will be saved
# project_folder = r"C:\AI Agents\AI_Agents\object_detection"
# temp_folder = os.path.join(project_folder, "temp")

# # Ensure that the temp folder exists
# if not os.path.exists(temp_folder):
#     os.makedirs(temp_folder)

# # File Uploader
# video_file = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov'])

# if video_file:
#     # Save the uploaded video
#     video_path = os.path.join(temp_folder, "input_video.mp4")
#     with open(video_path, 'wb') as f:
#         f.write(video_file.read())

#     st.video(video_path)  # Show uploaded video

#     if st.button("🔍 Detect Objects & Accidents"):
#         st.write("Processing video... This may take some time ⏳")

#         # OpenCV Video Processing
#         cap = cv2.VideoCapture(video_path)
#         frame_width = int(cap.get(3))
#         frame_height = int(cap.get(4))
#         fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Use 'avc1' for H.264 encoding

#         # Output video path
#         output_path = os.path.join(temp_folder, "output_video.mp4")
#         out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height), isColor=True)

#         # Dictionary to store vehicle positions across frames
#         vehicle_positions = {}


#        def calculate_velocity(prev_box, curr_box):
#             """Calculate velocity based on bounding box center point movement."""
#             if prev_box is None or curr_box is None:
#                 return 0  # If no previous position, assume stationary

#             # Extract box coordinates
#             prev_x1, prev_y1, prev_x2, prev_y2 = prev_box
#             curr_x1, curr_y1, curr_x2, curr_y2 = curr_box

#             # Calculate center points of bounding boxes
#             prev_center_x = (prev_x1 + prev_x2) // 2
#             prev_center_y = (prev_y1 + prev_y2) // 2
#             curr_center_x = (curr_x1 + curr_x2) // 2
#             curr_center_y = (curr_y1 + curr_y2) // 2

#             # Compute velocity based on Euclidean distance
#             velocity = ((curr_center_x - prev_center_x) ** 2 + (curr_center_y - prev_center_y) ** 2) ** 0.5
#             return velocity



#         def iou(box1, box2):
#             """Calculate Intersection over Union (IoU) between two bounding boxes."""
#             x1, y1, x2, y2 = box1
#             x1_p, y1_p, x2_p, y2_p = box2

#             # Calculate intersection
#             xi1 = max(x1, x1_p)
#             yi1 = max(y1, y1_p)
#             xi2 = min(x2, x2_p)
#             yi2 = min(y2, y2_p)
#             inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

#             # Calculate union
#             box1_area = (x2 - x1) * (y2 - y1)
#             box2_area = (x2_p - x1_p) * (y2_p - y1_p)
#             union_area = box1_area + box2_area - inter_area

#             return inter_area / union_area if union_area > 0 else 0

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # Perform object detection
#             results = model(frame)

#             accident_detected = False  # Flag to indicate if an accident has occurred
#             collisions = []  # Store which vehicles collided
#             current_positions = {}

#             for r in results:
#                 for box in r.boxes.xyxy:
#                     x1, y1, x2, y2 = map(int, box)  # Bounding box coordinates
#                     center_x = (x1 + x2) // 2
#                     center_y = (y1 + y2) // 2

#                     # Assign an ID based on center position
#                     vehicle_id = f"{center_x}_{center_y}"
#                     current_positions[vehicle_id] = (x1, y1, x2, y2)

#                     # Check if vehicle was in previous frame
#                     prev_pos = vehicle_positions.get(vehicle_id)
#                     velocity = calculate_velocity(prev_pos, (center_x, center_y))

#                     # Default bounding box color (green)
#                     color = (0, 255, 0)

#                     # Check for real collision
#                     for other_id, other_box in current_positions.items():
#                         if other_id == vehicle_id:
#                             continue  # Skip self-comparison

#                         if iou((x1, y1, x2, y2), other_box) > 0.5:  # IoU threshold for collision
#                             prev_other_pos = vehicle_positions.get(other_id)
#                             other_velocity = calculate_velocity(prev_other_pos, (other_box[0], other_box[1]))

#                             # Ensure both vehicles are moving before collision
#                             if velocity > 2 and other_velocity > 2:  # Velocity threshold
#                                 accident_detected = True
#                                 collisions.append((vehicle_id, other_id))
#                                 color = (0, 0, 255)  # Mark accident in red

#                     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

#             # Update vehicle positions
#             vehicle_positions = current_positions

#             # Display accident alert
#             if accident_detected:
#                 cv2.putText(frame, "⚠️ Accident Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
#                 for v1, v2 in collisions:
#                     cv2.putText(frame, f"Collision: {v1} & {v2}", (50, 100 + 30 * collisions.index((v1, v2))), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#             out.write(frame)
#             cv2.waitKey(1)

#         cap.release()
#         out.release()

#         # Ensure video file is written before displaying
#         time.sleep(2)

#         # Copy the output video to a new file before displaying
#         final_output_path = os.path.join(temp_folder, "final_output_video.mp4")
#         shutil.copy(output_path, final_output_path)

#         # Verify file existence and size
#         if os.path.exists(final_output_path) and os.path.getsize(final_output_path) > 0:
#             st.success("Object detection & accident detection completed! 🎯")
#             st.video(final_output_path)  # Display processed video
#         else:
#             st.error("Error: Output video file is empty or not found!")

#         # Cleanup
#         os.remove(video_path)


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
    video_path = os.path.join(temp_folder, "input_video.mp4")
    with open(video_path, 'wb') as f:
        f.write(video_file.read())

    st.video(video_path)  # Show uploaded video

    if st.button("🔍 Detect Objects & Accidents"):
        st.write("Processing video... This may take some time ⏳")

        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'avc1')

        output_path = os.path.join(temp_folder, "output_video.mp4")
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height), isColor=True)

        vehicle_positions = {}

        def calculate_velocity(prev_box, curr_box):
            if prev_box is None or curr_box is None:
                return 0
            prev_x1, prev_y1, prev_x2, prev_y2 = prev_box
            curr_x1, curr_y1, curr_x2, curr_y2 = curr_box
            prev_center_x = (prev_x1 + prev_x2) // 2
            prev_center_y = (prev_y1 + prev_y2) // 2
            curr_center_x = (curr_x1 + curr_x2) // 2
            curr_center_y = (curr_y1 + curr_y2) // 2
            return ((curr_center_x - prev_center_x) ** 2 + (curr_center_y - prev_center_y) ** 2) ** 0.5

        def iou(box1, box2):
            x1, y1, x2, y2 = box1
            x1_p, y1_p, x2_p, y2_p = box2
            xi1 = max(x1, x1_p)
            yi1 = max(y1, y1_p)
            xi2 = min(x2, x2_p)
            yi2 = min(y2, y2_p)
            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            box1_area = (x2 - x1) * (y2 - y1)
            box2_area = (x2_p - x1_p) * (y2_p - y1_p)
            union_area = box1_area + box2_area - inter_area
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
                for box in r.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    vehicle_id = f"{x1}_{y1}_{x2}_{y2}"
                    current_positions[vehicle_id] = (x1, y1, x2, y2)

                    prev_pos = vehicle_positions.get(vehicle_id)
                    velocity = calculate_velocity(prev_pos, (x1, y1, x2, y2))
                    color = (0, 255, 0)

                    for other_id, other_box in current_positions.items():
                        if other_id == vehicle_id:
                            continue
                        if iou((x1, y1, x2, y2), other_box) > 0.5:
                            prev_other_pos = vehicle_positions.get(other_id)
                            other_velocity = calculate_velocity(prev_other_pos, other_box)
                            if velocity > 2 and other_velocity > 2:
                                accident_detected = True
                                collisions.append((vehicle_id, other_id))
                                color = (0, 0, 255)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

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