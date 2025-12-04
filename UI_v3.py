import streamlit as st
import tempfile
import os
import cv2
from focus_algorithm_v16 import process_video
import numpy as np

st.set_page_config(layout="wide")
st.title("Auto-Zooming AI Cameraman for Basketball Games")

# ----------------------
# 1️⃣ Upload video
# ----------------------
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Save uploaded file to temp
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_input.write(uploaded_file.read())
    temp_input.close()
    video_path = temp_input.name

    st.subheader("Uploaded Video Preview")
    st.video(video_path)
    st.success("✅ Uploaded video preview")

    # ----------------------
    # 2️⃣ Realtime Visualization & Progress
    # ----------------------
    st.subheader("Realtime Visualization (Show every 5 frames for visualization purpose only)")
    debug_placeholder = st.empty()

    st.subheader("Progress")
    progress_bar = st.progress(0)
    fps_text = st.empty()

    def debug_callback(frame):
        """Display frame in Streamlit"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        debug_placeholder.image(frame_rgb, use_container_width=True)

    def progress_callback(frame_idx, total_frames, fps):
        progress = min(frame_idx / total_frames, 1.0)
        progress_bar.progress(progress)
        fps_text.text(f"Processing frame {frame_idx}/{total_frames} — FPS: {fps:.2f}")

    # ----------------------
    # 3️⃣ Process video
    # ----------------------
    st.write("Processing video... this may take a while ⏳")

    output_dir = "streamlit_out"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "result.mp4")
    output_path = os.path.abspath(output_path)


    # Call your algorithm
    process_video(
        ball_model_path="models/best.pt",
        player_model_path="models/yolov8s.pt",
        video_path=video_path,
        output_path=output_path,
        visualization=True,
        debug_callback=debug_callback,
        progress_callback=progress_callback
    )

    # ----------------------
    # 4️⃣ Show Final Exported Video
    # ----------------------
    st.subheader("Processed Video Output")
    # st.video(output_path, format="video/mp4", start_time=0)
    st.success(f"Finished! Video saved at: \n`{output_path}`")
    st.write("You can open this file in your local player.")