import streamlit as st
import sys
import os
import subprocess

# Paths
FOCUS_SCRIPT = "focus_algorithm_v4.py"  # Your argparse-enabled script
SAMPLE_INPUT_VIDEO = "demo_run/tt.mp4"
SAMPLE_OUTPUT_STEM = os.path.splitext(os.path.basename(SAMPLE_INPUT_VIDEO))[0]
SAMPLE_OUTPUT_VIDEO = os.path.join("focus_output", f"{SAMPLE_OUTPUT_STEM}_focused.mp4")
UPLOADS_DIR = "uploads"
FOCUS_OUTPUT_DIR = "focus_output"

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(FOCUS_OUTPUT_DIR, exist_ok=True)

# Function to run the focus algorithm using venv's Python and argparse
def run_focus_algorithm(input_path, original_name):
    output_filename = f"{os.path.splitext(original_name)[0]}_focused.mp4"
    output_path = os.path.join(FOCUS_OUTPUT_DIR, output_filename)
    
    # Use sys.executable for venv Python; pass args directly
    result = subprocess.run([
        sys.executable, FOCUS_SCRIPT,
        "--video_path", input_path,
        "--output_dir", FOCUS_OUTPUT_DIR,
        "--output_filename", output_filename
    ], capture_output=True)
    
    # Decode output with error handling to avoid UnicodeDecodeError
    stdout = result.stdout.decode('utf-8', errors='replace')
    stderr = result.stderr.decode('utf-8', errors='replace')
    
    if result.returncode != 0:
        raise Exception(f"Processing failed: {stderr}\n{stdout}")
    
    return output_path, output_filename  # Return both

# Streamlit App
st.title("Auto-Zooming AI Cameraman for Basketball Games - Demo UI")

st.write("""
This app allows you to upload a basketball video, process it with the AI focus algorithm, 
and view/download the output. You can also select from previously uploaded videos to view input and processed output.
""")

# Sample demo
if st.button("Show Sample Demo (No Processing)"):
    if os.path.exists(SAMPLE_INPUT_VIDEO):
        st.subheader("Sample Input Video")
        with open(SAMPLE_INPUT_VIDEO, "rb") as f:
            st.video(f.read(), format="video/mp4")
        
        if not os.path.exists(SAMPLE_OUTPUT_VIDEO):
            st.info("Sample output not found. Generating now using the focus algorithm...")
            with st.spinner("Generating sample output... This may take a while."):
                try:
                    output_path, output_filename = run_focus_algorithm(SAMPLE_INPUT_VIDEO, os.path.basename(SAMPLE_INPUT_VIDEO))
                except Exception as e:
                    st.error(f"Failed to generate sample output: {str(e)}")
        
        if os.path.exists(SAMPLE_OUTPUT_VIDEO):
            st.subheader("Sample Processed Output")
            with open(SAMPLE_OUTPUT_VIDEO, "rb") as f:
                video_bytes = f.read()
                st.video(video_bytes, format="video/mp4")
    else:
        st.warning("Sample input not found.")

# Initialize session state for selected video
if 'selected_video' not in st.session_state:
    st.session_state.selected_video = None

# Upload and process
uploaded_file = st.file_uploader("Upload a Video (MP4)", type=["mp4"])

if uploaded_file is not None:
    original_name = uploaded_file.name
    input_path = os.path.join(UPLOADS_DIR, original_name)
    
    # Save to uploads (persist)
    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())

    if st.button("Process Video"):
        with st.spinner("Processing... This may take a while depending on video length."):
            try:
                output_path, output_filename = run_focus_algorithm(input_path, original_name)
                st.success("Processing complete!")
                # Set the selected video to this one to trigger side-by-side display
                st.session_state.selected_video = original_name
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Always show the dropdown for selecting uploaded videos
st.subheader("Select Uploaded Video to View")
uploaded_videos = [f for f in os.listdir(UPLOADS_DIR) if f.endswith(".mp4")]
if uploaded_videos:
    st.session_state.selected_video = st.selectbox("Choose a video:", uploaded_videos, index=uploaded_videos.index(st.session_state.selected_video) if st.session_state.selected_video in uploaded_videos else 0)
    selected_video = st.session_state.selected_video
    if selected_video:
        input_path = os.path.join(UPLOADS_DIR, selected_video)
        output_filename = f"{os.path.splitext(selected_video)[0]}_focused.mp4"
        output_path = os.path.join(FOCUS_OUTPUT_DIR, output_filename)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Input Video")
            with open(input_path, "rb") as f:
                st.video(f.read(), format="video/mp4")
        with col2:
            st.subheader("Processed Output")
            if os.path.exists(output_path):
                with open(output_path, "rb") as f:
                    video_bytes = f.read()
                    st.video(video_bytes, format="video/mp4")
                # Download button for processed video
                st.download_button(
                    label="Download Processed Video",
                    data=video_bytes,
                    file_name=output_filename,
                    mime="video/mp4"
                )
            else:
                st.warning("No processed output yet. Click 'Process Video' above after uploading.")
else:
    st.info("No uploaded videos yet. Upload one to get started.")