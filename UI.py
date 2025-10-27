import streamlit as st
import tempfile
import os
import subprocess
import shutil

# Paths to your separate scripts (assume in same directory or adjust paths)
FOCUS_SCRIPT = "focus_algorithm_v4.py"  # Your main processing script
SAMPLE_INPUT_VIDEO = "tt.mp4"  # Assume you have a sample input video
SAMPLE_OUTPUT_VIDEO = "runs/after.mp4"  # Or a pre-processed sample output

# Function to run the focus algorithm script via subprocess
def run_focus_algorithm(input_path, output_dir="focus_output"):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "focused_output.mp4")
    
    # Temporarily modify the script to use the input_path (simple string replace for demo)
    # Note: For production, add argparse to focus_algorithm_v4.py instead
    with open(FOCUS_SCRIPT, 'r') as f:
        script_content = f.read()
    
    modified_content = script_content.replace('video_path = "tt.mp4"', f'video_path = "{input_path}"')
    modified_content = modified_content.replace('output_dir = "focus_crops2/focus_crops2"', f'output_dir = "{output_dir}"')
    modified_script_path = os.path.join(output_dir, "temp_focus.py")
    
    with open(modified_script_path, 'w') as f:
        f.write(modified_content)
    
    # Run the modified script
    result = subprocess.run(["python", modified_script_path], capture_output=True, text=True)
    
    if result.returncode != 0:
        raise Exception(f"Processing failed: {result.stderr}")
    
    return output_path

# Streamlit App
st.title("Auto-Zooming AI Cameraman for Basketball Games - Demo UI")

st.write("""
This app allows you to upload a basketball video, view the input, process it with the AI focus algorithm (run as a separate script), 
and view the output. For testing/demo without processing, use the 'Show Sample Demo' button.
""")

# Demo mode without processing
if st.button("Show Sample Demo (No Processing)"):
    if os.path.exists(SAMPLE_INPUT_VIDEO):
        st.subheader("Sample Input Video")
        st.video(SAMPLE_INPUT_VIDEO)
        
        if os.path.exists(SAMPLE_OUTPUT_VIDEO):
            st.subheader("Sample Processed Output (What the App Does)")
            st.video(SAMPLE_OUTPUT_VIDEO)
            st.write("This shows an example of auto-zooming: The output focuses on key actions like the ball and players, simulating camera movements.")
        else:
            st.warning("Sample output video not found. Run the focus algorithm on the sample input to generate it.")
    else:
        st.warning("Sample input video not found. Please add 'sample_basketball.mp4' to the directory.")

# Upload and process mode
uploaded_file = st.file_uploader("Upload a Video (MP4)", type=["mp4"])

if uploaded_file is not None:
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(uploaded_file.read())
        input_path = tfile.name

    st.subheader("Input Video")
    st.video(input_path)

    if st.button("Process Video"):
        with st.spinner("Processing... This may take a while depending on video length."):
            try:
                output_path = run_focus_algorithm(input_path)
                st.success("Processing complete!")

                st.subheader("Processed Output Video")
                st.video(output_path)

                # Download button
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="Download Processed Video",
                        data=f,
                        file_name="focused_output.mp4",
                        mime="video/mp4"
                    )
            except Exception as e:
                st.error(f"Error: {str(e)}")
            finally:
                # Cleanup
                if os.path.exists(input_path):
                    os.unlink(input_path)
                shutil.rmtree("focus_output", ignore_errors=True)