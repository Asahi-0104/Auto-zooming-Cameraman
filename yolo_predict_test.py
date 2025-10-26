import cv2
import os
import numpy as np
from pathlib import Path
from ultralytics import YOLO

VIDEO_PATH = "sample_basketball.mp4"
OUTPUT_DIR = "runs/cropped_objects"
KEYFRAME_INTERVAL = 25     # Analyze 1 frame every N frames
MOTION_THRESHOLD = 8000    # Min pixel change to consider motion
CONFIDENCE_THRESHOLD = 0.5 # Min confidence for detections

def setup_output_dir():
    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    # Clear previous crops
    for f in Path(OUTPUT_DIR).glob("*.jpg"):
        f.unlink()

def is_motion_detected(frame1, frame2, threshold=MOTION_THRESHOLD):
    # Detect motion between two frames using absolute difference
    if frame1 is None or frame2 is None:
        return True
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    return np.sum(thresh) > threshold

def crop_and_save_objects(frame, results, frame_idx):
    # Save cropped images
    frame_crops = 0
    for i, box in enumerate(results.boxes):
        cls = int(box.cls.item())
        conf = float(box.conf.item())
        if conf < CONFIDENCE_THRESHOLD:
            continue
        # Save person (class 0) and sports ball (class 32)
        if cls not in [0, 32]:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        # Name
        crop_name = f"frame_{frame_idx:05d}_class_{cls}_conf_{conf:.2f}.jpg"
        crop_path = os.path.join(OUTPUT_DIR, crop_name)
        cv2.imwrite(crop_path, crop)
        frame_crops += 1
    return frame_crops

def main():
    # Check video
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video not found: {VIDEO_PATH}")
        print("Please place 'sample_basketball.mp4' in this directory.")
        return

    # Setup
    setup_output_dir()
    model = YOLO("yolov8x.pt")

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Failed to open video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total_frames} frames at {fps:.1f} FPS")
    print(f"Keyframe interval: {KEYFRAME_INTERVAL}")
    print(f"Motion threshold: {MOTION_THRESHOLD}\n")
    frame_idx = 0
    prev_frame = None
    processed_frames = 0
    total_crops = 0

    # Store YOLO results per frame index
    frame_results = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Consider key frames
        if frame_idx % KEYFRAME_INTERVAL == 0:
            if is_motion_detected(prev_frame, frame):
                results = model(frame, verbose=False)
                # store the Results object
                frame_results[frame_idx] = results[0]
                # Save crops
                crops_saved = crop_and_save_objects(frame, results[0], frame_idx)
                total_crops += crops_saved
                processed_frames += 1
                print(f"Frame {frame_idx}: {crops_saved} objects cropped")
            else:
                print(f"Frame {frame_idx}: Skipped (no motion)")
        prev_frame = frame.copy()
        frame_idx += 1
    cap.release()

    print("Complete")
    print(f"Total frames processed: {processed_frames} / {total_frames}")
    print(f"Total cropped images saved: {total_crops}")

    # Output video
    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    output_video_path = "runs/yolo_sparse_output.mp4"
    Path("runs").mkdir(exist_ok=True)
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_idx = 0

    print("Rendering output video...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in frame_results:
            annotated = frame_results[frame_idx].plot()
            out.write(annotated)
        else:
            # Raw frame
            out.write(frame)
        frame_idx += 1
    cap.release()
    out.release()
    print("Complete")
    print(f"Output video saved to: {output_video_path}")
    
if __name__ == "__main__":
    main()
