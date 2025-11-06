import cv2
import os
from pathlib import Path
from ultralytics import YOLO

VIDEO_PATH = "sample_basketball.mp4"
OUTPUT_VIDEO_PATH = "runs/track/basketball_tracking.mp4"
CLASSES_OF_INTEREST = [0, 32] # 0: person, 32: sports ball
CONFIDENCE_THRESHOLD = 0.5

def main():
    # Validate input video
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video not found: {VIDEO_PATH}")
        return

    # Output directory
    Path("runs/track").mkdir(parents=True, exist_ok=True)

    # Load model
    model = YOLO("yolov8x.pt")

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Failed to open video.")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    print(f"Starting tracking with ByteTrack...")

    frame_count = 0
    track_ids_seen = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Run YOLOv8 + ByteTrack
        results = model.track(
            source=frame,
            persist=True,
            classes=CLASSES_OF_INTEREST, # Only track person and ball
            conf=CONFIDENCE_THRESHOLD,
            verbose=False
        )
        # Get annotated frame
        annotated_frame = results[0].plot()
        if hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            track_ids_seen.update(track_ids)
        # Write frame to output video
        out.write(annotated_frame)
        frame_count += 1
        # Progress update
        if frame_count % 60 == 0:  # every 2-3 seconds (at 30 FPS)
            print(f"  Processed {frame_count}/{total_frames} frames...")
    # Cleanup
    cap.release()
    out.release()

    print(f"Tracking complete!")
    print(f"Output video saved to: {os.path.abspath(OUTPUT_VIDEO_PATH)}")

if __name__ == "__main__":

    main()
