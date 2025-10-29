import argparse
from ultralytics import YOLO
import cv2
import numpy as np
import os

# Argument parser
parser = argparse.ArgumentParser(description="Auto-zooming focus algorithm for basketball videos")
parser.add_argument("--video_path", required=True, help="Path to the input video file")
parser.add_argument("--output_dir", default="focus_output", help="Directory to save the output video")
parser.add_argument("--output_filename", default="focused_output.mp4", help="Name of the output video file")
parser.add_argument("--model_path", default="focus_crops2/yolov8x.pt", help="Path to the YOLO model")
parser.add_argument("--padding", type=int, default=100, help="Padding around the focus area")
parser.add_argument("--radius", type=int, default=200, help="Radius for nearby players")
parser.add_argument("--key_frame_freq", type=int, default=20, help="Frequency of key frames for detection")
parser.add_argument("--smooth_alpha", type=float, default=0.3, help="Smoothing alpha (not used in current code)")
args = parser.parse_args()

# Setup with args
model_path = args.model_path
video_path = args.video_path
output_dir = args.output_dir
output_filename = args.output_filename
padding = args.padding
radius = args.radius
key_frame_freq = args.key_frame_freq
smooth_alpha = args.smooth_alpha  # Note: This is defined but not used in the code—consider implementing or removing

os.makedirs(output_dir, exist_ok=True)

# Load model
model = YOLO(model_path)
cap = cv2.VideoCapture(video_path)

# Get video info
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_ratio = w / (h + 0.00001)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Prepare output
fourcc = cv2.VideoWriter_fourcc(*'H264')  # Changed to H264 for better browser compatibility
output_path = os.path.join(output_dir, output_filename)
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

# State
prev_window = None
next_window = None
frame_idx = 0
prev_ball = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Check every key frame
    if frame_idx % key_frame_freq == 0:
        results = model.predict(frame, verbose=False)[0]
        boxes = results.boxes
        has_detections = False
        if boxes is not None and len(boxes) > 0:
            players, balls = [], []
            for box in boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if cls_id == 0:
                    players.append((cx, cy, x1, y1, x2, y2))
                elif cls_id == 32:
                    balls.append((cx, cy, x1, y1, x2, y2))

            if len(balls) > 0:
                has_detections = True
                bx, by, bx1, by1, bx2, by2 = balls[0]
                nearby_players = [p for p in players if np.hypot(p[0] - bx, p[1] - by) < radius]
                all_x1 = [bx1] + [p[2] for p in nearby_players]
                all_x2 = [bx2] + [p[4] for p in nearby_players]
                all_y1 = [by1] + [p[3] for p in nearby_players]
                all_y2 = [by2] + [p[5] for p in nearby_players]
                prev_ball = (bx, by, bx1, by1, bx2, by2)
            elif len(players) > 0:
                has_detections = True
                if prev_ball is None:
                    all_x1 = [p[2] for p in players]
                    all_x2 = [p[4] for p in players]
                    all_y1 = [p[3] for p in players]
                    all_y2 = [p[5] for p in players]
                else:
                    nearby_players = [p for p in players if np.hypot(p[0] - prev_ball[0], p[1] - prev_ball[1]) < radius]
                    all_x1 = [prev_ball[2]] + [p[2] for p in nearby_players]
                    all_x2 = [prev_ball[4]] + [p[4] for p in nearby_players]
                    all_y1 = [prev_ball[3]] + [p[3] for p in nearby_players]
                    all_y2 = [prev_ball[5]] + [p[5] for p in nearby_players]

        # Compute window or fallback to full frame
        if has_detections:
            x_min = int(min(all_x1) - padding)
            x_max = int(max(all_x2) + padding)
            y_min = int(min(all_y1) - padding)
            y_max = int(max(all_y2) + padding)
        else:
            x_min = 0
            x_max = w
            y_min = 0
            y_max = h

        x_min, x_max = max(0, x_min), min(w, x_max)
        y_min, y_max = max(0, y_min), min(h, y_max)

        next_window = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

        if prev_window is None:
            prev_window = next_window.copy()

        delta_window = (next_window - prev_window) / key_frame_freq

    # Interpolate between windows
    if next_window is not None and prev_window is not None:
        t = frame_idx % key_frame_freq
        smoothed_window = prev_window + delta_window * t
    else:
        # Fallback if no window yet (though initialization should prevent this)
        smoothed_window = np.array([0, 0, w, h], dtype=np.float32)

    # Maintain aspect ratio
    x_min, y_min, x_max, y_max = map(int, smoothed_window)
    crop_w = x_max - x_min
    crop_h = y_max - y_min
    crop_ratio = crop_w / crop_h if crop_h != 0 else video_ratio

    if abs(crop_ratio - video_ratio) > 1e-3:
        if crop_ratio > video_ratio:
            new_h = int(crop_w / video_ratio)
            diff = new_h - crop_h
            y_min = max(0, y_min - diff // 2)
            y_max = min(h, y_min + new_h)
        else:
            new_w = int(crop_h * video_ratio)
            diff = new_w - crop_w
            x_min = max(0, x_min - diff // 2)
            x_max = min(w, x_min + new_w)

    # Crop & resize
    crop = frame[y_min:y_max, x_min:x_max]
    if crop.size != 0:
        focused_full = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        focused_full = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)  # Fallback to full frame

    out.write(focused_full)

    # Comment out visualizations for non-interactive runs
    # cv2.imshow("Focused", focused_full)
    # vis_frame = frame.copy()
    # cv2.rectangle(vis_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    # cv2.imshow("Focus Box", vis_frame)
    # if cv2.waitKey(1) == 27:
    #     break

    if frame_idx % key_frame_freq == key_frame_freq - 1:
        prev_window = next_window.copy()

    frame_idx += 1

cap.release()
out.release()
# cv2.destroyAllWindows()  # Comment out if no windows were opened
print("✅ Focused video saved to:", output_path)