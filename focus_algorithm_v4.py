from ultralytics import YOLO
import cv2
import numpy as np
import os

# setup
model_path = "focus_crops2/yolov8x.pt"
video_path = "tt.mp4"
output_dir = "focus_crops2/focus_crops2"
padding = 100
radius = 200
key_frame_freq = 20
smooth_alpha = 0.3

os.makedirs(output_dir, exist_ok=True)

# load model
model = YOLO(model_path)
cap = cv2.VideoCapture(video_path)

# get video info
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_ratio = w / (h + 0.00001)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# prepare output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(os.path.join(output_dir, "focused_output.mp4"), fourcc, fps, (w, h))

# state
prev_window = None
next_window = None
frame_idx = 0
prev_ball = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # check every frame
    if frame_idx % key_frame_freq == 0:
        results = model.predict(frame, verbose=False)[0]
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            frame_idx += 1
            continue

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
            bx, by, bx1, by1, bx2, by2 = balls[0]
            nearby_players = [p for p in players if np.hypot(p[0] - bx, p[1] - by) < radius]
            all_x1 = [bx1] + [p[2] for p in nearby_players]
            all_x2 = [bx2] + [p[4] for p in nearby_players]
            all_y1 = [by1] + [p[3] for p in nearby_players]
            all_y2 = [by2] + [p[5] for p in nearby_players]
            prev_ball = (bx, by, bx1, by1, bx2, by2)
        elif len(players) > 0:
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

        else:
            frame_idx += 1
            continue

        x_min, x_max = int(min(all_x1) - padding), int(max(all_x2) + padding)
        y_min, y_max = int(min(all_y1) - padding), int(max(all_y2) + padding)
        x_min, x_max = max(0, x_min), min(w, x_max)
        y_min, y_max = max(0, y_min), min(h, y_max)

        next_window = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)


        if prev_window is None:
            prev_window = next_window.copy()


        delta_window = (next_window - prev_window) / key_frame_freq

    # ---------------------------
    # interpolate between windows
    # ---------------------------
    if next_window is not None and prev_window is not None:
        t = frame_idx % key_frame_freq
        smoothed_window = prev_window + delta_window * t
    else:
        frame_idx += 1
        continue

    # maintain aspect ratio
    x_min, y_min, x_max, y_max = map(int, smoothed_window)
    crop_w = x_max - x_min
    crop_h = y_max - y_min
    crop_ratio = crop_w / crop_h

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

    # crop & resize
    crop = frame[y_min:y_max, x_min:x_max]
    if crop.size != 0:
        focused_full = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
        out.write(focused_full)
        cv2.imshow("Focused", focused_full)

    # visualize focus box
    vis_frame = frame.copy()
    cv2.rectangle(vis_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.imshow("Focus Box", vis_frame)

    if frame_idx % key_frame_freq == key_frame_freq - 1:
        prev_window = next_window.copy()

    if cv2.waitKey(1) == 27:
        break

    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ Focused video saved to:", os.path.join(output_dir, "focused_output.mp4"))