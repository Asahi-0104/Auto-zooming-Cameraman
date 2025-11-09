from ultralytics import YOLO
import cv2
import numpy as np
import os

# ----------------------
# Configuration
# ----------------------
model_path = "runs/detect/train8/weights/best.pt"  # fine-tuned ball model
video_path = "uploads/tt3.mp4"                     # input video
output_dir = "output"                              # output folder
padding = 180                                      # box padding
radius = 250                                       # nearby player radius
key_frame_freq = 25                                # key frame step
conf_threshold = 0.25                              # confidence threshold for ball

os.makedirs(output_dir, exist_ok=True)

# ----------------------
# Load models
# ----------------------
ball_model = YOLO(model_path)      # fine-tuned ball model
player_model = YOLO("yolov8x.pt")  # default model for players

# ----------------------
# Load video
# ----------------------
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_ratio = w / (h + 1e-6)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# ----------------------
# Output
# ----------------------
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_path = os.path.join(output_dir, "focused_output.mp4")
out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

# ----------------------
# State
# ----------------------
prev_window = None
next_window_raw = None
delta_window = None
prev_ball = None
frame_idx = 0

print(f"Video loaded: {total_frames} frames, {fps:.1f} FPS")

# ----------------------
# 🧠 Helper: maintain aspect ratio
# ----------------------
def maintain_aspect(x_min, y_min, x_max, y_max, target_ratio, frame_w, frame_h):
    crop_w = x_max - x_min
    crop_h = y_max - y_min
    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2
    crop_ratio = crop_w / (crop_h + 1e-6)

    if abs(crop_ratio - target_ratio) > 1e-3:
        if crop_ratio > target_ratio:
            # too wide → extend height
            new_h = int(crop_w / target_ratio)
            new_w = crop_w
        else:
            # too tall → extend width
            new_w = int(crop_h * target_ratio)
            new_h = crop_h
        x_min = max(0, cx - new_w // 2)
        y_min = max(0, cy - new_h // 2)
        x_max = min(frame_w, x_min + new_w)
        y_max = min(frame_h, y_min + new_h)

    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

# ----------------------
# Process frames
# ----------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    balls, players = [], []

    # ----------------------
    # Key frame detection
    # ----------------------
    if frame_idx % key_frame_freq == 0:
        results_ball = ball_model.predict(frame, verbose=False, conf=conf_threshold)[0]
        results_player = player_model.predict(frame, verbose=False)[0]

        # select best ball box
        best_ball_box = None
        best_conf = 0.0
        for box in results_ball.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id == 0 and conf > best_conf:  # ball = 0
                best_conf = conf
                best_ball_box = box

        # record ball
        if best_ball_box is not None:
            x1, y1, x2, y2 = map(int, best_ball_box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            balls.append((cx, cy, x1, y1, x2, y2))
            prev_ball = (cx, cy, x1, y1, x2, y2)

        # record players
        for box in results_player.boxes:
            cls_id = int(box.cls[0])
            if cls_id == 0:  # person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                players.append((cx, cy, x1, y1, x2, y2))

        # ----------------------
        # compute region of interest
        # ----------------------
        if len(balls) > 0:
            bx, by, bx1, by1, bx2, by2 = balls[0]
            nearby_players = [p for p in players if np.hypot(p[0]-bx, p[1]-by)<radius]
            all_x1 = [bx1] + [p[2] for p in nearby_players]
            all_x2 = [bx2] + [p[4] for p in nearby_players]
            all_y1 = [by1] + [p[3] for p in nearby_players]
            all_y2 = [by2] + [p[5] for p in nearby_players]
        elif prev_ball is not None:
            bx, by = prev_ball[:2]
            nearby_players = [p for p in players if np.hypot(p[0]-bx, p[1]-by)<radius]
            all_x1 = [prev_ball[2]] + [p[2] for p in nearby_players]
            all_x2 = [prev_ball[4]] + [p[4] for p in nearby_players]
            all_y1 = [prev_ball[3]] + [p[3] for p in nearby_players]
            all_y2 = [prev_ball[5]] + [p[5] for p in nearby_players]
        else:
            frame_idx += 1
            continue

        # add padding
        x_min = max(0, int(min(all_x1)-padding))
        x_max = min(w, int(max(all_x2)+padding))
        y_min = max(0, int(min(all_y1)-padding))
        y_max = min(h, int(max(all_y2)+padding))

        # maintain aspect ratio
        next_window_raw = maintain_aspect(x_min, y_min, x_max, y_max, video_ratio, w, h)

        # compute interpolation delta
        if prev_window is not None:
            delta_window = (next_window_raw - prev_window) / key_frame_freq
        else:
            prev_window = next_window_raw.copy()
            delta_window = np.zeros_like(next_window_raw)

    # ----------------------
    # interpolate & crop
    # ----------------------
    if next_window_raw is not None and prev_window is not None:
        t = frame_idx % key_frame_freq
        smoothed_window = prev_window + delta_window * t
        x_min, y_min, x_max, y_max = maintain_aspect(*smoothed_window, video_ratio, w, h).astype(int)

        crop = frame[y_min:y_max, x_min:x_max]
        if crop.size != 0:
            focused_full = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
            out.write(focused_full)

        # ----------------------
        # visualization
        # ----------------------
        vis_frame = frame.copy()
        # green: focus window
        cv2.rectangle(vis_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        # red: ball box
        if prev_ball is not None:
            cv2.rectangle(vis_frame, (prev_ball[2], prev_ball[3]), (prev_ball[4], prev_ball[5]), (0, 0, 255), 2)
        combo = np.hstack((vis_frame, focused_full))
        cv2.imshow("Focus Box + Ball", combo)

        # update prev_window at key frame end
        if frame_idx % key_frame_freq == key_frame_freq - 1:
            prev_window = next_window_raw.copy()

    frame_idx += 1
    if cv2.waitKey(1) == 27:
        break

# ----------------------
# cleanup
# ----------------------
cap.release()
out.release()
cv2.destroyAllWindows()
print("Focused video saved to:", out_path)