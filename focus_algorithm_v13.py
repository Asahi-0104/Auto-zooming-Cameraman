from torch.distributed.tensor.debug import visualize_sharding
from ultralytics import YOLO
import cv2
import numpy as np
import os
import time

# ----------------------
# Configuration
# ----------------------
model_path = "models/best.pt"       # YOLOv8x 默认模型
player_model_path = "models/yolov8s.pt"
video_path = "uploads/tt12.mp4"  # 输入视频
output_dir = "output"           # 输出文件夹
padding = 200                   # 裁切扩展范围
radius = 250                   # 球周围检测球员半径
key_frame_freq = 30             # 关键帧间隔
conf_threshold = 0.25           # YOLO置信度阈值
lost_ball_max = 15          # 球丢失超过多少帧回退全屏
visualization = True


os.makedirs(output_dir, exist_ok=True)

# ----------------------
# Load YOLO model
# ----------------------
ball_model = YOLO(model_path)
player_model = YOLO("yolov8n.pt")
ball_model.to("mps")
player_model.to("mps")
print(ball_model.device)

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
# Output video writer
# ----------------------
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_path = os.path.join(output_dir, "focused_output.mp4")
out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

# ----------------------
# Kalman Filter initialization
# ----------------------
kf = cv2.KalmanFilter(4, 2)
kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0]], np.float32)
kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                [0, 1, 0, 1],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]], np.float32)
kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

# ----------------------
# Helper functions
# ----------------------
def maintain_aspect(x_min, y_min, x_max, y_max, target_ratio, frame_w, frame_h):
    crop_w = x_max - x_min
    crop_h = y_max - y_min
    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2
    crop_ratio = crop_w / (crop_h + 1e-6)

    if abs(crop_ratio - target_ratio) > 1e-3:
        if crop_ratio > target_ratio:
            new_h = int(crop_w / target_ratio)
            new_w = crop_w
        else:
            new_w = int(crop_h * target_ratio)
            new_h = crop_h
        x_min = max(0, cx - new_w // 2)
        y_min = max(0, cy - new_h // 2)
        x_max = min(frame_w, x_min + new_w)
        y_max = min(frame_h, y_min + new_h)
    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

# ----------------------
# State
# ----------------------
prev_window = None
next_window_raw = None
delta_window = None
prev_ball_yolo = None
prev_ball_flow = None
prev_gray = None
frame_idx = 0
lost_ball_count = 0

print(f"Video loaded: {total_frames} frames, {fps:.1f} FPS")

# ----------------------
# Main loop
# ----------------------

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ----------------------
    # YOLO detection on key frames
    # ----------------------
    if frame_idx % 100 == 0 and frame_idx != 0:
        curr_time = time.time()
        fps = 100 / (curr_time - prev_time)
        prev_time = curr_time
        print(str(frame_idx) + " frames processed. "  + f"Rate: FPS: {fps:.2f}")


    if frame_idx % key_frame_freq == 0:

        results_ball = ball_model.predict(frame, verbose=False, conf=conf_threshold)[0]
        results_player = player_model.predict(frame, verbose=False)[0]
        best_ball_box = None
        best_conf = 0.0
        players = []

        for box in results_ball.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id == 0 and conf > best_conf:  # ball = 0
                best_conf = conf
                best_ball_box = box

        for box in results_player.boxes:
            cls_id = int(box.cls[0])
            if cls_id == 0:  # person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                players.append((cx, cy, x1, y1, x2, y2))

        if best_ball_box is not None:
            x1, y1, x2, y2 = map(int, best_ball_box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            prev_ball_yolo = (cx, cy, x1, y1, x2, y2)
            kf.correct(np.array([[np.float32(cx)], [np.float32(cy)]]))
            lost_ball_count = 0
        else:
            lost_ball_count += 1

        # compute crop window
        if prev_ball_yolo is not None:
            bx, by, bx1, by1, bx2, by2 = prev_ball_yolo
            nearby_players = [p for p in players if np.hypot(p[0]-bx, p[1]-by)<radius]
            all_x1 = [bx1] + [p[2] for p in nearby_players]
            all_x2 = [bx2] + [p[4] for p in nearby_players]
            all_y1 = [by1] + [p[3] for p in nearby_players]
            all_y2 = [by2] + [p[5] for p in nearby_players]

            x_min = max(0, int(min(all_x1)-padding))
            x_max = min(w, int(max(all_x2)+padding))
            y_min = max(0, int(min(all_y1)-padding))
            y_max = min(h, int(max(all_y2)+padding))
            next_window_raw = maintain_aspect(x_min, y_min, x_max, y_max, video_ratio, w, h)

            if prev_window is not None:
                delta_window = (next_window_raw - prev_window) / key_frame_freq
            else:
                prev_window = next_window_raw.copy()
                delta_window = np.zeros_like(next_window_raw)

    # ----------------------
    # Optical Flow tracking
    # ----------------------
    if prev_ball_yolo is not None and prev_gray is not None:
        prev_pt = np.array([[prev_ball_yolo[0], prev_ball_yolo[1]]], dtype=np.float32).reshape(-1,1,2)
        next_pt, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pt, None, winSize=(15,15))
        if st[0][0]==1:
            fx, fy = next_pt[0][0]
            prev_ball_flow = (fx, fy)
            prediction = kf.predict()
            kf.correct(np.array([[np.float32(fx)], [np.float32(fy)]]))
            px, py = int(prediction[0]), int(prediction[1])
            # update Kalman ball
            prev_ball_kf = (px, py)
        else:
            prediction = kf.predict()
            px, py = int(prediction[0]), int(prediction[1])
            prev_ball_kf = (px, py)
    else:
        # 没有球时Kalman预测
        prediction = kf.predict()
        px, py = int(prediction[0]), int(prediction[1])
        prev_ball_kf = (px, py)

    prev_gray = gray.copy()

    # ----------------------
    # Smooth crop interpolation
    # ----------------------
    if next_window_raw is not None and prev_window is not None:
        t = frame_idx % key_frame_freq
        smoothed_window = prev_window + delta_window * t
        x_min, y_min, x_max, y_max = maintain_aspect(*smoothed_window, video_ratio, w, h).astype(int)
    else:
        # 缺球/未初始化 → 缓慢回退全屏
        x_min, y_min, x_max, y_max = 0, 0, w, h

    crop = frame[y_min:y_max, x_min:x_max]
    if crop.size != 0:
        focused_full = cv2.resize(crop, (w, h))
        out.write(focused_full)

    # ----------------------
    # Visualization
    # ----------------------
    if visualization:
        vis_frame = frame.copy()
        cv2.rectangle(vis_frame, (x_min, y_min), (x_max, y_max), (0,255,0),2)

        if prev_ball_yolo is not None:
            cv2.circle(vis_frame, (int(prev_ball_yolo[0]), int(prev_ball_yolo[1])), 8, (0,255,0), -1)  # YOLO绿点
        if prev_ball_flow is not None:
            cv2.circle(vis_frame, (int(prev_ball_flow[0]), int(prev_ball_flow[1])), 8, (0,0,255), -1)  # 光流红点
        if prev_ball_kf is not None:
            cv2.circle(vis_frame, (int(prev_ball_kf[0]), int(prev_ball_kf[1])), 8, (255,0,0), -1)   # Kalman蓝点

        combo = np.hstack((vis_frame, focused_full))
        cv2.imshow("Focus Box + Ball Tracking", combo)

    # ----------------------
    # Update prev_window
    # ----------------------
    if next_window_raw is not None and frame_idx % key_frame_freq == key_frame_freq-1:
        prev_window = next_window_raw.copy()

    frame_idx +=1
    if cv2.waitKey(1) == 27:  # ESC退出
        break

# ----------------------
# Cleanup
# ----------------------
end_process_time = datetime.datetime.now()
# total_process_time = end_process_time - start_time
# time_per_frame = total_process_time / (frame_idx + 1e-6)
print("frame processed " + str(frame_idx) + " .total time is " + str(end_process_time - start_time))
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Focused video saved to: {out_path}")