import cv2
import numpy as np
import os
import time
from ultralytics import YOLO

def process_video(
    ball_model_path,
    player_model_path,
    video_path,
    output_path,
    padding=200,
    radius=200,
    key_frame_freq=20,
    conf_threshold=0.25,
    lost_ball_max=25,
    visualization=False,
    device="mps",
    debug_callback=None,      # show visualization
    progress_callback=None    # show progress
):
    """
    Video focus cropping + ball tracking (YOLO + LK Flow + Kalman)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ----------------------
    # Load models
    # ----------------------
    ball_model = YOLO(ball_model_path)
    player_model = YOLO(player_model_path)
    ball_model.to(device)
    player_model.to(device)

    # ----------------------
    # Load video
    # ----------------------
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_ratio = w / (h + 1e-6)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # ----------------------
    # Kalman Filter
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
    # Helpers
    # ----------------------
    def maintain_aspect(x_min, y_min, x_max, y_max):
        crop_w = x_max - x_min
        crop_h = y_max - y_min
        cx = (x_min + x_max) // 2
        cy = (y_min + y_max) // 2
        crop_ratio = crop_w / (crop_h + 1e-6)
        if abs(crop_ratio - video_ratio) > 1e-3:
            if crop_ratio > video_ratio:
                new_h = int(crop_w / video_ratio)
                new_w = crop_w
            else:
                new_w = int(crop_h * video_ratio)
                new_h = crop_h
            x_min = max(0, cx - new_w // 2)
            y_min = max(0, cy - new_h // 2)
            x_max = min(w, x_min + new_w)
            y_max = min(h, y_min + new_h)
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
    prev_ball_kf = None
    frame_idx = 0
    lost_ball_count = 0
    prev_time = time.time()
    fps_now = 0

    print(f"🎬 Loaded video: {total_frames} frames, {fps:.1f} FPS")

    # ----------------------
    # Main loop
    # ----------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ----------------------
        # FPS Calculation
        # ----------------------
        if frame_idx % 100 == 0 and frame_idx != 0:
            curr_time = time.time()
            fps_now = 100 / (curr_time - prev_time)
            prev_time = curr_time


        # ----------------------
        # Key frame detection
        # ----------------------
        if frame_idx % key_frame_freq == 0:
            # Ball detection
            results_ball = ball_model.predict(frame, verbose=False, conf=conf_threshold)[0]
            best_ball_box = None
            best_conf = 0
            for box in results_ball.boxes:
                if int(box.cls[0]) == 0 and float(box.conf[0]) > best_conf:
                    best_conf = float(box.conf[0])
                    best_ball_box = box

            # Player detection
            results_player = player_model.predict(frame, verbose=False)[0]
            players = []
            for box in results_player.boxes:
                if int(box.cls[0]) == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1+x2)//2, (y1+y2)//2
                    players.append((cx, cy, x1, y1, x2, y2))

            # Update ball state
            if best_ball_box is not None:
                x1, y1, x2, y2 = map(int, best_ball_box.xyxy[0])
                cx, cy = (x1+x2)//2, (y1+y2)//2
                prev_ball_yolo = (cx, cy, x1, y1, x2, y2)
                kf.correct(np.array([[np.float32(cx)], [np.float32(cy)]]))
                lost_ball_count = 0
            else:
                lost_ball_count += 1

            # Compute crop window
            if prev_ball_yolo is not None:
                bx, by, bx1, by1, bx2, by2 = prev_ball_yolo
                nearby_players = [p for p in players if np.hypot(p[0]-bx, p[1]-by) < radius]
                all_x1 = [bx1] + [p[2] for p in nearby_players]
                all_x2 = [bx2] + [p[4] for p in nearby_players]
                all_y1 = [by1] + [p[3] for p in nearby_players]
                all_y2 = [by2] + [p[5] for p in nearby_players]

                x_min = max(0, int(min(all_x1)-padding))
                x_max = min(w, int(max(all_x2)+padding))
                y_min = max(0, int(min(all_y1)-padding))
                y_max = min(h, int(max(all_y2)+padding))
                next_window_raw = maintain_aspect(x_min, y_min, x_max, y_max)

                if prev_window is not None:
                    delta_window = (next_window_raw - prev_window) / key_frame_freq
                else:
                    prev_window = next_window_raw.copy()
                    delta_window = np.zeros_like(next_window_raw)

        # ----------------------
        # Optical flow
        # ----------------------
        if prev_ball_yolo is not None and prev_gray is not None:
            prev_pt = np.array([[prev_ball_yolo[0], prev_ball_yolo[1]]], dtype=np.float32).reshape(-1,1,2)
            next_pt, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pt, None)
            if st[0][0]==1:
                fx, fy = next_pt[0][0]
                prev_ball_flow = (fx, fy)
                prediction = kf.predict()
                kf.correct(np.array([[np.float32(fx)], [np.float32(fy)]]))
                prev_ball_kf = (int(prediction[0]), int(prediction[1]))
            else:
                prediction = kf.predict()
                prev_ball_kf = (int(prediction[0]), int(prediction[1]))
        else:
            prediction = kf.predict()
            prev_ball_kf = (int(prediction[0]), int(prediction[1]))

        prev_gray = gray.copy()

        # ----------------------
        # Smooth crop
        # ----------------------
        if next_window_raw is not None and prev_window is not None:
            if lost_ball_count > lost_ball_max:
                # full screen
                t_lost = min(1.0, (lost_ball_count - lost_ball_max) / 10)  # 10 frames
                full_window = np.array([0, 0, w, h], dtype=np.float32)
                smoothed_window = prev_window * (1 - t_lost) + full_window * t_lost
            else:
                t = frame_idx % key_frame_freq
                smoothed_window = prev_window + delta_window * t
            x_min, y_min, x_max, y_max = maintain_aspect(*smoothed_window)
        else:
            x_min, y_min, x_max, y_max = 0, 0, w, h

        x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
        crop = frame[y_min:y_max, x_min:x_max]
        if crop.size != 0:
            focused_full = cv2.resize(crop, (w, h))
            out.write(focused_full)

        # ----------------------
        # Visualization
        # ----------------------
        if visualization and debug_callback is not None and frame_idx % 5 == 0:
            vis = frame.copy()
            cv2.rectangle(vis, (x_min,y_min), (x_max,y_max), (0,255,0), 2)
            if prev_ball_yolo is not None:
                cv2.circle(vis, (int(prev_ball_yolo[0]), int(prev_ball_yolo[1])), 8, (0,255,0), -1)
            if prev_ball_flow is not None:
                cv2.circle(vis, (int(prev_ball_flow[0]), int(prev_ball_flow[1])), 8, (0,0,255), -1)
            if prev_ball_kf is not None:
                cv2.circle(vis, (prev_ball_kf[0], prev_ball_kf[1]), 8, (255,0,0), -1)
            combo = np.hstack((vis, focused_full))
            display_frame = cv2.resize(combo, (combo.shape[1]//2, combo.shape[0]//2))
            debug_callback(display_frame)

        # ----------------------
        # Progress callback
        # ----------------------
        if progress_callback is not None:
            progress_callback(frame_idx, total_frames, fps_now)

        if next_window_raw is not None and frame_idx % key_frame_freq == key_frame_freq-1:
            prev_window = next_window_raw.copy()

        frame_idx += 1

    cap.release()
    out.release()
    print(f"✅ Saved focused video to: {output_path}")
    return output_path