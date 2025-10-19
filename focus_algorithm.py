from ultralytics import YOLO
import cv2
import numpy as np
import os

# setup
model_path = "yolov8x.pt"
video_path = "tt.mp4"
output_dir = "focus_crops"
padding = 100          # frame padding
radius = 200           # distance of adjacent player from the ball

os.makedirs(output_dir, exist_ok=True)

# load the model
model = YOLO(model_path)
cap = cv2.VideoCapture(video_path)

frame_idx = 0

# track by frame
for result in model.track(source=video_path, stream=True, persist=True):
    frame = result.orig_img.copy()
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        frame_idx += 1
        continue

    # 1 extract ball and player(s)
    players, balls = [], []
    for box in boxes:
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1+x2)//2, (y1+y2)//2
        if cls_id == 0:
            players.append((cx, cy, x1, y1, x2, y2))
        elif cls_id == 32:
            balls.append((cx, cy, x1, y1, x2, y2))

    # 2 calculate focus window
    if len(balls) > 0:
        # grab first ball if more than one
        bx, by, bx1, by1, bx2, by2 = balls[0]

        # find the adjacent players
        nearby_players = [p for p in players if np.hypot(p[0]-bx, p[1]-by) < radius]

        # calculate the bounding window
        all_x1 = [bx1] + [p[2] for p in nearby_players]
        all_x2 = [bx2] + [p[4] for p in nearby_players]
        all_y1 = [by1] + [p[3] for p in nearby_players]
        all_y2 = [by2] + [p[5] for p in nearby_players]

        x_min, x_max = int(min(all_x1) - padding), int(max(all_x2) + padding)
        y_min, y_max = int(min(all_y1) - padding), int(max(all_y2) + padding)

    elif len(players) > 0:
        # if no ball, focus on all players
        all_x1 = [p[2] for p in players]
        all_x2 = [p[4] for p in players]
        all_y1 = [p[3] for p in players]
        all_y2 = [p[5] for p in players]

        x_min, x_max = int(min(all_x1) - padding), int(max(all_x2) + padding)
        y_min, y_max = int(min(all_y1) - padding), int(max(all_y2) + padding)

    else:
        frame_idx += 1
        continue

    # sanity check to avoid overflow
    h, w, _ = frame.shape
    x_min, x_max = max(0, x_min), min(w, x_max)
    y_min, y_max = max(0, y_min), min(h, y_max)

    # 3 draw/visualize the focus window
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.imshow("Focus", frame)
    if cv2.waitKey(1) == 27:  # ESC退出
        break

    # save the focused image from each frame
    crop = frame[y_min:y_max, x_min:x_max]
    if crop.size != 0:
        save_path = os.path.join(output_dir, f"frame{frame_idx:05d}_focus.jpg")
        cv2.imwrite(save_path, crop)

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
print("Cropped focus frames saved in:", output_dir)
