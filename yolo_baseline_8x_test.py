import cv2
from ultralytics import YOLO
import os

# ------------------------------
# Configuration
# ------------------------------
video_path = "uploads/V04_10.mp4"
model_path = "yolov8x.pt"
output_frames_dir = "ball_frames"
save_frames = False
output_txt = "detected_frames.txt"
frame_interval = 1  # Process every n-th frame
conf_threshold = 0.25

# ------------------------------
# Initialization
# ------------------------------
model = YOLO(model_path)
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise IOError(f"Cannot open video: {video_path}")

if save_frames:
    os.makedirs(output_frames_dir, exist_ok=True)

frame_id = 0
ball_frames = []

# ------------------------------
# Process each frame
# ------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id % frame_interval == 0:
        results = model.predict(frame, verbose=False, conf=conf_threshold)
        boxes = results[0].boxes
        detected = False

        for box in boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            if cls_id == 32:  # Basketball class
                detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if detected:
            ball_frames.append(frame_id)
            print(f"🏀 Basketball detected at frame {frame_id}")
            if save_frames:
                cv2.imwrite(f"{output_frames_dir}/frame_{frame_id:06d}.jpg", frame)

    # ------------------------------
    # Real-time display
    # ------------------------------
    # cv2.imshow("YOLOv8 Basketball Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()

# ------------------------------
# Save detected frame IDs
# ------------------------------
with open(output_txt, "w") as f:
    for fid in ball_frames:
        f.write(f"{fid}\n")


total_frames = frame_id  # total frames processed
if total_frames > 0:
    ball_ratio = len(ball_frames) / total_frames
else:
    ball_ratio = 0

print(f"\n✅ Done! Total frames with basketball detected: {len(ball_frames)}")
print(f"🏀 Basketball appearance ratio: {ball_ratio*100:.2f}%")
