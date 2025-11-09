from ultralytics import YOLO
import cv2
import os

# ----------------------
# Configuration
# ----------------------
model_path = "runs/detect/train8/weights/best.pt"  # Your trained YOLO model
video_path = "uploads/V04_10.mp4"

# ----------------------
# 1️⃣ Load YOLO model
# ----------------------
model = YOLO(model_path)

# ----------------------
# 2️⃣ Open video for frame-by-frame analysis
# ----------------------
cap = cv2.VideoCapture(video_path)
frame_idx = 0
has_ball_frames = []  # Record whether a basketball is detected in each frame
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video loaded: {frame_count} frames, {fps:.1f} FPS")

# ----------------------
# 3️⃣ Process each frame
# ----------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    # YOLO inference (disable verbose for speed)
    results = model.predict(
        source=frame,
        conf=0.5,       # Confidence threshold, adjust as needed
        verbose=False
    )

    # Check for detection results
    boxes = results[0].boxes
    has_ball = False

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0])  # Class ID
            conf = float(box.conf[0]) # Confidence

            # If it's the "basketball" class (assuming only one class, cls_id=0)
            if cls_id == 0 and conf > 0.5:
                has_ball = True
                break

    has_ball_frames.append(has_ball)

    # Optional: print progress every 50 frames
    if frame_idx % 50 == 0:
        print(f"Processed frame {frame_idx}/{frame_count}, has_ball={has_ball}")

cap.release()

# ----------------------
# 4️⃣ Statistics
# ----------------------
ball_ratio = sum(has_ball_frames) / len(has_ball_frames) if has_ball_frames else 0
print("✅ Video analysis completed")
print(f"📊 Frames with basketball detected: {sum(has_ball_frames)} / {len(has_ball_frames)}")
print(f"🏀 Basketball appearance ratio: {ball_ratio*100:.2f}%")