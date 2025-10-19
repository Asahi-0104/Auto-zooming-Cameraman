from ultralytics import YOLO

# 1 load the model
model = YOLO("yolov8x.pt")

# 2 predict the model
results = model.predict(
    source="tt.mp4",
    save=True,
    show=False
)

# 3 result location
print("Results saved to:", results[0].save_dir)
