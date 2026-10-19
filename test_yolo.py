from ultralytics import YOLO

# ✅ Use your actual model path here:
model = YOLO("E:/bone_fracture_system/quick_train/fracture_model/weights/best.pt")

print("✅ Model loaded successfully!")
print("Classes:", model.names)
