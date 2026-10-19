# quick_train.py
from ultralytics import YOLO
import os
import shutil

def quick_train():
    print("🚀 Starting Quick YOLO Training...")
    
    # Check dataset
    if not os.path.exists('bone_fracture_dataset/dataset.yaml'):
        print("❌ Dataset not found! Please check your dataset structure.")
        return
    
    # Load pre-trained model
    model = YOLO('yolov8n.pt')
    
    # Train with minimal settings for speed
    results = model.train(
        data='bone_fracture_dataset/dataset.yaml',
        epochs=20,           # Reduced for quick training
        imgsz=320,           # Smaller images = faster training
        batch=8,             # Smaller batch size
        device='cpu',        # Use CPU (change to '0' for GPU)
        workers=2,           # Fewer workers
        patience=5,          # Early stopping
        save=True,
        project='quick_train',
        name='fracture_model',
        exist_ok=True,
        verbose=True
    )
    
    print("✅ Training completed!")
    
    # Copy the best model
    best_model_path = 'quick_train/fracture_model/weights/best.pt'
    if os.path.exists(best_model_path):
        shutil.copy(best_model_path, 'models/best.pt')
        print(f"✅ Best model copied to: models/best.pt")
    else:
        print("❌ No best model found!")

if __name__ == '__main__':
    quick_train()