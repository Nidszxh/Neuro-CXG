from ultralytics import YOLO
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import (
    RESULTS_DIR,
    CONFIG_BRAIN_YAML,
    YOLO_PROJECT_NAME,
    YOLO_MODEL_SIZE,
    YOLO_EPOCHS,
    YOLO_IMGSZ,
    YOLO_BATCH_SIZE,
    YOLO_HSV_H,
    YOLO_HSV_S,
    YOLO_HSV_V,
    YOLO_DEGREES,
    YOLO_FLIPLR,
    YOLO_FLIPUD,
    YOLO_MOSAIC,
)

def main():
    # 1. Load YOLO11s 
    model = YOLO(YOLO_MODEL_SIZE) 

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("🚀 Initiating Anatomically-Preserving ROI Training...")

    # 3. Training Parameters (from config)
    results = model.train(
        data=str(CONFIG_BRAIN_YAML),
        epochs=YOLO_EPOCHS,
        imgsz=YOLO_IMGSZ,
        batch=YOLO_BATCH_SIZE,
        device=0,
        project=str(RESULTS_DIR),
        name=YOLO_PROJECT_NAME,
        seed=42,
        deterministic=True,
        plots=True,
        save=True,
        val=True,
        patience=25,
        workers=8,
        optimizer='AdamW',
        lr0=0.001,
        label_smoothing=0.0,
        box=7.5,
        cls=2.0,
        # Spatial Augmentation - ANATOMICAL PROTECTION (from config)
        hsv_h=YOLO_HSV_H,
        hsv_s=YOLO_HSV_S,
        hsv_v=YOLO_HSV_V,
        degrees=YOLO_DEGREES,
        fliplr=YOLO_FLIPLR,
        flipud=YOLO_FLIPUD,
        mosaic=YOLO_MOSAIC,
        mixup=0.0
    )

    # Optional: evaluate test split after training if defined in brain.yaml
    try:
        model.val(data=str(CONFIG_BRAIN_YAML))
    except Exception:
        pass

    print(f"\n[SUCCESS] Training Complete. Model aligned for consistent lobe detection.")

if __name__ == "__main__":
    main()