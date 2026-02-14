from ultralytics import YOLO
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import (
    RESULTS_DIR,
    CONFIG_BRAIN_YAML,
    YOLO_PROJECT_NAME,
    YOLO_MODEL_SIZE,
    YOLO_TRAIN_CONFIG,
)

def main():
    # 1. Load YOLO26n
    model = YOLO(YOLO_MODEL_SIZE) 

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("🚀 Initiating Anatomically-Preserving ROI Training...")

    # 3. Training Parameters (consolidated from config.YOLO_TRAIN_CONFIG)
    results = model.train(
        data=str(CONFIG_BRAIN_YAML),
        project=str(RESULTS_DIR / "experiments" / "detection"),
        name=YOLO_PROJECT_NAME,
        exist_ok=True,
        **YOLO_TRAIN_CONFIG  # All training hyperparameters from config
    )

    # Optional: evaluate test split after training if defined in brain.yaml
    try:
        model.val(data=str(CONFIG_BRAIN_YAML))
    except Exception:
        pass

    print(f"\n[SUCCESS] Training Complete. Model aligned for consistent lobe detection.")

if __name__ == "__main__":
    main()