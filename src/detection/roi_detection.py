import sys
import logging
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import (
    RESULTS_DIR,
    CONFIG_BRAIN_YAML,
    YOLO_PROJECT_NAME,
    YOLO_MODEL_SIZE,
    YOLO_TRAIN_CONFIG,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    # 1. Load YOLO26n
    model = YOLO(YOLO_MODEL_SIZE) 
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("🚀 Initiating Anatomically-Preserving ROI Training...")

    # 2. Training Parameters (consolidated from config.YOLO_TRAIN_CONFIG)
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
    except Exception as e:
        logger.warning("Post-training validation skipped: %s", e)

    print(f"\n[SUCCESS] Training Complete. Model aligned for consistent lobe detection.")

if __name__ == "__main__":
    main()