import logging

from ultralytics import YOLO

from src.core.config import (
    CONFIG_BRAIN_YAML,
    RESULTS_DIR,
    YOLO_MODEL_SIZE,
    YOLO_PROJECT_NAME,
    YOLO_TRAIN_CONFIG,
)

logger = logging.getLogger(__name__)


def main():
    # 1. Load YOLO26n
    model = YOLO(YOLO_MODEL_SIZE)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Initiating Anatomically-Preserving ROI Training...")

    # 2. Training Parameters (consolidated from config.YOLO_TRAIN_CONFIG)
    model.train(
        data=str(CONFIG_BRAIN_YAML),
        project=str(RESULTS_DIR / "experiments" / "detection"),
        name=YOLO_PROJECT_NAME,
        exist_ok=True,
        **YOLO_TRAIN_CONFIG,  # All training hyperparameters from config
    )

    # Optional: evaluate test split after training if defined in brain.yaml
    try:
        model.val(data=str(CONFIG_BRAIN_YAML))
    except Exception as e:
        logger.warning("Post-training validation skipped: %s", e)

    logger.info("Training Complete. Model aligned for consistent lobe detection.")


if __name__ == "__main__":
    main()
