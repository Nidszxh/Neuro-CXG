from ultralytics import YOLO
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import RESULTS_DIR, CONFIG_BRAIN_YAML

def main():
    # 1. Load YOLO11s 
    # Small is the correct choice for capturing the boundaries of the 5 lobe groups.
    model = YOLO("yolo11s.pt") 

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("🚀 Initiating Anatomically-Preserving ROI Training...")

    # 3. Corrected Training Parameters
    results = model.train(
        data=str(CONFIG_BRAIN_YAML),      
        epochs=100,             
        imgsz=640,              
        batch=24,                   # RTX 4060 8GB can handle batch 24 at 640px (note:b24 5.94 GB mem_use, b32 dies ie OOM)
        device=0,               
        project=str(RESULTS_DIR),    
        name="ROI_Detection_v20", 
        seed=42,                
        deterministic=True,     
        plots=True,             
        save=True,              
        val=True,               
        patience=25,            
        workers=8,              
        optimizer='AdamW',      
        
        # --- CRITICAL CORRECTIONS START HERE ---
        
        lr0=0.001,              
        label_smoothing=0.0,    # Disable for 5 broad classes; boundaries are distinct enough.
        
        # Balance Box vs Class: 
        # Previously, box=10.0 caused the model to ignore WHAT the lobe was.
        box=7.5,                
        cls=2.0,                # Increase Class weight to ensure 'Frontal' isn't confused with 'Parietal'
        
        # Spatial Augmentation - ANATOMICAL PROTECTION:
        hsv_h=0.0,              
        hsv_s=0.0,
        hsv_v=0.1,              # Minimal brightness variation
        
        degrees=0.0,            # CRITICAL: Set to 0. Rotation ruins 3D Z-axis alignment.
        
        fliplr=0.0,             # CRITICAL: Set to 0. Flipping the brain makes Left = Right.
                                # This is why your model cannot learn connectivity patterns.
        
        flipud=0.0,             # Keep superior/inferior orientation constant.
        
        mosaic=0.0,             # Disable Mosaic for medical images; it breaks global spatial context.
        mixup=0.0               # Disable Mixup; biological features should not be blended.
    )

    print(f"\n[SUCCESS] Training Complete. Model aligned for consistent lobe detection.")

if __name__ == "__main__":
    main()