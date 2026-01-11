from ultralytics import YOLO
import os

def main():
    # 1. Load YOLO11 - Using 's' (Small) cause its better than nano
    # The 's' model has more parameters to capture the subtle curvature of brain lobes
    model = YOLO("yolo11s.pt") 

    os.makedirs("./results", exist_ok=True)

    print("🚀 Initiating Q1-Standard ROI Training...")

    # 3. Optimized Training Parameters (Phase 3.3)
    results = model.train(
        data="./configs/brain.yaml",      
        epochs=100,             
        imgsz=640,              
        batch=24,               # RTX 4060 8GB can handle batch 24 at 640px (note:b24 5.94 GB mem_use, b32 dies ie OOM)
        device=0,               
        project="./results",    
        name="ROI_Detection_v20_Final", 
        seed=42,                # Mandatory for Scientific Reproducibility
        deterministic=True,      # Forces consistent results across runs
        plots=True,             
        save=True,              
        val=True,               
        patience=25,            
        workers=8,              
        optimizer='AdamW',      
        lr0=0.001,              # Slightly lower LR for medical stability
        lrf=0.01,               
        cos_lr=True,            
        label_smoothing=0.1,    # Handles 'fuzzy' brain boundaries (Q1 standard)
        box=10.0,               # Higher weight on BOX accuracy for Graph Node stability
        cls=1.0,                
        hsv_h=0.0,              # Medical images don't have color variation; disable HSV
        hsv_s=0.0,
        hsv_v=0.2,              # Only vary brightness (simulates scan intensity)
        degrees=10,             # Subtle rotation for head-tilt simulation
        fliplr=0.5,             # Symmetrical augmentation (Left/Right)
        flipud=0.0              # Anatomical protection (Superior/Inferior)
    )

    print(f"\n[SUCCESS] Training Complete. Model ready for Phase 3.5.")

if __name__ == "__main__":
    main()