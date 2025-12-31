from ultralytics import YOLO
import os

def main():
    # 1. Load YOLOv11 (Pre-trained weights)
    # yolo11n is the Nano version: fast, lightweight, and great for feature extraction
    model = YOLO("yolo11n.pt")

    # 2. Configure Paths
    # We ensure the results directory exists to avoid I/O errors
    os.makedirs("./results", exist_ok=True)

    print("Starting YOLO Training for Brain ROI Detection...")
    print("Goal: Extract Graph Nodes for Causal Analysis")

    # 3. Training parameters
    results = model.train(
        data="./configs/brain.yaml",      
        epochs=100,             
        imgsz=640,              
        batch=24,               
        device=0,               
        project="./results",    
        name="ROI_Detection_v20", 
        plots=True,             # Essential for explainability (shows confusion matrix)
        save=True,              # Saves best.pt for your GNN pipeline
        val=True,               
        patience=20,            # Stop early if the model stops improving
        workers=8,              
        optimizer='AdamW',      # Superior to SGD for medical imaging tasks
        lr0=0.01,               
        cos_lr=True,            # Smoothly decays learning rate
        overlap_mask=False,     # Keeps lobe bboxes distinct for cleaner node extraction
        box=7.5,                # Increases weight of bounding box accuracy
        cls=0.5                 # Classification weight
    )

    # 4. Exporting for the next phase (GNN)
    # Exporting to ONNX or TorchScript can make feature extraction faster later
    # model.export(format="onnx")

    print(f"\n[SUCCESS] Training Complete.")
    print(f"Weights saved to: ./results/ROI_Detection_v20/weights/best.pt")
    print(f"Examine 'results.png' and 'confusion_matrix.png' for explainability metrics.")

if __name__ == "__main__":
    main()