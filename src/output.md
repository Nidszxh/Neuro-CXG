  source /home/nidszxh/.venv312/bin/activate
❯ python src/run_pipeline.py --skip-download --force-reset --auto

======================================================================
NEURO-CXG PIPELINE RUNNER
12-Region Causal GNN for fMRI Analysis
======================================================================
INFO:pipeline:
Stage 0: Pre-Flight Validation
INFO:src.core.config:VALIDATING NEURO-CXG 12-REGION ARCHITECTURE
INFO:src.core.config:✓ validate_lobe_mapping: 12 regions, 170 ROIs, no duplicates, full coverage
INFO:src.core.config:✓ Target: 12 nodes | Features: 28
INFO:src.core.config:✓ Device: cuda
INFO:pipeline:Environment validation passed
WARNING:pipeline:Train/val/test splits not found.

======================================================================
EXECUTION PLAN
======================================================================
1. ABIDE Download                           ○ SKIP          Download ABIDE fMRI data + 7-slice ALFF export (Stage 1)
2. Train/Val/Test Split (2D Stratified)     ✓ WILL RUN      2D stratification by DX_GROUP + SITE_ID (Stage 2)
3. Generate Master Manifest                 ✓ WILL RUN      Maps subjects to phenotypes (Stage 3)
4. Atlas Validation                         ✓ WILL RUN      Verify AAL3 atlas files exist and are valid (Stage 4)
5. Pipeline Validation (Comprehensive Health Check) ✓ WILL RUN      Full pipeline health check (Stage 5)
6. Post-Download Integrity Check            ✓ WILL RUN      Validate PNG/NPY files after download (Stage 6)
7. Atlas-Based Label Annotation             ✓ WILL RUN      Generate YOLO training labels from AAL3 atlas (Stage 7)
8. YOLO Training (ROI Detection)            ✓ WILL RUN      Force retrain
9. Spatial Feature Extraction (12-region)   ✓ WILL RUN      YOLO inference → 3D spatial coords aggregation (Stage 9)
10. Temporal Feature Extraction              ✓ WILL RUN      20 features per ROI: 8 time-domain + 12 frequency (Stage 10)
11. Feature Harmonization                    ✓ WILL RUN      Fold-safe neuroHarmonize, protects DX_GROUP (Stage 11)
12. Pre-GNN Integrity Check                  ✓ WILL RUN      Validate dataset completeness per split (Stage 12)
13. Pipeline Diagnostics                     ✓ WILL RUN      Comprehensive health report after graphs built (Stage 13)
14. Quality Validation (YOLO & Graph Sparsity) ✓ WILL RUN      YOLO quality, graph sparsity, stratification (Stage 14)
15. Causal Graph Construction (12×12)        ✓ WILL RUN      Granger causality/lagged correlation graphs (Stage 15)
16. GNN Training (5-Fold CV)                 ✓ WILL RUN      Main training phase (Phase 3)
17. Generate Visualizations                  ✓ WILL RUN      Generate comprehensive visualizations (Phase 9 Reporting)
18. Comprehensive Evaluation                 ✓ WILL RUN      Ensemble evaluation, bootstrap CI, permutation test, subgroups (Phase 9.2)
19. Explainability Analysis                  ✓ WILL RUN      Node/edge importance, feature attribution, literature validation (Phase 8)
20. Result Interpretation & Analysis         ✓ WILL RUN      Per-subject predictions, misclassification analysis, site effects (Phase 9.3)
======================================================================

INFO:pipeline:Cleaning legacy pipeline state for 12-region alignment...
INFO:pipeline:Reset Causal Graph directory (cleared old 170x170 matrices)
INFO:pipeline:⏭️  Skipping: ABIDE Download
INFO:pipeline:Running: Train/Val/Test Split (2D Stratified)
INFO:__main__:Filtered to 1025 subjects with valid stratification groups
INFO:__main__:📦 Organizing train set (717 subjects)...
INFO:__main__:📦 Organizing val set (154 subjects)...
INFO:__main__:📦 Organizing test set (154 subjects)...
INFO:__main__:
✅ SUCCESS: Stratified split complete. Saved to /home/nidszxh/Projects/Neuro-CXG/data/final
INFO:pipeline:Completed: Train/Val/Test Split (2D Stratified)
INFO:pipeline:Running: Generate Master Manifest
WARNING:__main__:TR column missing from phenotype CSV — defaulting to TR=2.0 s for all subjects
INFO:__main__:Manifest successfully synchronized with 1025 subjects.
INFO:__main__:Breakdown:
DX_GROUP    1    2
split             
test       75   79
train     348  369
val        75   79
INFO:pipeline:Completed: Generate Master Manifest
INFO:pipeline:Running: Atlas Validation
<frozen runpy>:128: RuntimeWarning: 'src.validation.atlas_validator' found in sys.modules after import of package 'src.validation', but prior to execution of 'src.validation.atlas_validator'; this may result in unpredictable behaviour
INFO:__main__:============================================================
INFO:__main__:ATLAS SETUP
INFO:__main__:============================================================
INFO:__main__:Scanning atlas directory: /home/nidszxh/Projects/Neuro-CXG/data/raw/atlases
INFO:__main__:Checking AAL3v1.nii
INFO:__main__:✓ Valid atlas | Shape=(91, 109, 91) | ROIs=166
INFO:__main__:✓ Found valid atlas: /home/nidszxh/Projects/Neuro-CXG/data/raw/atlases/AAL3v1.nii
INFO:__main__:✓ Using existing atlas: AAL3v1.nii
INFO:__main__:✓ Metadata generated: /home/nidszxh/Projects/Neuro-CXG/data/metadata/atlas_metadata.json
✅ Atlas ready
INFO:pipeline:Completed: Atlas Validation
INFO:pipeline:Running: Pipeline Validation (Comprehensive Health Check)
<frozen runpy>:128: RuntimeWarning: 'src.validation.pipeline_checks' found in sys.modules after import of package 'src.validation', but prior to execution of 'src.validation.pipeline_checks'; this may result in unpredictable behaviour
INFO:__main__:Starting comprehensive pipeline validation...
INFO:__main__:======================================================================
INFO:__main__:STAGE 1: ENVIRONMENT VALIDATION
INFO:__main__:======================================================================
INFO:__main__:[Environment] Python 3.12
INFO:__main__:[Environment] CUDA available: NVIDIA GeForce RTX 4060 Laptop GPU
INFO:__main__:[Environment] Feature dims: GNN_IN_CHANNELS=28 (20 temporal + 6 spatial + internal)
INFO:__main__:Checking atlas...
INFO:__main__:[Atlas] Atlas loaded: 166 ROIs
INFO:__main__:Checking LOBE_MAPPING...
INFO:__main__:[Config] LOBE_MAPPING valid: 12 lobes, 170 ROIs
INFO:__main__:
======================================================================
INFO:__main__:STAGE 2: DATA VALIDATION
INFO:__main__:======================================================================
INFO:__main__:[Data] 1025 subjects with complete data
ERROR:__main__:[Data] Sample validation: 10 corrupted, 0 wrong shape
ERROR:__main__:  Fix: Re-run data download
INFO:__main__:Checking manifest...
INFO:__main__:[Manifest] 1025 subjects across 3 splits
INFO:__main__:
======================================================================
INFO:__main__:STAGE 3: FEATURE VALIDATION
INFO:__main__:======================================================================
ERROR:__main__:[Features] Temporal features not found
ERROR:__main__:  Fix: Run: python -m src.features.extract_temporal
INFO:__main__:
======================================================================
INFO:__main__:STAGE 4: GRAPH VALIDATION
INFO:__main__:======================================================================
ERROR:__main__:[Graphs] No graph files found
ERROR:__main__:  Fix: Run: python -m src.features.construct_causal
INFO:__main__:Checking stratification...
INFO:__main__:[Stratification] No data leakage across 3 splits
INFO:__main__:
======================================================================
INFO:__main__:STAGE 5: MODEL VALIDATION
INFO:__main__:======================================================================
INFO:__main__:[Models] 5 trained models, mean AUC: 0.6194, F1: 0.7132
INFO:__main__:
======================================================================
INFO:__main__:VALIDATION REPORT
INFO:__main__:======================================================================
ERROR:__main__:
CRITICAL ISSUES (3):
ERROR:__main__:----------------------------------------------------------------------
ERROR:__main__:  [Data] Sample validation: 10 corrupted, 0 wrong shape
ERROR:__main__:    -> Fix: Re-run data download
ERROR:__main__:  [Features] Temporal features not found
ERROR:__main__:    -> Fix: Run: python -m src.features.extract_temporal
ERROR:__main__:  [Graphs] No graph files found
ERROR:__main__:    -> Fix: Run: python -m src.features.construct_causal
INFO:__main__:
PASSED CHECKS (9):
INFO:__main__:----------------------------------------------------------------------
INFO:__main__:  Python 3.12
INFO:__main__:  CUDA available: NVIDIA GeForce RTX 4060 Laptop GPU
INFO:__main__:  Feature dims: GNN_IN_CHANNELS=28 (20 temporal + 6 spatial + internal)
INFO:__main__:  Atlas loaded: 166 ROIs
INFO:__main__:  LOBE_MAPPING valid: 12 lobes, 170 ROIs
INFO:__main__:  1025 subjects with complete data
INFO:__main__:  1025 subjects across 3 splits
INFO:__main__:  No data leakage across 3 splits
INFO:__main__:  5 trained models, mean AUC: 0.6194, F1: 0.7132
INFO:__main__:
======================================================================
INFO:__main__:SUMMARY:
INFO:__main__:  Passed: 9
INFO:__main__:  Warnings: 0
INFO:__main__:  Critical: 3
INFO:__main__:======================================================================
ERROR:__main__:
PIPELINE HAS CRITICAL ISSUES
INFO:pipeline:Completed: Pipeline Validation (Comprehensive Health Check)
INFO:pipeline:Running: Post-Download Integrity Check
INFO:src.validation.pipeline_checks:Starting post-download integrity check...
INFO:src.validation.pipeline_checks:No images found to check.
INFO:src.validation.pipeline_checks:Scanning 0 time-series files...
INFO:src.validation.pipeline_checks:
========================================
INFO:src.validation.pipeline_checks:POST-DOWNLOAD INTEGRITY REPORT
INFO:src.validation.pipeline_checks:========================================
INFO:src.validation.pipeline_checks:Corrupted PNGs:      0
INFO:src.validation.pipeline_checks:Corrupted NPYs:      0
INFO:src.validation.pipeline_checks:Incomplete Subjects: 0 (Missing slices or TS)
INFO:src.validation.pipeline_checks:----------------------------------------
INFO:pipeline:Completed: Post-Download Integrity Check
INFO:pipeline:Running: Atlas-Based Label Annotation
INFO:__main__:Pre-calculating atlas bounding boxes for 7 percentile slices...
INFO:__main__:Generated annotations for 7 z-slices (atlas z_dim=91)
INFO:__main__:Annotating train split (5019 images)...
100%|████████████████████████████████████████████████████████████████████████████████████████████████| 5019/5019 [00:00<00:00, 149083.68it/s]
INFO:__main__:Annotating val split (1078 images)...
100%|████████████████████████████████████████████████████████████████████████████████████████████████| 1078/1078 [00:00<00:00, 163601.68it/s]
INFO:__main__:Annotating test split (1078 images)...
100%|████████████████████████████████████████████████████████████████████████████████████████████████| 1078/1078 [00:00<00:00, 137631.19it/s]
INFO:__main__:Annotation complete. Labels synced with split images.
INFO:pipeline:Completed: Atlas-Based Label Annotation
INFO:pipeline:Running: YOLO Training (ROI Detection)
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 9% ━─────────── 496.0KB/5.3MB 4.8MB/s 
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 21% ━━╸───────── 1.1/5.3MB 6.0MB/s 0.2
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 37% ━━━━──────── 2.0/5.3MB 8.3MB/s 0.3
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 52% ━━━━━━────── 2.8/5.3MB 7.2MB/s 0.4
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 70% ━━━━━━━━──── 3.7/5.3MB 9.0MB/s 0.5
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 89% ━━━━━━━━━━╸─ 4.7/5.3MB 7.8MB/s 0.6
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 100% ━━━━━━━━━━━━ 5.3MB 7.1MB/s 0.7s
🚀 Initiating Anatomically-Preserving ROI Training...
New https://pypi.org/project/ultralytics/8.4.19 available 😃 Update with 'pip install -U ultralytics'
WARNING ⚠️ 'label_smoothing' is deprecated and will be removed in the future.
Ultralytics 8.4.5 🚀 Python-3.12.12 torch-2.9.0+cu128 CUDA:0 (NVIDIA GeForce RTX 4060 Laptop GPU, 7806MiB)
engine/trainer: agnostic_nms=False, amp=True, angle=1.0, augment=False, auto_augment=randaugment, batch=32, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=2.0, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=/home/nidszxh/Projects/Neuro-CXG/configs/brain.yaml, degrees=0.0, deterministic=True, device=0, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, epochs=100, erasing=0.4, exist_ok=True, fliplr=0.0, flipud=0.0, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.0, hsv_s=0.0, hsv_v=0.1, imgsz=640, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.001, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=yolo26n.pt, momentum=0.937, mosaic=0.0, multi_scale=0.0, name=ROI_Detection_v28, nbs=64, nms=False, opset=None, optimize=False, optimizer=AdamW, overlap_mask=True, patience=25, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=/home/nidszxh/Projects/Neuro-CXG/results/experiments/detection, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/home/nidszxh/Projects/Neuro-CXG/results/experiments/detection/ROI_Detection_v28, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.5, seed=42, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.1, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
Overriding model.yaml nc=80 with nc=12

                   from  n    params  module                                       arguments                     
  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]                 
  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]                
  2                  -1  1      6640  ultralytics.nn.modules.block.C3k2            [32, 64, 1, False, 0.25]      
  3                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                
  4                  -1  1     26080  ultralytics.nn.modules.block.C3k2            [64, 128, 1, False, 0.25]     
  5                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
  6                  -1  1     87040  ultralytics.nn.modules.block.C3k2            [128, 128, 1, True]           
  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]              
  8                  -1  1    346112  ultralytics.nn.modules.block.C3k2            [256, 256, 1, True]           
  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5, 3, True]        
 10                  -1  1    249728  ultralytics.nn.modules.block.C2PSA           [256, 256, 1]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  1    119808  ultralytics.nn.modules.block.C3k2            [384, 128, 1, True]           
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  1     34304  ultralytics.nn.modules.block.C3k2            [256, 64, 1, True]            
 17                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  1     95232  ultralytics.nn.modules.block.C3k2            [192, 128, 1, True]           
 20                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  1    463104  ultralytics.nn.modules.block.C3k2            [384, 256, 1, True, 0.5, True]
 23        [16, 19, 22]  1    245856  ultralytics.nn.modules.head.Detect           [12, 1, True, [64, 128, 256]] 
YOLO26n summary: 260 layers, 2,508,480 parameters, 2,508,480 gradients, 5.8 GFLOPs

Transferred 606/708 items from pretrained weights
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed ✅
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 68.8±47.8 MB/s, size: 44.5 KB)
train: Scanning /home/nidszxh/Projects/Neuro-CXG/data/final/train/labels.cache... 1434 images, 3585 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 5019/5019 1.5Git/s 0.0s
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 3153.5±1813.9 MB/s, size: 50.0 KB)
val: Scanning /home/nidszxh/Projects/Neuro-CXG/data/final/val/labels.cache... 308 images, 770 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 1078/1078 141.3Mit/s 0.0s
Plotting labels to /home/nidszxh/Projects/Neuro-CXG/results/experiments/detection/ROI_Detection_v29/labels.jpg... 
optimizer: AdamW(lr=0.001, momentum=0.937) with parameter groups 114 weight(decay=0.0), 126 weight(decay=0.0005), 126 bias(decay=0.0)
/home/nidszxh/.venv312/lib64/python3.12/site-packages/mlflow/tracking/_tracking_service/utils.py:140: FutureWarning: Filesystem tracking backend (e.g., './mlruns') is deprecated. Please switch to a database backend (e.g., 'sqlite:///mlflow.db'). For feedback, see: https://github.com/mlflow/mlflow/issues/18534
  return FileStore(store_uri, store_uri)
MLflow: logging run_id(470fde785a1144e3bfbbec700e3a870c) to runs/mlflow
MLflow: view at http://127.0.0.1:5000 with 'mlflow server --backend-store-uri runs/mlflow'
MLflow: disable with 'yolo settings mlflow=False'
Image sizes 640 train, 640 val
Using 8 dataloader workers
Logging results to /home/nidszxh/Projects/Neuro-CXG/results/experiments/detection/ROI_Detection_v29
Starting training for 100 epochs...


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      1/100      4.77G       1.68       14.9    0.03076         51        640: 100% ━━━━━━━━━━━━ 157/157 3.6it/s 44.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 3.1it/s 5.4s
                   all       1078       2002      0.252      0.532      0.306      0.122

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      2/100      4.81G      1.183      5.972    0.01926         47        640: 100% ━━━━━━━━━━━━ 157/157 4.5it/s 34.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.508      0.693       0.63      0.411

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      3/100       4.8G      1.047      4.566    0.01667         58        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.744      0.758      0.841      0.562

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      4/100      4.79G      1.036      3.941    0.01677         28        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 5.9it/s 2.9s
                   all       1078       2002      0.759      0.797       0.88      0.614

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      5/100      4.79G     0.9345      3.236    0.01461         39        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.792      0.851      0.923      0.722

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      6/100      4.79G     0.9252      2.894    0.01473         68        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.793      0.898      0.925      0.721

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      7/100      4.79G     0.8681      2.639    0.01333         45        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 5.9it/s 2.9s
                   all       1078       2002      0.764      0.869      0.913      0.685

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      8/100      4.81G     0.8396      2.404    0.01278         68        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.871       0.91      0.962      0.778

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      9/100      4.79G     0.7883       2.17    0.01192         37        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002       0.86      0.916      0.958      0.799

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     10/100      4.79G     0.7751      2.135    0.01185         54        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.856      0.876      0.941      0.783

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     11/100       4.8G     0.7261      1.923    0.01062         49        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.929      0.917      0.972      0.832

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     12/100      4.79G     0.7158       1.84    0.01085         26        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.879      0.952      0.971      0.834

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     13/100      4.79G      0.718      1.836    0.01092         49        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.946      0.933      0.979       0.83

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     14/100      4.79G     0.6751      1.646    0.01004         26        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.911      0.935      0.975      0.848

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     15/100       4.8G     0.6913      1.689    0.01009         48        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.918      0.941      0.975      0.825

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     16/100      4.79G     0.7006      1.659    0.01046         49        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.911      0.949      0.975      0.839

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     17/100       4.8G     0.6772       1.57    0.01015         57        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.927      0.951      0.982      0.853

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     18/100       4.8G     0.6488      1.499   0.009507         49        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.945      0.958      0.985      0.871

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     19/100      4.79G     0.6504      1.494   0.009535         63        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002       0.94      0.925      0.982      0.862

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     20/100       4.8G     0.6208       1.43    0.00914         75        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.931      0.928      0.982      0.887

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     21/100      4.79G     0.6062      1.401   0.008676         54        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.958       0.96      0.987        0.9

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     22/100       4.8G     0.6078      1.353   0.008711         55        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.911      0.948       0.98      0.872

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     23/100       4.8G      0.588      1.309   0.008433         36        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.948      0.945       0.98      0.852

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     24/100      4.79G     0.5889      1.307    0.00861         70        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.952      0.946      0.986      0.902

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     25/100      4.79G     0.5494      1.256   0.007932         55        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.953      0.963      0.988      0.897

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     26/100      4.79G     0.5981      1.268   0.008531         63        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002       0.93      0.935      0.981      0.866

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     27/100      4.79G     0.5979      1.221   0.008606         67        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.959      0.941      0.988      0.897

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     28/100       4.8G     0.5421       1.17   0.007829         72        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.945      0.954      0.985      0.891

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     29/100      4.79G     0.5483      1.147   0.007752         62        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.962      0.957      0.986      0.862

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     30/100      4.81G      0.525      1.101   0.007436         76        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.949      0.963      0.986      0.892

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     31/100       4.8G     0.5433      1.162   0.007637         86        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.958      0.979       0.99      0.918

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     32/100      4.79G     0.5107      1.085    0.00728         36        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.968      0.966      0.989      0.918

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     33/100      4.81G     0.5273      1.094   0.007426         63        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.962      0.968      0.988      0.889
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     34/100       4.8G     0.5413      1.077   0.007662         47        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.972      0.973       0.99      0.924

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     35/100       4.8G     0.4821      1.025   0.006655         80        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.954      0.969      0.989      0.914

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     36/100      4.79G      0.486      1.034   0.006639         23        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.957      0.956       0.99      0.916

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     37/100      4.79G     0.4966      1.027   0.006845         46        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.964      0.972      0.989      0.912

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     38/100      4.79G     0.4904     0.9834   0.007008         52        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.966      0.974      0.991       0.92

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     39/100      4.81G     0.4852      1.051   0.006706         28        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.964       0.96      0.988      0.923

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     40/100       4.8G     0.4889     0.9907   0.006916         52        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.971      0.953      0.989      0.929

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     41/100      4.79G     0.4822     0.9792    0.00666         49        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.969      0.957      0.989      0.927

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     42/100       4.8G     0.4754     0.9361   0.006625         60        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.959       0.97      0.989      0.921

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     43/100      4.79G     0.4647     0.9304   0.006482         44        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.969      0.969      0.989      0.923

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     44/100      4.79G     0.4535     0.8975   0.006314         31        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002       0.97      0.974       0.99      0.933

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     45/100      4.79G     0.4449     0.8907   0.006229         55        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.973      0.964      0.991      0.941

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     46/100      4.79G     0.4495     0.8973    0.00615         52        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.977       0.98      0.992      0.939

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     47/100      4.79G     0.4303     0.8416   0.005778         65        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.975      0.966      0.991      0.933

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     48/100       4.8G     0.4329     0.8684   0.005856         28        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.971      0.971       0.99       0.92

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     49/100      4.82G     0.4346     0.8455   0.005928         75        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.974      0.977      0.993      0.942

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     50/100       4.8G     0.4335     0.8438   0.005931         38        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.967       0.98      0.991      0.936

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     51/100      4.79G     0.4249     0.8124   0.005833         47        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.978      0.981      0.992      0.931

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     52/100      4.79G     0.4105     0.7691   0.005672         47        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.978      0.967      0.991      0.929

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     53/100      4.79G     0.4111     0.7738   0.005606         69        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.972      0.971      0.991      0.932

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     54/100       4.8G     0.4101     0.7818   0.005662         42        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.972      0.977      0.989      0.934

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     55/100      4.79G     0.4192     0.7693   0.005716         38        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.966      0.964      0.989      0.933

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     56/100      4.81G     0.4091      0.757    0.00559         46        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.966      0.974      0.991      0.934

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     57/100      4.79G     0.4211     0.7486   0.005703         62        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.968      0.973      0.991      0.939

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     58/100       4.8G     0.4079     0.7794    0.00563        108        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.975      0.969      0.991      0.934

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     59/100      4.79G     0.4042     0.7287   0.005477         73        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.983      0.973      0.992      0.947

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     60/100      4.79G     0.3974     0.7095   0.005401         29        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.969      0.981      0.992      0.945

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     61/100      4.79G      0.389     0.7063   0.005234         47        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.969      0.979      0.991      0.945

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     62/100      4.79G     0.3695     0.6953   0.005081         37        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.976      0.978      0.993      0.946

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     63/100       4.8G     0.3817     0.7114   0.005122         44        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.969      0.977       0.99      0.943

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     64/100      4.79G      0.409     0.7142   0.005475         59        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002       0.96      0.977      0.991      0.936

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     65/100       4.8G     0.3716     0.6758   0.005096         73        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.973      0.981      0.992      0.949

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     66/100      4.79G     0.3697     0.6503   0.005055         31        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.964      0.973      0.991      0.944

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     67/100      4.79G     0.3659     0.6485   0.004881         42        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002       0.98      0.982      0.992       0.95

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     68/100      4.79G     0.3543      0.627   0.004822         70        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.979      0.983      0.992      0.952

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     69/100       4.8G     0.3607     0.6124   0.004862         74        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.978      0.981      0.992      0.948

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     70/100      4.79G     0.3489     0.6096   0.004593         64        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.981      0.976      0.992       0.95

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     71/100      4.79G     0.3445     0.6087   0.004543         80        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.983      0.978      0.992      0.947

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     72/100       4.8G     0.3343     0.5905   0.004509         71        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.981      0.981      0.991      0.947

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     73/100       4.8G     0.3391     0.5947   0.004582         73        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.982      0.965      0.992      0.951
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     74/100       4.8G     0.3302     0.5737   0.004412         48        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.977      0.975      0.992      0.948

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     75/100       4.8G       0.33     0.5808   0.004347         55        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.984       0.98      0.993      0.952

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     76/100       4.8G     0.3204     0.5491   0.004311         45        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.986      0.989      0.993      0.952
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     77/100      4.82G     0.3165     0.5374   0.004214         46        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.982      0.983      0.993      0.946

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     78/100       4.8G     0.3199     0.5386   0.004211         47        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002       0.98      0.985      0.992      0.949

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     79/100      4.79G     0.3163     0.5467   0.004177         57        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.984      0.985      0.993      0.951

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     80/100      4.79G     0.3156     0.5305   0.004253         63        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.982      0.985      0.994      0.948

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     81/100      4.79G     0.3058     0.5128   0.004057         55        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.982      0.981      0.992      0.949

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     82/100      4.79G      0.315     0.5168   0.004266         37        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.978      0.981      0.993      0.954

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     83/100      4.79G     0.3057     0.5045    0.00409         29        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.981      0.977      0.992      0.957

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     84/100      4.79G     0.2952     0.4919   0.003977         60        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.979      0.986      0.993      0.954

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     85/100      4.79G     0.2941     0.5151   0.003905         73        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.982      0.974      0.992      0.952

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     86/100       4.8G     0.2869     0.4817   0.003904         34        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.974      0.987      0.993      0.954

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     87/100       4.8G     0.2885     0.4987     0.0039         31        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.982      0.983      0.993      0.956

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     88/100       4.8G     0.2865      0.499   0.003865         47        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.982      0.987      0.993      0.956

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     89/100      4.79G      0.278     0.4772   0.003813         68        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.983      0.979      0.993      0.954

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     90/100      4.79G      0.282     0.4916   0.003892         39        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.983      0.983      0.992      0.957
Closing dataloader mosaic

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     91/100      4.79G     0.2773     0.4633   0.003681         25        640: 100% ━━━━━━━━━━━━ 157/157 4.5it/s 34.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.979      0.989      0.993      0.957

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     92/100       4.8G     0.2744      0.473   0.003597         31        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.985      0.985      0.993      0.957

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     93/100      4.79G     0.2583     0.4493    0.00348         46        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.981      0.986      0.993      0.956

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     94/100      4.79G     0.2654      0.455   0.003571         47        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.983      0.983      0.993      0.958

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     95/100      4.79G     0.2592     0.4457   0.003501         54        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.984      0.986      0.993      0.959

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     96/100      4.79G     0.2629     0.4522   0.003533         50        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.988      0.982      0.993      0.959

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     97/100      4.81G     0.2569     0.4323   0.003509         57        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.985       0.98      0.992      0.958

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     98/100       4.8G     0.2521     0.4387   0.003372         52        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.987      0.985      0.993      0.958

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     99/100      4.79G     0.2495     0.4342    0.00336         36        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.984      0.987      0.992      0.958
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    100/100       4.8G       0.25     0.4219   0.003341         57        640: 100% ━━━━━━━━━━━━ 157/157 4.6it/s 34.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 6.0it/s 2.8s
                   all       1078       2002      0.985      0.986      0.992      0.959

100 epochs completed in 1.045 hours.
Optimizer stripped from /home/nidszxh/Projects/Neuro-CXG/results/experiments/detection/ROI_Detection_v29/weights/last.pt, 5.4MB
Optimizer stripped from /home/nidszxh/Projects/Neuro-CXG/results/experiments/detection/ROI_Detection_v29/weights/best.pt, 5.4MB

Validating /home/nidszxh/Projects/Neuro-CXG/results/experiments/detection/ROI_Detection_v29/weights/best.pt...
Ultralytics 8.4.5 🚀 Python-3.12.12 torch-2.9.0+cu128 CUDA:0 (NVIDIA GeForce RTX 4060 Laptop GPU, 7806MiB)
YOLO26n summary (fused): 122 layers, 2,377,176 parameters, 0 gradients, 5.2 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 5.3it/s 3.2s
                   all       1078       2002      0.985      0.984      0.993      0.959
      Frontal_Superior        154        154      0.984      0.994      0.995      0.993
        Motor_Premotor        154        154      0.991          1      0.994      0.949
             Cingulate        154        154       0.99      0.994      0.995      0.991
                Limbic        308        308      0.984      0.974      0.987      0.904
             Occipital        308        308      0.975      0.994      0.995      0.947
              Parietal        154        154       0.98      0.939      0.992      0.922
              Temporal        308        308      0.978      0.984      0.992      0.961
           Subcortical        154        154      0.986      0.994      0.995       0.99
            Cerebellum        308        308      0.993      0.988      0.993      0.971
Speed: 0.2ms preprocess, 1.7ms inference, 0.0ms loss, 0.2ms postprocess per image
Results saved to /home/nidszxh/Projects/Neuro-CXG/results/experiments/detection/ROI_Detection_v29
MLflow: results logged to runs/mlflow
MLflow: disable with 'yolo settings mlflow=False'
Ultralytics 8.4.5 🚀 Python-3.12.12 torch-2.9.0+cu128 CUDA:0 (NVIDIA GeForce RTX 4060 Laptop GPU, 7806MiB)
YOLO26n summary (fused): 122 layers, 2,377,176 parameters, 0 gradients, 5.2 GFLOPs
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2974.6±1362.8 MB/s, size: 49.2 KB)
val: Scanning /home/nidszxh/Projects/Neuro-CXG/data/final/val/labels.cache... 308 images, 770 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 1078/1078 645.9Mit/s 0.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 68/68 18.9it/s 3.6s
                   all       1078       2002      0.984      0.986      0.993      0.959
      Frontal_Superior        154        154      0.983      0.994      0.995      0.993
        Motor_Premotor        154        154       0.99          1      0.994      0.949
             Cingulate        154        154       0.99      0.994      0.995      0.991
                Limbic        308        308      0.984       0.98      0.987      0.904
             Occipital        308        308       0.97      0.994      0.995      0.947
              Parietal        154        154       0.98      0.943      0.992      0.922
              Temporal        308        308      0.977      0.987      0.992       0.96
           Subcortical        154        154      0.985      0.994      0.995       0.99
            Cerebellum        308        308      0.993      0.989      0.993      0.971
Speed: 0.5ms preprocess, 2.3ms inference, 0.0ms loss, 0.0ms postprocess per image
Results saved to /home/nidszxh/Projects/Neuro-CXG/runs/detect/val

[SUCCESS] Training Complete. Model aligned for consistent lobe detection.
INFO:pipeline:Completed: YOLO Training (ROI Detection)
INFO:pipeline:Running: Spatial Feature Extraction (12-region)





INFO:__main__:Total detections: 13475
INFO:__main__:Unique subjects: 1025
INFO:__main__:Unique ROI classes detected: [0, 2, 4, 5, 6, 7, 8, 9, 10]
INFO:__main__:Aggregating 2D detections into 3D lobe nodes...
Building Subject Nodes: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1025/1025 [00:02<00:00, 408.37it/s]
INFO:__main__:Subjects processed: 1025
INFO:__main__:Subjects filtered (< 9 regions): 25
INFO:__main__:Subjects kept (9+ regions): 1000
INFO:__main__:Note: Accepting subjects with 9+ of 12 regions (missing regions filled with 0s)
INFO:__main__:Success: 1000 subjects with 12-node architecture
INFO:__main__:6 spatial features per lobe
INFO:__main__:Saved to /home/nidszxh/Projects/Neuro-CXG/data/metadata/node_features_3d.csv
INFO:pipeline:Completed: Spatial Feature Extraction (12-region)
INFO:pipeline:Running: Temporal Feature Extraction
INFO:__main__:Extracting temporal features for 1025 subjects...
INFO:__main__:Features per ROI: 20 (with frequency features)
Subjects: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1025/1025 [02:25<00:00,  7.07it/s]
INFO:__main__:Saved temporal features to /home/nidszxh/Projects/Neuro-CXG/data/metadata/node_attributes_temporal.csv
INFO:pipeline:Completed: Temporal Feature Extraction
INFO:pipeline:Running: Feature Harmonization
INFO:__main__:================================================================================
INFO:__main__:FOLD-SAFE HARMONIZATION
INFO:__main__:================================================================================
INFO:__main__:Feature validation: 1025 subjects × 3400 features | NaN: 0 (0.0%) | Inf: 0 | Subjects with NaN: 0 | Features with NaN: 0
INFO:__main__:Log-transformed 850 spectral-power columns
INFO:__main__:Repair complete: 1025 subjects remaining (removed 0)
INFO:__main__:================================================================================
INFO:__main__:PROCESSING FOLD 1 / 5
INFO:__main__:================================================================================
WARNING:__main__:Dropping 140 constant features before harmonization
INFO:__main__:  Aggregating ROIs to lobes...
INFO:__main__:Aggregating 170 ROIs → 12 regions…
INFO:__main__:Aggregated → 820 subjects × 240 lobe-features
INFO:__main__:Aggregating 170 ROIs → 12 regions…
INFO:__main__:Aggregated → 205 subjects × 240 lobe-features
INFO:__main__:  Saved: data/metadata/harmonized_folds_cv/fold0_train_lobes.csv
INFO:__main__:  Saved: data/metadata/harmonized_folds_cv/fold0_val_lobes.csv
INFO:__main__:  Fold 1 complete
INFO:__main__:================================================================================
INFO:__main__:PROCESSING FOLD 2 / 5
INFO:__main__:================================================================================
WARNING:__main__:Dropping 140 constant features before harmonization
INFO:__main__:  Aggregating ROIs to lobes...
INFO:__main__:Aggregating 170 ROIs → 12 regions…
INFO:__main__:Aggregated → 820 subjects × 240 lobe-features
INFO:__main__:Aggregating 170 ROIs → 12 regions…
INFO:__main__:Aggregated → 205 subjects × 240 lobe-features
INFO:__main__:  Saved: data/metadata/harmonized_folds_cv/fold1_train_lobes.csv
INFO:__main__:  Saved: data/metadata/harmonized_folds_cv/fold1_val_lobes.csv
INFO:__main__:  Fold 2 complete
INFO:__main__:================================================================================
INFO:__main__:PROCESSING FOLD 3 / 5
INFO:__main__:================================================================================
WARNING:__main__:Dropping 140 constant features before harmonization
INFO:__main__:  Aggregating ROIs to lobes...
INFO:__main__:Aggregating 170 ROIs → 12 regions…
INFO:__main__:Aggregated → 820 subjects × 240 lobe-features
INFO:__main__:Aggregating 170 ROIs → 12 regions…
INFO:__main__:Aggregated → 205 subjects × 240 lobe-features
INFO:__main__:  Saved: data/metadata/harmonized_folds_cv/fold2_train_lobes.csv
INFO:__main__:  Saved: data/metadata/harmonized_folds_cv/fold2_val_lobes.csv
INFO:__main__:  Fold 3 complete
INFO:__main__:================================================================================
INFO:__main__:PROCESSING FOLD 4 / 5
INFO:__main__:================================================================================
WARNING:__main__:Dropping 140 constant features before harmonization
INFO:__main__:  Aggregating ROIs to lobes...
INFO:__main__:Aggregating 170 ROIs → 12 regions…
INFO:__main__:Aggregated → 820 subjects × 240 lobe-features
INFO:__main__:Aggregating 170 ROIs → 12 regions…
INFO:__main__:Aggregated → 205 subjects × 240 lobe-features
INFO:__main__:  Saved: data/metadata/harmonized_folds_cv/fold3_train_lobes.csv
INFO:__main__:  Saved: data/metadata/harmonized_folds_cv/fold3_val_lobes.csv
INFO:__main__:  Fold 4 complete
INFO:__main__:================================================================================
INFO:__main__:PROCESSING FOLD 5 / 5
INFO:__main__:================================================================================
WARNING:__main__:Dropping 140 constant features before harmonization
INFO:__main__:  Aggregating ROIs to lobes...
INFO:__main__:Aggregating 170 ROIs → 12 regions…
INFO:__main__:Aggregated → 820 subjects × 240 lobe-features
INFO:__main__:Aggregating 170 ROIs → 12 regions…
INFO:__main__:Aggregated → 205 subjects × 240 lobe-features
INFO:__main__:  Saved: data/metadata/harmonized_folds_cv/fold4_train_lobes.csv
INFO:__main__:  Saved: data/metadata/harmonized_folds_cv/fold4_val_lobes.csv
INFO:__main__:  Fold 5 complete
INFO:__main__:================================================================================
INFO:__main__:ALL FOLDS HARMONIZED
INFO:__main__:================================================================================
INFO:__main__:Saved combined harmonized lobe-level features → /home/nidszxh/Projects/Neuro-CXG/data/metadata/node_attributes_harmonized.csv (1025 subjects, 241 columns)
INFO:__main__:============================================================
INFO:__main__:HARMONIZATION QUALITY CHECK
INFO:__main__:============================================================
INFO:__main__:No overlapping columns — aggregating originals to lobes for comparison
INFO:__main__:Aggregating 170 ROIs → 12 regions…
INFO:__main__:Aggregated → 1025 subjects × 240 lobe-features
INFO:__main__:  Original variance  : 91748440078.6859
INFO:__main__:  Harmonized variance: 14531368986.1707
INFO:__main__:  Variance retention : 15.84%
INFO:__main__:  Per-feature: 15.0% within, 85.0% low, 0.0% high
WARNING:__main__:  Many features lost >30%% variance after harmonization
INFO:__main__:  No NaN values introduced
INFO:pipeline:Completed: Feature Harmonization
INFO:pipeline:Running: Pre-GNN Integrity Check
INFO:src.validation.pipeline_checks:Starting pre-GNN integrity check...
INFO:src.validation.pipeline_checks:Dataset Completeness Report (Target: 7 slices/subject)
INFO:src.validation.pipeline_checks:
Split: TRAIN
INFO:src.validation.pipeline_checks:  Total Subjects: 717
INFO:src.validation.pipeline_checks:  ✓ 7 slices: 717 subjects
INFO:src.validation.pipeline_checks:  ✓ Image/Label count matches (5019 files)
INFO:src.validation.pipeline_checks:
Split: VAL
INFO:src.validation.pipeline_checks:  Total Subjects: 154
INFO:src.validation.pipeline_checks:  ✓ 7 slices: 154 subjects
INFO:src.validation.pipeline_checks:  ✓ Image/Label count matches (1078 files)
INFO:src.validation.pipeline_checks:
Split: TEST
INFO:src.validation.pipeline_checks:  Total Subjects: 154
INFO:src.validation.pipeline_checks:  ✓ 7 slices: 154 subjects
INFO:src.validation.pipeline_checks:  ✓ Image/Label count matches (1078 files)
INFO:src.validation.pipeline_checks:
Pre-GNN integrity check complete.
INFO:pipeline:Completed: Pre-GNN Integrity Check
INFO:pipeline:Running: Causal Graph Construction (12×12)
INFO:__main__:============================================================
INFO:__main__:CONSTRUCTING 12×12 CAUSAL GRAPHS (Lag=1)
INFO:__main__:Sparsity: Keep top 30% of edges
INFO:__main__:============================================================
Building Graphs:   0%|                                                                                                                                                   | 0/1025 [00:00<?, ?it/s]/home/nidszxh/Projects/Neuro-CXG/src/features/construct_causal.py:90: UserWarning: torch.linalg.svd: During SVD computation with the selected cusolver driver, batches 0 failed to converge. A more accurate method will be used to compute the SVD as a fallback. Check doc at https://pytorch.org/docs/stable/generated/torch.linalg.svd.html (Triggered internally at /pytorch/aten/src/ATen/native/cuda/linalg/BatchLinearAlgebraLib.cpp:702.)
  u, s, vh = torch.linalg.svd(centered, full_matrices=False)
Building Graphs:  37%|██████████████████████████████████████████████████▍                                                                                      | 377/1025 [01:20<02:17,  4.72it/s]WARNING:__main__:SDSU_0050209: Causal matrix is all zeros - skipping
Building Graphs:  63%|██████████████████████████████████████████████████████████████████████████████████████▌                                                  | 648/1025 [02:18<01:15,  4.98it/s]WARNING:__main__:Caltech_0051464: Causal matrix is all zeros - skipping
Building Graphs:  78%|███████████████████████████████████████████████████████████████████████████████████████████████████████████                              | 801/1025 [02:51<00:46,  4.86it/s]WARNING:__main__:SDSU_0050195: Causal matrix is all zeros - skipping
Building Graphs: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1025/1025 [03:39<00:00,  4.67it/s]
INFO:__main__:
============================================================
INFO:__main__:GRAPH CONSTRUCTION SUMMARY
INFO:__main__:============================================================
INFO:__main__:Total subjects: 1025
INFO:__main__:✓ Successfully constructed: 1022
INFO:__main__:✗ Failed: 3
INFO:__main__:  ↳ Zero edges: 3
INFO:__main__:  ↳ Missing time series: 0
INFO:__main__:
Success rate: 99.7%
INFO:__main__:Output directory: /home/nidszxh/Projects/Neuro-CXG/data/processed/causal_graphs
INFO:__main__:============================================================
INFO:pipeline:Completed: Causal Graph Construction (12×12)
INFO:pipeline:Running: Pipeline Diagnostics
INFO:src.validation.pipeline_checks:Loaded metadata: 1112 records
INFO:src.validation.pipeline_checks:Found 7175 PNG slices from 1025 subjects
INFO:src.validation.pipeline_checks:Matched 1025 subjects to metadata
INFO:src.validation.pipeline_checks:
========================================
INFO:src.validation.pipeline_checks:         DATASET HEALTH REPORT          
INFO:src.validation.pipeline_checks:========================================
INFO:src.validation.pipeline_checks:Unique Subjects:   1025
INFO:src.validation.pipeline_checks:Total PNG Slices:  7175
INFO:src.validation.pipeline_checks:Avg Slices/Sub:    7.0 (Target: 7)
INFO:src.validation.pipeline_checks:----------------------------------------
INFO:src.validation.pipeline_checks:CLASS BALANCE
INFO:src.validation.pipeline_checks:  Autism (ASD):     498
INFO:src.validation.pipeline_checks:  Controls (TC):    527
INFO:src.validation.pipeline_checks:  Ratio (ASD/TC):   0.94
INFO:src.validation.pipeline_checks:----------------------------------------
INFO:src.validation.pipeline_checks:DEMOGRAPHICS
INFO:src.validation.pipeline_checks:  Avg Age:          16.9 years
INFO:src.validation.pipeline_checks:  Sex Ratio (M/F):  869/156
INFO:src.validation.pipeline_checks:----------------------------------------
INFO:src.validation.pipeline_checks:TOP SITES
INFO:src.validation.pipeline_checks:  NYU            : 175 subjects
INFO:src.validation.pipeline_checks:  UM_1           : 106 subjects
INFO:src.validation.pipeline_checks:  UCLA_1         : 72 subjects
INFO:src.validation.pipeline_checks:  USM            : 71 subjects
INFO:src.validation.pipeline_checks:  YALE           : 56 subjects
INFO:src.validation.pipeline_checks:----------------------------------------
INFO:src.validation.pipeline_checks:DATA COMPLETENESS
INFO:src.validation.pipeline_checks:  All downloaded subjects have metadata
WARNING:src.validation.pipeline_checks:  WARNING: Subjects with metadata but no images: 11
INFO:src.validation.pipeline_checks:----------------------------------------
INFO:src.validation.pipeline_checks:SLICE DISTRIBUTION
INFO:src.validation.pipeline_checks:  Complete (7 slices): 1025/1025
INFO:src.validation.pipeline_checks:  All subjects have complete slice sets (7/7)
INFO:src.validation.pipeline_checks:----------------------------------------
INFO:src.validation.pipeline_checks:TIME SERIES FILES
INFO:src.validation.pipeline_checks:  Time series files:  1025
INFO:src.validation.pipeline_checks:  All downloaded subjects have time series
INFO:src.validation.pipeline_checks:----------------------------------------
INFO:src.validation.pipeline_checks:FEATURE EXTRACTION STATUS
INFO:src.validation.pipeline_checks:  Spatial Features         : 1000 subjects
INFO:src.validation.pipeline_checks:  Temporal Features        : 1025 subjects
INFO:src.validation.pipeline_checks:  Harmonized Features      : 1025 subjects
INFO:src.validation.pipeline_checks:----------------------------------------
INFO:src.validation.pipeline_checks:GRAPH CONSTRUCTION STATUS
INFO:src.validation.pipeline_checks:  Graph files:        1022
INFO:src.validation.pipeline_checks:  Status:             Graphs constructed
INFO:src.validation.pipeline_checks:
========================================
INFO:src.validation.pipeline_checks:Health report complete.
INFO:pipeline:Completed: Pipeline Diagnostics
INFO:pipeline:Running: Quality Validation (YOLO & Graph Sparsity)
INFO:src.validation.pipeline_checks:Starting comprehensive pipeline validation...
INFO:src.validation.pipeline_checks:======================================================================
INFO:src.validation.pipeline_checks:STAGE 1: ENVIRONMENT VALIDATION
INFO:src.validation.pipeline_checks:======================================================================
INFO:src.validation.pipeline_checks:[Environment] Python 3.12
INFO:src.validation.pipeline_checks:[Environment] CUDA available: NVIDIA GeForce RTX 4060 Laptop GPU
INFO:src.validation.pipeline_checks:[Environment] Feature dims: GNN_IN_CHANNELS=28 (20 temporal + 6 spatial + internal)
INFO:src.validation.pipeline_checks:Checking atlas...
INFO:src.validation.pipeline_checks:[Atlas] Atlas loaded: 166 ROIs
INFO:src.validation.pipeline_checks:Checking LOBE_MAPPING...
INFO:src.validation.pipeline_checks:[Config] LOBE_MAPPING valid: 12 lobes, 170 ROIs
INFO:src.validation.pipeline_checks:
======================================================================
INFO:src.validation.pipeline_checks:STAGE 2: DATA VALIDATION
INFO:src.validation.pipeline_checks:======================================================================
INFO:src.validation.pipeline_checks:[Data] 1025 subjects with complete data
INFO:src.validation.pipeline_checks:Checking manifest...
INFO:src.validation.pipeline_checks:[Manifest] 1025 subjects across 3 splits
INFO:src.validation.pipeline_checks:
======================================================================
INFO:src.validation.pipeline_checks:STAGE 3: FEATURE VALIDATION
INFO:src.validation.pipeline_checks:======================================================================
INFO:src.validation.pipeline_checks:[Features] Temporal features: 1025 subjects, 0.00% NaN
WARNING:src.validation.pipeline_checks:[Features] 12064 extreme values detected (|x| > 1e6)
INFO:src.validation.pipeline_checks:[Features] Spatial features: 1000/1000 complete (100.0%)
INFO:src.validation.pipeline_checks:[Features] Harmonized features: 1025 subjects, clean
INFO:src.validation.pipeline_checks:
======================================================================
INFO:src.validation.pipeline_checks:STAGE 4: GRAPH VALIDATION
INFO:src.validation.pipeline_checks:======================================================================
INFO:src.validation.pipeline_checks:[Graphs] 1022 graphs — mean edges: 33.0, median: 34
INFO:src.validation.pipeline_checks:Checking stratification...
INFO:src.validation.pipeline_checks:[Stratification] No data leakage across 3 splits
INFO:src.validation.pipeline_checks:
======================================================================
INFO:src.validation.pipeline_checks:STAGE 5: MODEL VALIDATION
INFO:src.validation.pipeline_checks:======================================================================
INFO:src.validation.pipeline_checks:[Models] 5 trained models, mean AUC: 0.6194, F1: 0.7132
INFO:src.validation.pipeline_checks:
======================================================================
INFO:src.validation.pipeline_checks:VALIDATION REPORT
INFO:src.validation.pipeline_checks:======================================================================
WARNING:src.validation.pipeline_checks:
WARNINGS (1):
WARNING:src.validation.pipeline_checks:----------------------------------------------------------------------
WARNING:src.validation.pipeline_checks:  [Features] 12064 extreme values detected (|x| > 1e6)
WARNING:src.validation.pipeline_checks:    -> Suggestion: Check feature extraction for numerical issues
INFO:src.validation.pipeline_checks:
PASSED CHECKS (13):
INFO:src.validation.pipeline_checks:----------------------------------------------------------------------
INFO:src.validation.pipeline_checks:  Python 3.12
INFO:src.validation.pipeline_checks:  CUDA available: NVIDIA GeForce RTX 4060 Laptop GPU
INFO:src.validation.pipeline_checks:  Feature dims: GNN_IN_CHANNELS=28 (20 temporal + 6 spatial + internal)
INFO:src.validation.pipeline_checks:  Atlas loaded: 166 ROIs
INFO:src.validation.pipeline_checks:  LOBE_MAPPING valid: 12 lobes, 170 ROIs
INFO:src.validation.pipeline_checks:  1025 subjects with complete data
INFO:src.validation.pipeline_checks:  1025 subjects across 3 splits
INFO:src.validation.pipeline_checks:  Temporal features: 1025 subjects, 0.00% NaN
INFO:src.validation.pipeline_checks:  Spatial features: 1000/1000 complete (100.0%)
INFO:src.validation.pipeline_checks:  Harmonized features: 1025 subjects, clean
INFO:src.validation.pipeline_checks:  1022 graphs — mean edges: 33.0, median: 34
INFO:src.validation.pipeline_checks:  No data leakage across 3 splits
INFO:src.validation.pipeline_checks:  5 trained models, mean AUC: 0.6194, F1: 0.7132
INFO:src.validation.pipeline_checks:
======================================================================
INFO:src.validation.pipeline_checks:SUMMARY:
INFO:src.validation.pipeline_checks:  Passed: 13
INFO:src.validation.pipeline_checks:  Warnings: 1
INFO:src.validation.pipeline_checks:  Critical: 0
INFO:src.validation.pipeline_checks:======================================================================
INFO:src.validation.pipeline_checks:
Pipeline functional with warnings
INFO:pipeline:Completed: Quality Validation (YOLO & Graph Sparsity)
INFO:pipeline:Running: GNN Training (5-Fold CV)
INFO:src.features.graph_factory:✓ Feature dimensions validated
INFO:src.features.graph_factory:Initialized train dataset with 702 subjects
INFO:src.features.graph_factory:  Node features: 28 (20 temporal+internal + 6 spatial)
INFO:__main__:Class distribution: Control=340, ASD=362
INFO:__main__:Class weights: Control=1.032, ASD=0.970
INFO:src.analysis.diagnostics:TrainingMonitor initialised — output: /home/nidszxh/Projects/Neuro-CXG/results/experiments/training, folds: 5
INFO:__main__:
======================================================================
INFO:__main__:GNN TRAINING - 5-FOLD CROSS-VALIDATION
INFO:__main__:======================================================================
INFO:__main__:Total subjects: 702
INFO:__main__:OneCycle max LR: 0.003
INFO:__main__:Hidden channels: 128
INFO:__main__:Input features: 28 (20 temporal + 6 spatial)
INFO:__main__:Site conditioning: True
INFO:__main__:Demographics: True
INFO:__main__:Early stopping patience: 20
INFO:__main__:Focal Loss: α=0.62, γ=2.0
INFO:__main__:======================================================================

INFO:__main__:
======================================================================
INFO:__main__:FOLD 1/5
INFO:__main__:======================================================================
INFO:__main__:Train: Control=272, ASD=289
INFO:__main__:Val: Control=68, ASD=73
INFO:src.models.training_utils:Early stopping triggered after 20 epochs without improvement
INFO:src.models.training_utils:Fold 0: early stopping at epoch 43
INFO:__main__:Epoch 010 | LR: 0.000885 | Loss: 0.6731 | AUC: 0.6485 | AUPRC: 0.6283 | F1@0.56: 0.7200
INFO:__main__:Epoch 020 | LR: 0.002368 | Loss: 0.6588 | AUC: 0.6507 | AUPRC: 0.6439 | F1@0.51: 0.7232
INFO:__main__:Epoch 030 | LR: 0.002998 | Loss: 0.6504 | AUC: 0.6332 | AUPRC: 0.6330 | F1@0.57: 0.7200
INFO:__main__:Epoch 040 | LR: 0.002821 | Loss: 0.6489 | AUC: 0.6324 | AUPRC: 0.6325 | F1@0.53: 0.7196
INFO:src.models.training_utils:✓ Saved checkpoint: best_model_fold0.pt (epoch 23, auc=0.6622)
INFO:__main__:✓ Best fold 0: AUC=0.6622, AUPRC=0.6628, F1=0.7200
INFO:__main__:
Fold 1 Final Results:
INFO:__main__:  Best epoch: 23
INFO:__main__:  Training time: 7.6s
INFO:__main__:  AUC: 0.6622
INFO:__main__:  F1: 0.7200 (threshold=0.532)
INFO:__main__:  Accuracy: 0.6525
INFO:__main__:  Confusion Matrix:
INFO:__main__:    [[29 39]
 [10 63]]
INFO:__main__:
Generating fold visualizations...
INFO:src.analysis.diagnostics:Training curves saved → /home/nidszxh/Projects/Neuro-CXG/results/experiments/training/fold_plots/training_curves_fold_0.png
INFO:__main__:  Training curves saved to: /home/nidszxh/Projects/Neuro-CXG/results/experiments/training/fold_plots/training_curves_fold_0.png
INFO:src.analysis.diagnostics:Training history saved → /home/nidszxh/Projects/Neuro-CXG/results/experiments/training/training_history_fold0.json
INFO:__main__:  Training history saved to: /home/nidszxh/Projects/Neuro-CXG/results/experiments/training/training_history_fold0.json
INFO:__main__:
======================================================================
INFO:__main__:FOLD 2/5
INFO:__main__:======================================================================
INFO:__main__:Train: Control=272, ASD=289
INFO:__main__:Val: Control=68, ASD=73
INFO:src.models.training_utils:Early stopping triggered after 20 epochs without improvement
INFO:src.models.training_utils:Fold 1: early stopping at epoch 59
INFO:__main__:Epoch 010 | LR: 0.000885 | Loss: 0.6820 | AUC: 0.6279 | AUPRC: 0.6166 | F1@0.56: 0.7302
INFO:__main__:Epoch 020 | LR: 0.002368 | Loss: 0.6546 | AUC: 0.6207 | AUPRC: 0.6152 | F1@0.51: 0.7128
INFO:__main__:Epoch 030 | LR: 0.002998 | Loss: 0.6347 | AUC: 0.6273 | AUPRC: 0.6121 | F1@0.54: 0.7254
INFO:__main__:Epoch 040 | LR: 0.002821 | Loss: 0.6424 | AUC: 0.6416 | AUPRC: 0.6107 | F1@0.55: 0.7396
INFO:__main__:Epoch 050 | LR: 0.002382 | Loss: 0.6396 | AUC: 0.6372 | AUPRC: 0.6085 | F1@0.54: 0.7302
INFO:src.models.training_utils:✓ Saved checkpoint: best_model_fold1.pt (epoch 39, auc=0.6442)
INFO:__main__:✓ Best fold 1: AUC=0.6442, AUPRC=0.6168, F1=0.7396
INFO:__main__:
Fold 2 Final Results:
INFO:__main__:  Best epoch: 39
INFO:__main__:  Training time: 9.4s
INFO:__main__:  AUC: 0.6442
INFO:__main__:  F1: 0.7396 (threshold=0.537)
INFO:__main__:  Accuracy: 0.6454
INFO:__main__:  Confusion Matrix:
INFO:__main__:    [[20 48]
 [ 2 71]]
INFO:__main__:
Generating fold visualizations...
INFO:src.analysis.diagnostics:Training curves saved → /home/nidszxh/Projects/Neuro-CXG/results/experiments/training/fold_plots/training_curves_fold_1.png
INFO:__main__:  Training curves saved to: /home/nidszxh/Projects/Neuro-CXG/results/experiments/training/fold_plots/training_curves_fold_1.png
INFO:src.analysis.diagnostics:Training history saved → /home/nidszxh/Projects/Neuro-CXG/results/experiments/training/training_history_fold1.json
INFO:__main__:  Training history saved to: /home/nidszxh/Projects/Neuro-CXG/results/experiments/training/training_history_fold1.json
INFO:__main__:
======================================================================
INFO:__main__:FOLD 3/5
INFO:__main__:======================================================================
INFO:__main__:Train: Control=272, ASD=290
INFO:__main__:Val: Control=68, ASD=72
INFO:src.models.training_utils:Early stopping triggered after 20 epochs without improvement
INFO:src.models.training_utils:Fold 2: early stopping at epoch 39
INFO:__main__:Epoch 010 | LR: 0.000885 | Loss: 0.6859 | AUC: 0.5944 | AUPRC: 0.5893 | F1@0.55: 0.6900
INFO:__main__:Epoch 020 | LR: 0.002368 | Loss: 0.6548 | AUC: 0.5792 | AUPRC: 0.5543 | F1@0.53: 0.6863
INFO:__main__:Epoch 030 | LR: 0.002998 | Loss: 0.6450 | AUC: 0.6101 | AUPRC: 0.5798 | F1@0.53: 0.7234
INFO:src.models.training_utils:✓ Saved checkpoint: best_model_fold2.pt (epoch 19, auc=0.6495)
INFO:__main__:✓ Best fold 2: AUC=0.6495, AUPRC=0.6366, F1=0.7150
INFO:__main__:
Fold 3 Final Results:
INFO:__main__:  Best epoch: 19
INFO:__main__:  Training time: 6.7s
INFO:__main__:  AUC: 0.6495
INFO:__main__:  F1: 0.7150 (threshold=0.501)
INFO:__main__:  Accuracy: 0.6071
INFO:__main__:  Confusion Matrix:
INFO:__main__:    [[16 52]
 [ 3 69]]
INFO:__main__:
Generating fold visualizations...
INFO:src.analysis.diagnostics:Training curves saved → /home/nidszxh/Projects/Neuro-CXG/results/experiments/training/fold_plots/training_curves_fold_2.png
INFO:__main__:  Training curves saved to: /home/nidszxh/Projects/Neuro-CXG/results/experiments/training/fold_plots/training_curves_fold_2.png
INFO:src.analysis.diagnostics:Training history saved → /home/nidszxh/Projects/Neuro-CXG/results/experiments/training/training_history_fold2.json
INFO:__main__:  Training history saved to: /home/nidszxh/Projects/Neuro-CXG/results/experiments/training/training_history_fold2.json
INFO:__main__:
======================================================================
INFO:__main__:FOLD 4/5
INFO:__main__:======================================================================
INFO:__main__:Train: Control=272, ASD=290
INFO:__main__:Val: Control=68, ASD=72
INFO:src.models.training_utils:Early stopping triggered after 20 epochs without improvement
INFO:src.models.training_utils:Fold 3: early stopping at epoch 47
INFO:__main__:Epoch 010 | LR: 0.000885 | Loss: 0.6904 | AUC: 0.5259 | AUPRC: 0.5226 | F1@0.54: 0.6730
INFO:__main__:Epoch 020 | LR: 0.002368 | Loss: 0.6565 | AUC: 0.5276 | AUPRC: 0.5281 | F1@0.49: 0.6730
INFO:__main__:Epoch 030 | LR: 0.002998 | Loss: 0.6416 | AUC: 0.5768 | AUPRC: 0.5757 | F1@0.50: 0.6731
INFO:__main__:Epoch 040 | LR: 0.002821 | Loss: 0.6422 | AUC: 0.5558 | AUPRC: 0.5506 | F1@0.57: 0.6734
INFO:src.models.training_utils:✓ Saved checkpoint: best_model_fold3.pt (epoch 27, auc=0.5862)
INFO:__main__:✓ Best fold 3: AUC=0.5862, AUPRC=0.5860, F1=0.6762
INFO:__main__:
Fold 4 Final Results:
INFO:__main__:  Best epoch: 27
INFO:__main__:  Training time: 7.8s
INFO:__main__:  AUC: 0.5862
INFO:__main__:  F1: 0.6762 (threshold=0.519)
INFO:__main__:  Accuracy: 0.5143
INFO:__main__:  Confusion Matrix:
INFO:__main__:    [[ 1 67]
 [ 1 71]]
INFO:__main__:
Generating fold visualizations...
INFO:src.analysis.diagnostics:Training curves saved → /home/nidszxh/Projects/Neuro-CXG/results/experiments/training/fold_plots/training_curves_fold_3.png
INFO:__main__:  Training curves saved to: /home/nidszxh/Projects/Neuro-CXG/results/experiments/training/fold_plots/training_curves_fold_3.png
INFO:src.analysis.diagnostics:Training history saved → /home/nidszxh/Projects/Neuro-CXG/results/experiments/training/training_history_fold3.json
INFO:__main__:  Training history saved to: /home/nidszxh/Projects/Neuro-CXG/results/experiments/training/training_history_fold3.json
INFO:__main__:
======================================================================
INFO:__main__:FOLD 5/5
INFO:__main__:======================================================================
INFO:__main__:Train: Control=272, ASD=290
INFO:__main__:Val: Control=68, ASD=72
INFO:src.models.training_utils:Early stopping triggered after 20 epochs without improvement
INFO:src.models.training_utils:Fold 4: early stopping at epoch 56
INFO:__main__:Epoch 010 | LR: 0.000885 | Loss: 0.6835 | AUC: 0.5427 | AUPRC: 0.5351 | F1@0.55: 0.7113
INFO:__main__:Epoch 020 | LR: 0.002368 | Loss: 0.6485 | AUC: 0.5566 | AUPRC: 0.5427 | F1@0.53: 0.7083
INFO:__main__:Epoch 030 | LR: 0.002998 | Loss: 0.6491 | AUC: 0.5605 | AUPRC: 0.5441 | F1@0.46: 0.7113
INFO:__main__:Epoch 040 | LR: 0.002821 | Loss: 0.6485 | AUC: 0.5558 | AUPRC: 0.5249 | F1@0.57: 0.7083
INFO:__main__:Epoch 050 | LR: 0.002382 | Loss: 0.6464 | AUC: 0.5549 | AUPRC: 0.5169 | F1@0.56: 0.7083
INFO:src.models.training_utils:✓ Saved checkpoint: best_model_fold4.pt (epoch 36, auc=0.5707)
INFO:__main__:✓ Best fold 4: AUC=0.5707, AUPRC=0.6067, F1=0.7150
INFO:__main__:
Fold 5 Final Results:
INFO:__main__:  Best epoch: 36
INFO:__main__:  Training time: 9.1s
INFO:__main__:  AUC: 0.5707
INFO:__main__:  F1: 0.7150 (threshold=0.531)
INFO:__main__:  Accuracy: 0.6071
INFO:__main__:  Confusion Matrix:
INFO:__main__:    [[16 52]
 [ 3 69]]
INFO:__main__:
Generating fold visualizations...
INFO:src.analysis.diagnostics:Training curves saved → /home/nidszxh/Projects/Neuro-CXG/results/experiments/training/fold_plots/training_curves_fold_4.png
INFO:__main__:  Training curves saved to: /home/nidszxh/Projects/Neuro-CXG/results/experiments/training/fold_plots/training_curves_fold_4.png
INFO:src.analysis.diagnostics:Training history saved → /home/nidszxh/Projects/Neuro-CXG/results/experiments/training/training_history_fold4.json
INFO:__main__:  Training history saved to: /home/nidszxh/Projects/Neuro-CXG/results/experiments/training/training_history_fold4.json
INFO:src.models.training_utils:
======================================================================
INFO:src.models.training_utils:FINAL CROSS-VALIDATION RESULTS
INFO:src.models.training_utils:======================================================================
INFO:src.models.training_utils:Mean AUC: 0.6226 ± 0.0368
INFO:src.models.training_utils:Mean F1: 0.7132 ± 0.0206
INFO:src.models.training_utils:Mean Accuracy: 0.6053 ± 0.0492
INFO:src.models.training_utils:Mean Threshold: 0.524
INFO:src.models.training_utils:Mean Best Epoch: 28.8
INFO:src.models.training_utils:
Per-fold AUCs: ['0.6622', '0.6442', '0.6495', '0.5862', '0.5707']
INFO:src.models.training_utils:Per-fold F1s: ['0.7200', '0.7396', '0.7150', '0.6762', '0.7150']
INFO:src.models.training_utils:Per-fold Best Epochs: [23, 39, 19, 27, 36]
INFO:src.models.training_utils:======================================================================

INFO:__main__:
======================================================================
INFO:__main__:ENSEMBLE EVALUATION (TEST SET)
INFO:__main__:======================================================================
INFO:src.features.graph_factory:✓ Feature dimensions validated
INFO:src.features.graph_factory:Initialized test dataset with 146 subjects
INFO:src.features.graph_factory:  Node features: 28 (20 temporal+internal + 6 spatial)
INFO:src.models.training_utils:Loaded checkpoint: best_model_fold0.pt (epoch 23)
INFO:src.models.training_utils:Loaded checkpoint: best_model_fold1.pt (epoch 39)
INFO:src.models.training_utils:Loaded checkpoint: best_model_fold2.pt (epoch 19)
INFO:src.models.training_utils:Loaded checkpoint: best_model_fold3.pt (epoch 27)
INFO:src.models.training_utils:Loaded checkpoint: best_model_fold4.pt (epoch 36)
INFO:__main__:Using AUC-weighted ensemble: [0.21272557 0.20696573 0.20865917 0.18831818 0.18333136]
INFO:__main__:
Ensemble Results (Test Set):
INFO:__main__:  AUC: 0.6017
INFO:__main__:  F1: 0.7014 (threshold=0.522)
INFO:__main__:  Accuracy: 0.5685
INFO:__main__:  Confusion Matrix:
INFO:__main__:    [[ 9 61]
 [ 2 74]]
INFO:__main__:======================================================================

INFO:__main__:
======================================================================
INFO:__main__:POST-TRAINING ANALYSIS
INFO:__main__:======================================================================

INFO:__main__:Running feature attribution analysis...
INFO:src.features.graph_factory:✓ Feature dimensions validated
INFO:src.features.graph_factory:Initialized test dataset with 146 subjects
INFO:src.features.graph_factory:  Node features: 28 (20 temporal+internal + 6 spatial)
INFO:src.models.training_utils:Loaded checkpoint: best_model_fold0.pt (epoch 23)
INFO:src.analysis.feature_attribution:FeatureAttributionAnalyzer initialized
INFO:src.analysis.feature_attribution:  Device: cuda
INFO:src.analysis.feature_attribution:  Features: 28
INFO:src.analysis.feature_attribution:Computing feature attributions...
INFO:src.analysis.feature_attribution:  Method: Gradient-based Saliency
Computing attributions:   0%|                                                                                                                                               | 0/5 [00:00<?, ?it/s]
WARNING:__main__:Feature attribution analysis failed: mat1 and mat2 shapes cannot be multiplied (32x128 and 131x128)
INFO:__main__:
Running causal graph analysis...
INFO:src.analysis.diagnostics:CausalGraphAnalyzer initialised — dir: /home/nidszxh/Projects/Neuro-CXG/data/processed/causal_graphs, subjects: 1025
INFO:src.analysis.diagnostics:Computing properties for 1022 graphs…
Graph properties: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1022/1022 [00:00<00:00, 1454.50it/s]
INFO:src.analysis.diagnostics:Computed properties for 1022 graphs
INFO:src.analysis.diagnostics:Comparing ASD vs Control graph topology…
INFO:src.analysis.diagnostics:  ASD: 497, Control: 525

======================================================================
GRAPH TOPOLOGY COMPARISON (ASD vs CONTROL)
======================================================================

Num Edges:
  ASD:     31.6298 ± 8.5864
  Control: 32.1810 ± 8.4738
  Mann-Whitney U=126294.50, p=0.3766, d=-0.065

Density:
  ASD:     0.2396 ± 0.0650
  Control: 0.2438 ± 0.0642
  Mann-Whitney U=126294.50, p=0.3766, d=-0.065

Avg Clustering:
  ASD:     0.4859 ± 0.1096
  Control: 0.4909 ± 0.1018
  Mann-Whitney U=128947.00, p=0.7480, d=-0.047

Frontal Superior In Degree:
  ASD:     4.0342 ± 1.6617
  Control: 4.2248 ± 1.6021
  Mann-Whitney U=121695.50, p=0.0587, d=-0.117

Frontal Superior Out Degree:
  ASD:     4.0362 ± 1.6355
  Control: 4.1943 ± 1.5628
  Mann-Whitney U=124689.50, p=0.2127, d=-0.099

Frontal Orbital In Degree:
  ASD:     3.9738 ± 1.7629
  Control: 3.8933 ± 1.7468
  Mann-Whitney U=133884.00, p=0.4616, d=0.046

Frontal Orbital Out Degree:
  ASD:     3.9577 ± 1.7431
  Control: 3.9714 ± 1.6702
  Mann-Whitney U=129603.00, p=0.8532, d=-0.008

Motor Premotor In Degree:
  ASD:     3.9698 ± 1.8179
  Control: 3.9390 ± 1.7846
  Mann-Whitney U=131295.00, p=0.8580, d=0.017

Motor Premotor Out Degree:
  ASD:     4.0644 ± 1.7830
  Control: 3.9543 ± 1.7342
  Mann-Whitney U=135852.50, p=0.2456, d=0.063

Insula In Degree:
  ASD:     3.8913 ± 1.6604
  Control: 3.9867 ± 1.6998
  Mann-Whitney U=126845.50, p=0.4361, d=-0.057

Insula Out Degree:
  ASD:     4.0543 ± 1.6515
  Control: 4.0381 ± 1.6672
  Mann-Whitney U=131660.50, p=0.7964, d=0.010

Cingulate In Degree:
  ASD:     0.0000 ± 0.0000
  Control: 0.0000 ± 0.0000
  Mann-Whitney U=130462.50, p=1.0000, d=0.000

Cingulate Out Degree:
  ASD:     0.0000 ± 0.0000
  Control: 0.0000 ± 0.0000
  Mann-Whitney U=130462.50, p=1.0000, d=0.000

Limbic In Degree:
  ASD:     4.1509 ± 1.6146
  Control: 4.2171 ± 1.5646
  Mann-Whitney U=128036.50, p=0.6005, d=-0.042

Limbic Out Degree:
  ASD:     3.9396 ± 1.6458
  Control: 4.0495 ± 1.5740
  Mann-Whitney U=125936.00, p=0.3289, d=-0.068

Occipital In Degree:
  ASD:     4.0724 ± 1.6453
  Control: 4.1295 ± 1.6476
  Mann-Whitney U=128455.50, p=0.6656, d=-0.035

Occipital Out Degree:
  ASD:     4.0080 ± 1.6891
  Control: 4.0476 ± 1.6595
  Mann-Whitney U=128461.00, p=0.6666, d=-0.024

Parietal In Degree:
  ASD:     4.0584 ± 1.5650
  Control: 4.2286 ± 1.5147
  Mann-Whitney U=123068.00, p=0.1099, d=-0.111

Parietal Out Degree:
  ASD:     4.0845 ± 1.5188
  Control: 4.3105 ± 1.5955
  Mann-Whitney U=120188.00, p=0.0266, d=-0.145
  ASD has significantly lower parietal_out_degree (small effect)

Temporal In Degree:
  ASD:     0.0000 ± 0.0000
  Control: 0.0000 ± 0.0000
  Mann-Whitney U=130462.50, p=1.0000, d=0.000

Temporal Out Degree:
  ASD:     0.0000 ± 0.0000
  Control: 0.0000 ± 0.0000
  Mann-Whitney U=130462.50, p=1.0000, d=0.000

Subcortical In Degree:
  ASD:     0.0000 ± 0.0000
  Control: 0.0000 ± 0.0000
  Mann-Whitney U=130462.50, p=1.0000, d=0.000

Subcortical Out Degree:
  ASD:     0.0000 ± 0.0000
  Control: 0.0000 ± 0.0000
  Mann-Whitney U=130462.50, p=1.0000, d=0.000

Cerebellum In Degree:
  ASD:     3.4789 ± 2.0751
  Control: 3.5619 ± 2.0735
  Mann-Whitney U=127857.00, p=0.5767, d=-0.040

Cerebellum Out Degree:
  ASD:     3.4849 ± 2.0634
  Control: 3.6152 ± 1.9626
  Mann-Whitney U=126836.00, p=0.4363, d=-0.065

Brainstem In Degree:
  ASD:     0.0000 ± 0.0000
  Control: 0.0000 ± 0.0000
  Mann-Whitney U=130462.50, p=1.0000, d=0.000

Brainstem Out Degree:
  ASD:     0.0000 ± 0.0000
  Control: 0.0000 ± 0.0000
  Mann-Whitney U=130462.50, p=1.0000, d=0.000
======================================================================

INFO:src.analysis.diagnostics:Topology comparison saved → /home/nidszxh/Projects/Neuro-CXG/results/experiments/training/graphs/topology_comparison.png
INFO:__main__:  Graph analysis plots saved to: /home/nidszxh/Projects/Neuro-CXG/results/experiments/training/graphs
INFO:__main__:
======================================================================
INFO:__main__:TRAINING AND ANALYSIS COMPLETE
INFO:__main__:======================================================================

INFO:pipeline:Completed: GNN Training (5-Fold CV)
INFO:pipeline:Running: Generate Visualizations
2026-03-01 13:58:11,800 - __main__ - INFO - ============================================================
2026-03-01 13:58:11,800 - __main__ - INFO - NEURO-CXG VISUALIZATION PIPELINE
2026-03-01 13:58:11,800 - __main__ - INFO - ============================================================
2026-03-01 13:58:11,800 - __main__ - INFO - Output directory: /home/nidszxh/Projects/Neuro-CXG/results/visualizations
2026-03-01 13:58:11,800 - __main__ - INFO - Generating accuracy metrics visualization...
2026-03-01 13:58:11,800 - __main__ - WARNING - No training history files found. Run 5-fold CV with gnn_model.py first to generate history JSONs, then call this function again.
2026-03-01 13:58:11,800 - __main__ - INFO - Generating basic statistics visualizations...
2026-03-01 13:58:11,951 - src.features.graph_factory - INFO - ✓ Feature dimensions validated
2026-03-01 13:58:11,951 - src.features.graph_factory - INFO - Initialized train dataset with 702 subjects
2026-03-01 13:58:11,951 - src.features.graph_factory - INFO -   Node features: 28 (20 temporal+internal + 6 spatial)
2026-03-01 13:58:12,098 - src.features.graph_factory - INFO - ✓ Feature dimensions validated
2026-03-01 13:58:12,098 - src.features.graph_factory - INFO - Initialized val dataset with 150 subjects
2026-03-01 13:58:12,098 - src.features.graph_factory - INFO -   Node features: 28 (20 temporal+internal + 6 spatial)
2026-03-01 13:58:12,245 - src.features.graph_factory - INFO - ✓ Feature dimensions validated
2026-03-01 13:58:12,245 - src.features.graph_factory - INFO - Initialized test dataset with 146 subjects
2026-03-01 13:58:12,245 - src.features.graph_factory - INFO -   Node features: 28 (20 temporal+internal + 6 spatial)
2026-03-01 13:58:15,598 - __main__ - INFO - Saved dataset statistics to /home/nidszxh/Projects/Neuro-CXG/results/visualizations/dataset_statistics.png
2026-03-01 13:58:15,598 - __main__ - INFO - Running advanced feature importance analysis...
/home/nidszxh/Projects/Neuro-CXG/src/models/causal_gnn.py:119: UserWarning: 'nn.glob.GlobalAttention' is deprecated, use 'nn.aggr.AttentionalAggregation' instead
  self.att_pool = GlobalAttention(
2026-03-01 13:58:15,906 - src.features.graph_factory - INFO - ✓ Feature dimensions validated
2026-03-01 13:58:15,906 - src.features.graph_factory - INFO - Initialized test dataset with 146 subjects
2026-03-01 13:58:15,906 - src.features.graph_factory - INFO -   Node features: 28 (20 temporal+internal + 6 spatial)
2026-03-01 13:58:16,019 - src.analysis.feature_attribution - INFO - FeatureAttributionAnalyzer initialized
2026-03-01 13:58:16,019 - src.analysis.feature_attribution - INFO -   Device: cuda
2026-03-01 13:58:16,019 - src.analysis.feature_attribution - INFO -   Features: 28
2026-03-01 13:58:16,019 - src.analysis.feature_attribution - INFO - Computing feature attributions...
2026-03-01 13:58:16,019 - src.analysis.feature_attribution - INFO -   Method: Gradient-based Saliency
Computing attributions:   0%|                                                                                                                                               | 0/5 [00:00<?, ?it/s]
2026-03-01 13:58:16,163 - __main__ - ERROR - Advanced feature importance failed: mat1 and mat2 shapes cannot be multiplied (32x128 and 131x128)
Traceback (most recent call last):
  File "/home/nidszxh/Projects/Neuro-CXG/src/analysis/visualizations.py", line 409, in run_visualization_pipeline
    attributions = analyzer.compute_attributions()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/nidszxh/Projects/Neuro-CXG/src/analysis/feature_attribution.py", line 130, in compute_attributions
    out = self.model(
          ^^^^^^^^^^^
  File "/home/nidszxh/.venv312/lib64/python3.12/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/nidszxh/.venv312/lib64/python3.12/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/nidszxh/Projects/Neuro-CXG/src/models/causal_gnn.py", line 273, in forward
    class_logits = self.classifier(g)
                   ^^^^^^^^^^^^^^^^^^
  File "/home/nidszxh/.venv312/lib64/python3.12/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/nidszxh/.venv312/lib64/python3.12/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/nidszxh/.venv312/lib64/python3.12/site-packages/torch/nn/modules/container.py", line 250, in forward
    input = module(input)
            ^^^^^^^^^^^^^
  File "/home/nidszxh/.venv312/lib64/python3.12/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/nidszxh/.venv312/lib64/python3.12/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/nidszxh/.venv312/lib64/python3.12/site-packages/torch/nn/modules/linear.py", line 134, in forward
    return F.linear(input, self.weight, self.bias)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x128 and 131x128)
2026-03-01 13:58:16,167 - __main__ - INFO - Generating training history visualizations...
2026-03-01 13:58:16,167 - src.analysis.diagnostics - INFO - TrainingMonitor initialised — output: /home/nidszxh/Projects/Neuro-CXG/results/experiments/training, folds: 5
2026-03-01 13:58:16,845 - src.analysis.diagnostics - INFO - Training curves saved → /home/nidszxh/Projects/Neuro-CXG/results/experiments/training/fold_plots/training_curves_fold_0.png
2026-03-01 13:58:17,494 - src.analysis.diagnostics - INFO - Training curves saved → /home/nidszxh/Projects/Neuro-CXG/results/experiments/training/fold_plots/training_curves_fold_1.png
2026-03-01 13:58:18,154 - src.analysis.diagnostics - INFO - Training curves saved → /home/nidszxh/Projects/Neuro-CXG/results/experiments/training/fold_plots/training_curves_fold_2.png
2026-03-01 13:58:18,915 - src.analysis.diagnostics - INFO - Training curves saved → /home/nidszxh/Projects/Neuro-CXG/results/experiments/training/fold_plots/training_curves_fold_3.png
2026-03-01 13:58:19,560 - src.analysis.diagnostics - INFO - Training curves saved → /home/nidszxh/Projects/Neuro-CXG/results/experiments/training/fold_plots/training_curves_fold_4.png
2026-03-01 13:58:19,862 - src.analysis.diagnostics - INFO - Fold comparison saved → /home/nidszxh/Projects/Neuro-CXG/results/experiments/training/fold_comparison.png
2026-03-01 13:58:19,862 - __main__ - INFO - Training history visualizations completed
2026-03-01 13:58:19,862 - __main__ - INFO - Running graph topology analysis...
2026-03-01 13:58:19,862 - __main__ - ERROR - Graph analysis failed: name 'pd' is not defined
Traceback (most recent call last):
  File "/home/nidszxh/Projects/Neuro-CXG/src/analysis/visualizations.py", line 460, in run_visualization_pipeline
    manifest = pd.read_csv(MASTER_MANIFEST)
               ^^
NameError: name 'pd' is not defined
2026-03-01 13:58:19,863 - __main__ - INFO - ============================================================
2026-03-01 13:58:19,863 - __main__ - INFO - VISUALIZATION PIPELINE COMPLETE
2026-03-01 13:58:19,863 - __main__ - INFO - All visualizations saved to: /home/nidszxh/Projects/Neuro-CXG/results/visualizations
2026-03-01 13:58:19,863 - __main__ - INFO - ============================================================
INFO:pipeline:Completed: Generate Visualizations
INFO:pipeline:Running: Comprehensive Evaluation
INFO:__main__:Device       : cuda
INFO:__main__:Output dir   : /home/nidszxh/Projects/Neuro-CXG/results/evaluation
INFO:__main__:Loading datasets…
INFO:src.features.graph_factory:✓ Feature dimensions validated
INFO:src.features.graph_factory:Initialized test dataset with 146 subjects
INFO:src.features.graph_factory:  Node features: 28 (20 temporal+internal + 6 spatial)
INFO:src.features.graph_factory:✓ Feature dimensions validated
INFO:src.features.graph_factory:Initialized train dataset with 702 subjects
INFO:src.features.graph_factory:  Node features: 28 (20 temporal+internal + 6 spatial)
INFO:__main__:  Test: 146  Train: 702
INFO:__main__:============================================================
INFO:__main__:SECTION 1 — ENSEMBLE TEST-SET EVALUATION
INFO:__main__:============================================================
/home/nidszxh/Projects/Neuro-CXG/src/models/causal_gnn.py:119: UserWarning: 'nn.glob.GlobalAttention' is deprecated, use 'nn.aggr.AttentionalAggregation' instead
  self.att_pool = GlobalAttention(
INFO:__main__:  Fold 0  AUC=0.6064  (n=146)
/home/nidszxh/Projects/Neuro-CXG/src/models/causal_gnn.py:119: UserWarning: 'nn.glob.GlobalAttention' is deprecated, use 'nn.aggr.AttentionalAggregation' instead
  self.att_pool = GlobalAttention(
INFO:__main__:  Fold 1  AUC=0.6124  (n=146)
/home/nidszxh/Projects/Neuro-CXG/src/models/causal_gnn.py:119: UserWarning: 'nn.glob.GlobalAttention' is deprecated, use 'nn.aggr.AttentionalAggregation' instead
  self.att_pool = GlobalAttention(
INFO:__main__:  Fold 2  AUC=0.5624  (n=146)
/home/nidszxh/Projects/Neuro-CXG/src/models/causal_gnn.py:119: UserWarning: 'nn.glob.GlobalAttention' is deprecated, use 'nn.aggr.AttentionalAggregation' instead
  self.att_pool = GlobalAttention(
INFO:__main__:  Fold 3  AUC=0.5961  (n=146)
/home/nidszxh/Projects/Neuro-CXG/src/models/causal_gnn.py:119: UserWarning: 'nn.glob.GlobalAttention' is deprecated, use 'nn.aggr.AttentionalAggregation' instead
  self.att_pool = GlobalAttention(
INFO:__main__:  Fold 4  AUC=0.6291  (n=146)
/home/nidszxh/Projects/Neuro-CXG/src/models/causal_gnn.py:119: UserWarning: 'nn.glob.GlobalAttention' is deprecated, use 'nn.aggr.AttentionalAggregation' instead
  self.att_pool = GlobalAttention(
/home/nidszxh/Projects/Neuro-CXG/src/models/causal_gnn.py:119: UserWarning: 'nn.glob.GlobalAttention' is deprecated, use 'nn.aggr.AttentionalAggregation' instead
  self.att_pool = GlobalAttention(
/home/nidszxh/Projects/Neuro-CXG/src/models/causal_gnn.py:119: UserWarning: 'nn.glob.GlobalAttention' is deprecated, use 'nn.aggr.AttentionalAggregation' instead
  self.att_pool = GlobalAttention(
/home/nidszxh/Projects/Neuro-CXG/src/models/causal_gnn.py:119: UserWarning: 'nn.glob.GlobalAttention' is deprecated, use 'nn.aggr.AttentionalAggregation' instead
  self.att_pool = GlobalAttention(
/home/nidszxh/Projects/Neuro-CXG/src/models/causal_gnn.py:119: UserWarning: 'nn.glob.GlobalAttention' is deprecated, use 'nn.aggr.AttentionalAggregation' instead
  self.att_pool = GlobalAttention(
INFO:__main__:
─────────────────────────────────────────────
INFO:__main__:Metric            Value                95% CI
INFO:__main__:─────────────────────────────────────────────
INFO:__main__:  Auc            0.6047   [0.5193, 0.6941]
INFO:__main__:  Auprc          0.5833   [nan, nan]
INFO:__main__:  F1             0.7075   [0.6336, 0.7692]
INFO:__main__:  Accuracy       0.5753   [0.4932, 0.6507]
INFO:__main__:  Sensitivity    0.9868   [0.9571, 1.0000]
INFO:__main__:  Specificity    0.1286   [0.0580, 0.2069]
INFO:__main__:─────────────────────────────────────────────
INFO:__main__:  n_total=146  n_asd=76  n_control=70
INFO:__main__:  Threshold (max-F1): 0.5000
INFO:__main__:
  Per-fold AUC on test set:
INFO:__main__:    Fold 0  AUC=0.6064  F1=0.6909  Acc=0.5342
INFO:__main__:    Fold 1  AUC=0.6124  F1=0.6878  Acc=0.5274
INFO:__main__:    Fold 2  AUC=0.5624  F1=0.6970  Acc=0.5890
INFO:__main__:    Fold 3  AUC=0.5961  F1=0.6878  Acc=0.5274
INFO:__main__:    Fold 4  AUC=0.6291  F1=0.6847  Acc=0.5205
INFO:__main__:============================================================
INFO:__main__:SECTION 2 — PERMUTATION SIGNIFICANCE TEST  (n=1000)
INFO:__main__:============================================================
INFO:__main__:  Observed AUC : 0.6047
INFO:__main__:  Null AUC     : 0.4996 ± 0.0479  (mean ± std)
INFO:__main__:  p-value      : 0.0120  (✓ significant)
INFO:__main__:  Plot saved → /home/nidszxh/Projects/Neuro-CXG/results/evaluation/permutation_test.png
/home/nidszxh/Projects/Neuro-CXG/src/models/causal_gnn.py:119: UserWarning: 'nn.glob.GlobalAttention' is deprecated, use 'nn.aggr.AttentionalAggregation' instead
  self.att_pool = GlobalAttention(
INFO:__main__:============================================================
INFO:__main__:SECTION 3 — SUBGROUP ANALYSIS
INFO:__main__:============================================================
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/home/nidszxh/Projects/Neuro-CXG/src/run_evaluation.py", line 897, in <module>
    main()
  File "/home/nidszxh/Projects/Neuro-CXG/src/run_evaluation.py", line 875, in main
    sg_result = run_subgroup_analysis(
                ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/nidszxh/Projects/Neuro-CXG/src/run_evaluation.py", line 427, in run_subgroup_analysis
    probs_f, labels_f = _predict_probs(model, _build_loader(test_graphs, batch_size=1))
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/nidszxh/.venv312/lib64/python3.12/site-packages/torch/utils/_contextlib.py", line 120, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/nidszxh/Projects/Neuro-CXG/src/run_evaluation.py", line 148, in _predict_probs
    out = model(
          ^^^^^^
  File "/home/nidszxh/.venv312/lib64/python3.12/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/nidszxh/.venv312/lib64/python3.12/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/nidszxh/Projects/Neuro-CXG/src/models/causal_gnn.py", line 268, in forward
    demo = torch.stack([age.squeeze(), sex.squeeze(), fiq.squeeze()], dim=1)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)
ERROR:pipeline:Module src.run_evaluation failed with exit code 1