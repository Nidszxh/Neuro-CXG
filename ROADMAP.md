# Detailed Low-Level Project Workflow
## Explainable Causal Graph Neural Models for Neuroimaging-Based Brain Disorder Classification

---

## PHASE 1: PROJECT SETUP & ENVIRONMENT CONFIGURATION

### 1.1 Development Environment Setup
**Tasks:**
- Install Python 3.8+ environment
- Set up virtual environment (conda/venv)
- Install core dependencies:
  - PyTorch/TensorFlow for deep learning
  - PyTorch Geometric for GNN implementation
  - nibabel for neuroimaging data handling
  - nilearn for neuroimaging analysis
  - scikit-learn for ML utilities
  - NumPy, Pandas for data manipulation
- Install YOLO framework (YOLOv5/v8/v10)
- Install visualization libraries (matplotlib, seaborn, networkx)
- Set up GPU support and CUDA environment
- Configure Jupyter/IDE environment

**Deliverables:**
- `requirements.txt` with all dependencies
- Environment setup documentation
- GPU verification script

**Timeline:** 1 days

---

### 1.2 Project Structure & Repository Setup
**Tasks:**
- Create project directory structure:
  ```
  project/
  ├── data/
  │   ├── raw/
  │   ├── processed/
  │   └── splits/
  ├── models/
  │   ├── yolo/
  │   ├── gnn/
  │   └── checkpoints/
  ├── src/
  │   ├── data_preprocessing/
  │   ├── roi_detection/
  │   ├── feature_extraction/
  │   ├── graph_construction/
  │   ├── gnn_models/
  │   ├── causal_reasoning/
  │   └── explainability/
  ├── notebooks/
  ├── results/
  ├── configs/
  └── tests/
  ```
- Initialize Git repository
- Create `.gitignore` for large files
- Set up experiment tracking (WandB/MLflow)
- Create configuration files (YAML/JSON)

**Deliverables:**
- Organized project structure
- Version control setup
- Configuration templates

**Timeline:** 1 days

---

## PHASE 2: DATA ACQUISITION & EXPLORATION

### 2.1 ABIDE Dataset Download
**Tasks:**
- Register for ABIDE data access
- Download preprocessed rs-fMRI data from ABIDE I and/or ABIDE II
- Download demographic information (age, sex, site)
- Download phenotypic data (diagnosis labels, severity scores)
- Verify data integrity (checksums, file completeness)
- Document data preprocessing pipeline used by ABIDE

**Deliverables:**
- Downloaded dataset (organized by subject ID)
- Data manifest file (subject IDs, file paths, labels)
- Data acquisition documentation

**Timeline:** 2-3 days

---

### 2.2 Data Organization & Quality Control
**Tasks:**
- Create subject metadata database (CSV/JSON)
- Map subject IDs to diagnosis labels (ASD/Control)
- Check for missing data or corrupted files
- Document inclusion/exclusion criteria
- Balance dataset or note class imbalance
- Create train/validation/test splits (e.g., 70/15/15)
- Ensure site distribution across splits
- Stratify by diagnosis, age, and site

**Deliverables:**
- `subjects_metadata.csv` with all annotations
- Data quality report
- Train/val/test split files
- Data statistics summary

**Timeline:** 2-3 days

---

### 2.3 Exploratory Data Analysis (EDA)
**Tasks:**
- Visualize sample fMRI volumes
- Analyze data distributions (signal intensity, temporal characteristics)
- Check demographic distributions (age, sex, site effects)
- Visualize class distribution (ASD vs Control)
- Identify potential confounding variables
- Document preprocessing already applied to ABIDE data
- Create EDA visualizations and report

**Deliverables:**
- EDA Jupyter notebook
- Distribution plots and statistics
- Data quality assessment report

**Timeline:** 3-4 days

---

## PHASE 3: ROI DETECTION USING YOLO

### 3.1 fMRI to Image Conversion
**Tasks:**
- Load 4D fMRI data (x, y, z, time) using nibabel
- Extract representative 2D slices:
  - Axial slices at multiple z-coordinates
  - Mean activation maps across time
  - Maximum intensity projections
- Normalize intensity values (0-255 for image format)
- Apply contrast enhancement if needed
- Save as image files (PNG/JPG) for YOLO input
- Create mapping: image filename → subject ID → original 3D coordinates

**Deliverables:**
- Image conversion pipeline script (`fmri_to_images.py`)
- Converted image dataset
- Coordinate mapping file
- Sample visualization of converted images

**Timeline:** 4-5 days

---

### 3.2 ROI Annotation for YOLO Training
**Tasks:**
- Define ROI categories based on neuroanatomy:
  - Example categories: prefrontal cortex, temporal lobe, parietal regions, occipital regions, subcortical structures, cerebellum
- Create annotation strategy (bounding boxes around ROIs)
- Select subset of subjects for annotation (e.g., 100-200 subjects)
- Use annotation tools (LabelImg, CVAT, Roboflow)
- Manually annotate ROIs in 2D brain slices
- Quality check annotations (inter-rater reliability if multiple annotators)
- Convert annotations to YOLO format (class, x_center, y_center, width, height)
- Split annotated data into train/val sets for YOLO

**Deliverables:**
- Annotated dataset in YOLO format
- Annotation guidelines document
- ROI category definitions
- Train/val split for YOLO training

**Timeline:** 10-14 days (annotation-intensive)

---

### 3.3 YOLO Model Training
**Tasks:**
- Choose YOLO version (YOLOv5, YOLOv8, or YOLOv10)
- Configure YOLO hyperparameters:
  - Image size (e.g., 640x640)
  - Batch size based on GPU memory
  - Learning rate, epochs, augmentation
- Set up data augmentation (rotation, flipping, brightness)
- Initialize with pre-trained weights (transfer learning)
- Train YOLO model on annotated brain slice data
- Monitor training metrics (mAP, precision, recall, loss)
- Implement early stopping
- Validate on held-out validation set
- Fine-tune hyperparameters if needed

**Deliverables:**
- Trained YOLO model weights
- Training configuration file
- Training logs and curves
- Validation performance metrics

**Timeline:** 5-7 days

---

### 3.4 YOLO Model Evaluation & Optimization
**Tasks:**
- Test YOLO on held-out test images
- Calculate detection metrics:
  - Mean Average Precision (mAP)
  - Precision and recall per ROI class
  - Confidence score distributions
- Visualize detection results (bounding boxes overlaid)
- Identify failure cases and error patterns
- Optimize confidence threshold for detection
- Implement Non-Maximum Suppression (NMS) tuning
- Test inference speed
- Document model limitations

**Deliverables:**
- Test set performance report
- Detection visualization samples
- Optimized YOLO inference pipeline
- Error analysis document

**Timeline:** 3-4 days

---

### 3.5 ROI Detection on Full Dataset
**Tasks:**
- Apply trained YOLO model to all subjects in dataset
- For each subject:
  - Load converted brain images
  - Run YOLO inference
  - Extract bounding box coordinates
  - Extract confidence scores
  - Map 2D bounding boxes back to 3D ROI coordinates
- Filter low-confidence detections (threshold tuning)
- Save ROI locations per subject
- Handle cases with missing or extra ROIs
- Validate ROI consistency across subjects

**Deliverables:**
- ROI detection results for all subjects (`roi_detections.pkl`)
- Per-subject ROI coordinate files
- Detection quality metrics
- ROI detection pipeline script

**Timeline:** 3-4 days

---

## PHASE 4: ROI FEATURE EXTRACTION

### 4.1 Signal Extraction from Detected ROIs
**Tasks:**
- For each subject and each detected ROI:
  - Map 2D bounding box back to 3D volume coordinates
  - Define 3D ROI mask (sphere or box around center)
  - Extract time series from all voxels within ROI
  - Compute mean time series for the ROI
  - Handle boundary cases (ROIs near edge of brain)
- Validate extracted signals (check for NaN, outliers)
- Visualize sample time series per ROI

**Deliverables:**
- ROI signal extraction script (`extract_roi_signals.py`)
- Per-subject ROI time series data
- Signal quality validation report

**Timeline:** 4-5 days

---

### 4.2 Temporal Feature Computation
**Tasks:**
- For each ROI time series, compute:
  - **Statistical features:**
    - Mean, standard deviation, variance
    - Skewness, kurtosis
    - Range, percentiles
  - **Temporal features:**
    - Autocorrelation at different lags
    - Temporal entropy
  - **Frequency-domain features:**
    - Power spectral density (PSD)
    - Dominant frequency
    - Power in different frequency bands (low, high frequency)
  - **Signal complexity:**
    - Sample entropy
    - Hurst exponent
- Normalize features (z-score or min-max)
- Handle missing or invalid feature values

**Deliverables:**
- Feature extraction pipeline (`compute_roi_features.py`)
- Feature vectors for each ROI per subject
- Feature description document

**Timeline:** 5-6 days

---

### 4.3 Feature Quality Control & Selection
**Tasks:**
- Check feature distributions across subjects
- Identify and handle outliers
- Check for multicollinearity among features
- Perform feature importance analysis (preliminary)
- Select subset of most informative features (optional)
- Document feature engineering decisions
- Create feature visualization (distributions, correlations)

**Deliverables:**
- Feature quality report
- Selected feature set
- Feature correlation analysis
- Feature selection justification

**Timeline:** 3-4 days

---

## PHASE 5: BRAIN GRAPH CONSTRUCTION

### 5.1 Functional Connectivity Matrix Computation
**Tasks:**
- For each subject, compute pairwise connectivity between all ROIs:
  - **Pearson correlation** between ROI time series
  - **Partial correlation** (optional, more computationally intensive)
  - **Mutual information** (alternative measure)
- Create symmetric connectivity matrix (n_rois × n_rois)
- Handle numerical stability issues
- Fisher z-transform correlations if needed
- Visualize sample connectivity matrices

**Deliverables:**
- Connectivity computation script (`compute_connectivity.py`)
- Per-subject connectivity matrices
- Sample connectivity visualizations

**Timeline:** 4-5 days

---

### 5.2 Graph Adjacency Matrix Construction
**Tasks:**
- Convert connectivity matrices to adjacency matrices:
  - **Thresholding:** Keep top k% strongest connections
  - **Sparsification:** Remove weak edges below threshold
  - **Statistical thresholding:** Keep only significant correlations
- Experiment with different threshold values
- Create binary or weighted adjacency matrices
- Ensure graph connectivity (no isolated nodes)
- Compare different thresholding strategies
- Document chosen approach and rationale

**Deliverables:**
- Adjacency matrix construction script (`build_adjacency.py`)
- Per-subject adjacency matrices
- Threshold selection analysis
- Graph statistics (density, degree distribution)

**Timeline:** 3-4 days

---

### 5.3 Graph Object Creation
**Tasks:**
- Convert adjacency matrices to graph objects:
  - Use PyTorch Geometric `Data` objects
  - Store node features (ROI feature vectors)
  - Store edge indices and edge weights
  - Store graph-level label (ASD/Control)
- Add metadata (subject ID, demographics)
- Validate graph structures
- Compute graph-level statistics:
  - Number of nodes, edges
  - Average degree, clustering coefficient
  - Graph diameter, modularity
- Create graph dataset class for PyTorch Geometric

**Deliverables:**
- Graph dataset creation script (`create_graph_dataset.py`)
- PyTorch Geometric dataset of brain graphs
- Graph statistics summary
- Sample graph visualizations

**Timeline:** 4-5 days

---

### 5.4 Graph-Level Analysis
**Tasks:**
- Compute and compare graph metrics between ASD and Control groups:
  - Global efficiency, local efficiency
  - Modularity, small-worldness
  - Hub identification (node centrality measures)
- Perform statistical tests (t-tests, permutation tests)
- Visualize group differences in connectivity patterns
- Identify regions with altered connectivity in ASD
- Document graph-theoretic findings

**Deliverables:**
- Graph analysis notebook
- Statistical comparison results
- Group-level connectivity visualizations
- Preliminary findings report

**Timeline:** 3-4 days

---

## PHASE 6: GRAPH NEURAL NETWORK DEVELOPMENT

### 6.1 GNN Architecture Design
**Tasks:**
- Choose GNN architecture:
  - Graph Convolutional Network (GCN)
  - Graph Attention Network (GAT)
  - GraphSAGE
  - Graph Isomorphism Network (GIN)
- Design model layers:
  - Number of graph convolution layers (2-5)
  - Hidden dimensions per layer
  - Pooling strategy (global mean/max/attention pooling)
  - Final classification head (MLP)
- Add regularization (dropout, layer normalization)
- Design multi-layer architecture diagram
- Implement model class in PyTorch Geometric

**Deliverables:**
- GNN model architecture code (`gnn_models.py`)
- Architecture diagram
- Model configuration files
- Parameter count and complexity analysis

**Timeline:** 5-6 days

---

### 6.2 Training Pipeline Implementation
**Tasks:**
- Implement data loaders for graph batching
- Define loss function (binary cross-entropy)
- Choose optimizer (Adam, AdamW) and learning rate
- Implement learning rate scheduling
- Set up training loop:
  - Forward pass
  - Loss computation
  - Backward pass and optimization
  - Gradient clipping if needed
- Implement validation loop
- Add checkpointing (save best model)
- Implement early stopping
- Set up experiment logging (WandB/TensorBoard)

**Deliverables:**
- Training pipeline script (`train_gnn.py`)
- Configuration files for training
- Logging and checkpointing setup
- Training utilities module

**Timeline:** 5-6 days

---

### 6.3 Model Training & Hyperparameter Tuning
**Tasks:**
- Train baseline GNN model
- Monitor training metrics:
  - Training/validation loss
  - Training/validation accuracy
  - Overfitting indicators
- Perform hyperparameter search:
  - Learning rate (1e-5 to 1e-2)
  - Hidden dimensions (64, 128, 256)
  - Number of layers (2, 3, 4, 5)
  - Dropout rate (0.1 to 0.5)
  - Pooling strategy
  - Batch size
- Use grid search or random search
- Apply k-fold cross-validation
- Select best hyperparameters based on validation performance

**Deliverables:**
- Trained GNN models (multiple configurations)
- Hyperparameter search results
- Training curves and logs
- Best model checkpoint

**Timeline:** 7-10 days

---

### 6.4 Model Evaluation
**Tasks:**
- Evaluate best model on test set
- Compute classification metrics:
  - Accuracy, precision, recall, F1-score
  - ROC-AUC, PR-AUC
  - Confusion matrix
  - Sensitivity and specificity
- Perform statistical significance testing
- Compare against baseline methods:
  - Traditional ML (SVM, Random Forest on features)
  - Simple MLP on concatenated ROI features
- Analyze per-site performance (handle site effects)
- Document model strengths and limitations

**Deliverables:**
- Test set evaluation report
- Performance metrics table
- ROC and PR curves
- Baseline comparison results
- Statistical analysis

**Timeline:** 4-5 days

---

## PHASE 7: CAUSAL REASONING LAYER

### 7.1 Causal Direction Inference Design
**Tasks:**
- Research causal discovery methods for fMRI:
  - Granger causality
  - Transfer entropy
  - Convergent Cross Mapping (CCM)
  - PC algorithm or similar constraint-based methods
  - Linear Non-Gaussian Acyclic Model (LiNGAM)
- Choose lightweight approach suitable for fMRI data
- Design integration with existing undirected graphs
- Plan computational requirements

**Deliverables:**
- Causal inference method selection document
- Theoretical background and justification
- Implementation plan

**Timeline:** 3-4 days

---

### 7.2 Temporal Precedence Analysis
**Tasks:**
- For each pair of connected ROIs:
  - Compute time-lagged correlations
  - Test for temporal precedence using Granger causality
  - Compute transfer entropy (directional information flow)
- Handle multiple time lags (1-5 TRs)
- Create directed edges based on significant causal relationships
- Construct directed adjacency matrices
- Compare directed vs undirected connectivity patterns

**Deliverables:**
- Causal inference computation script (`compute_causality.py`)
- Directed adjacency matrices per subject
- Temporal precedence analysis results
- Comparison report (directed vs undirected)

**Timeline:** 6-7 days

---

### 7.3 Directed Graph Construction
**Tasks:**
- Convert causal relationships to directed graph structures
- Create directed PyTorch Geometric `Data` objects
- Handle edge directionality in GNN architecture:
  - Use directed message passing if available
  - Or create bidirectional edges with different weights
- Validate directed graph properties:
  - Check for cycles (DAG vs cyclic)
  - Analyze in-degree and out-degree distributions
- Visualize directed graphs (arrows showing causality)

**Deliverables:**
- Directed graph dataset
- Directed GNN-compatible data structures
- Directed graph visualizations
- Graph properties analysis

**Timeline:** 4-5 days

---

### 7.4 Causal GNN Training & Comparison
**Tasks:**
- Adapt GNN architecture for directed graphs
- Train causal GNN on directed brain graphs
- Use same training pipeline with directed data
- Compare performance: undirected GNN vs causal GNN
- Analyze learned patterns in directed connectivity
- Identify causal pathways specific to ASD
- Document improvements or differences from causality

**Deliverables:**
- Causal GNN model
- Training results for causal GNN
- Comparative performance analysis
- Causal pathway visualizations

**Timeline:** 5-6 days

---

## PHASE 8: EXPLAINABILITY MODULE

### 8.1 Node Importance Analysis
**Tasks:**
- Implement node importance methods:
  - **GradCAM for graphs:** Compute gradients w.r.t. node features
  - **Integrated Gradients:** Attribution method for node features
  - **Attention weights:** Extract from GAT layers if used
  - **Perturbation-based:** Remove nodes and measure impact
- Compute importance scores for each ROI per subject
- Aggregate importance across subjects (per class)
- Identify consistently important ROIs for ASD classification
- Validate biological plausibility of important regions

**Deliverables:**
- Node importance computation script (`node_importance.py`)
- Per-subject node importance scores
- Aggregated importance maps (ASD vs Control)
- Important ROI visualization on brain templates

**Timeline:** 6-7 days

---

### 8.2 Edge Importance Analysis
**Tasks:**
- Implement edge importance methods:
  - **Edge masking:** Remove edges and measure classification change
  - **Gradient-based edge attribution**
  - **Attention on edges:** If using attention-based GNN
- Compute importance for each edge (connection) per subject
- Identify critical functional/causal connections for classification
- Analyze differences in important edges: ASD vs Control
- Visualize important subnetworks (connectomes)

**Deliverables:**
- Edge importance computation script (`edge_importance.py`)
- Per-subject edge importance scores
- Critical connectivity patterns visualization
- Subnetwork analysis (disorder-specific circuits)

**Timeline:** 6-7 days

---

### 8.3 Saliency Map Generation
**Tasks:**
- Create saliency maps overlaid on brain anatomy:
  - Map important ROIs to MNI coordinates
  - Use brain visualization tools (nilearn, BrainNetViewer)
  - Color-code by importance score
- Generate connectivity visualizations:
  - Chord diagrams
  - Circular network plots
  - 3D brain renderings with connections
- Create per-subject explanation visualizations
- Generate group-level explanation visualizations

**Deliverables:**
- Visualization pipeline (`visualize_explanations.py`)
- Saliency maps for sample subjects
- Group-level importance visualizations
- High-quality figures for publication

**Timeline:** 5-6 days

---

### 8.4 Clinical Validation & Interpretation
**Tasks:**
- Review identified important regions with neuroimaging literature
- Validate against known ASD-related brain regions:
  - Default Mode Network (DMN)
  - Social brain network
  - Salience network
  - Executive function networks
- Compare findings with previous ASD neuroimaging studies
- Consult with neuroscience/clinical experts if possible
- Document alignment and discrepancies with literature
- Assess clinical utility of explanations

**Deliverables:**
- Literature comparison report
- Clinical validation document
- Neuroscience interpretation of findings
- Discussion of clinical implications

**Timeline:** 4-5 days

---

## PHASE 9: FINAL INTEGRATION & EVALUATION

### 9.1 End-to-End Pipeline Integration
**Tasks:**
- Integrate all components into unified pipeline:
  - Data loading → ROI detection → Feature extraction → Graph construction → GNN prediction → Explanation generation
- Create master script that runs full pipeline for new subjects
- Implement error handling and logging throughout
- Optimize computational efficiency
- Add progress tracking and intermediate result saving
- Create pipeline configuration file
- Test on sample subjects

**Deliverables:**
- End-to-end pipeline script (`run_pipeline.py`)
- Pipeline configuration template
- Pipeline documentation
- Test run results

**Timeline:** 5-6 days

---

### 9.2 Comprehensive Model Evaluation
**Tasks:**
- Perform final evaluation on test set:
  - Classification performance metrics
  - Explanation quality metrics
  - Computational efficiency metrics (inference time)
- Conduct ablation studies:
  - Impact of ROI detection vs atlas-based ROIs
  - Impact of causal layer
  - Impact of different GNN architectures
  - Impact of different connectivity measures
- Perform sensitivity analysis
- Test robustness to data quality variations
- Compare with state-of-the-art methods from literature

**Deliverables:**
- Comprehensive evaluation report
- Ablation study results
- Sensitivity analysis
- Benchmarking comparison table

**Timeline:** 6-7 days

---

### 9.3 Result Interpretation & Analysis
**Tasks:**
- Analyze per-subject predictions and explanations
- Identify patterns in misclassified cases
- Examine relationship between explanation patterns and clinical severity
- Investigate site effects and confounds
- Perform subgroup analysis (by age, sex, etc.)
- Validate that important regions align with ASD neurobiology
- Create case studies of interesting subjects

**Deliverables:**
- Result interpretation document
- Case study examples
- Error analysis report
- Subgroup analysis results

**Timeline:** 5-6 days

---

## PHASE 10: DOCUMENTATION & DISSEMINATION

### 10.1 Code Documentation
**Tasks:**
- Add docstrings to all functions and classes
- Create API documentation (using Sphinx)
- Write detailed README with:
  - Project overview
  - Installation instructions
  - Usage examples
  - Dataset preparation guide
- Create tutorial notebooks (Jupyter)
- Document all configuration options
- Add inline comments for complex code sections
- Create developer guide for extending the project

**Deliverables:**
- Fully documented codebase
- README.md
- API documentation website
- Tutorial notebooks
- Developer guide

**Timeline:** 6-7 days

---

### 10.2 Experimental Documentation
**Tasks:**
- Document all experiments and results:
  - Experimental setup
  - Hyperparameter choices and rationale
  - Training procedures
  - Evaluation protocols
- Create tables summarizing all experimental results
- Compile all visualization figures
- Organize supplementary materials
- Write methods section (detailed technical description)
- Create results section with figures and tables

**Deliverables:**
- Experimental log document
- Results summary tables
- Organized figure repository
- Draft methods and results sections

**Timeline:** 5-6 days

---

### 10.3 Research Paper Preparation
**Tasks:**
- Write manuscript sections:
  - Abstract
  - Introduction (motivation, background, related work)
  - Methods (detailed technical approach)
  - Results (quantitative and qualitative findings)
  - Discussion (interpretation, limitations, future work)
  - Conclusion
- Create publication-quality figures
- Format references and citations
- Write supplementary materials
- Select target journal/conference
- Format according to submission guidelines
- Internal review and revisions

**Deliverables:**
- Complete manuscript draft
- Publication-ready figures
- Supplementary materials
- Formatted submission package

**Timeline:** 15-20 days

---

### 10.4 Code Release Preparation
**Tasks:**
- Clean up code repository
- Remove hardcoded paths and credentials
- Create requirements.txt with exact versions
- Add license file (e.g., MIT, Apache 2.0)
- Create example data and pretrained models
- Set up continuous integration (optional)
- Prepare GitHub repository for public release
- Create release notes and version tagging
- Add badges (build status, license, etc.)

**Deliverables:**
- Clean public repository
- License and contribution guidelines
- Example data and models
- Release documentation

**Timeline:** 4-5 days