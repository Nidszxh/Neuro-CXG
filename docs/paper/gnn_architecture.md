# GNN Architecture

## Forward Pass Architecture

```mermaid
flowchart TB
    subgraph Input["Input Layer"]
        A["Node Features<br/>(12 lobes × 24 features)"]
        B["Edge Index<br/>(Directed edges)"]
        C["Edge Weights<br/>(Causal strengths)"]
    end

    subgraph Embedding["Feature Embedding"]
        D["Node Embedding<br/>(Learnable position embeddings)"]
        E["Linear: 24 → 128"]
        F["LayerNorm + GELU"]
    end

    subgraph GNNLayers["GATv2 Layers × 2"]
        G["GATv2 Head 1<br/>(Multi-head attention)"]
        H["GATv2 Head 2"]
        I["GATv2 Head 3"]
        J["GATv2 Head 4"]
    end

    subgraph EdgeGate["Edge Gate"]
        K["Edge Weight Gating<br/>(sigmoid(MLP))"]
    end

    subgraph Pooling["Anatomical Pooling"]
        L["Mean + Max + Sum<br/>(Pooling strategy)"]
    end

    subgraph Classifier["MLP Classifier"]
        M["Linear: 384 → 128"]
        N["GELU + Dropout"]
        O["Linear: 128 → 2"]
    end

    A --> D
    D --> E
    E --> F
    F --> G
    F --> H
    F --> I
    F --> J
    
    B --> K
    C --> K
    
    G --> L
    H --> L
    I --> L
    J --> L
    K -.-> L
    
    L --> M
    M --> N
    N --> O
    
    O --> P["Logits<br/>(Control, ASD)"]
```

## Component Details

| Component | Shape | Description |
|-----------|-------|-------------|
| Node Features | (batch, 12, 24) | 12 lobes × 24 features (temporal + frequency + spatial) |
| Node Embedding | (batch, 12, 16) | Learnable positional embeddings per lobe |
| Linear (in) | (24 + 16) → 128 | Project to hidden dimension |
| GATv2 Layers | 2 layers × 4 heads | Multi-head self-attention with edge gating |
| Edge Gate | MLP(sigmoid) | Learnable edge weight modulation |
| Pooling | mean + max + sum | Concatenate three pooling strategies → 384-dim |
| Classifier MLP | 384 → 128 → 2 | Final classification head |

## Optional Components

- **Site Embedding**: 20-dim site ID → learned embedding (enabled with `use_site_embedding=True`)
- **Demographics**: Age, sex, FIQ concatenated before classifier (enabled with `use_demographics=True`)
- **Gradient Reversal Layer (GRL)**: Domain adversarial training for site debiasing (enabled with `use_grl=True`)