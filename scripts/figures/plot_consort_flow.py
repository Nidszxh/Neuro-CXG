"""
CONSORT-Style Flow Diagram
==========================

Generates subject flow diagram following CONSORT guidelines for clinical/medical papers.

Output: results/paper_figures/consort_flow.png
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

SUBJECT_FLOW = {
    "initial": {"n": 1112, "label": "ABIDE-I Assessed for Eligibility"},
    "excluded": {"n": 97, "reasons": ["Phantoms (24)", "Excluded subjects (73)"]},
    "analyzed": {"n": 1015, "label": "Analyzed (n=1015)"},

    "train_val": {"n": 707, "label": "Training + Validation (n=707)"},
    "test": {"n": 308, "label": "Test Set (n=308)"},

    "train": {"n": 566, "label": "Training (n=566)"},
    "val": {"n": 141, "label": "Validation (n=141)"},

    "train_detailed": {
        "asd": 341, "control": 225,
        "sites": 13
    },
    "val_detailed": {
        "asd": 82, "control": 59,
        "sites": 13
    },
    "test_detailed": {
        "asd": 175, "control": 133,
        "sites": 13
    }
}


def generate_consort_diagram(output_dir: Path | None = None, dpi: int = 300) -> Path:
    """Generate CONSORT-style subject flow diagram."""

    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "results" / "paper_figures"

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 14))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 14)
    ax.axis("off")

    box_width = 4.0
    box_height = 0.8
    box_style = "round,pad=0.05,rounding_size=0.1"

    def draw_box(x, y, width, height, text, color="#E6E6E6", text_color="black"):
        FancyBboxPatch((x - width/2, y - height/2), width, height,
                      boxstyle=box_style, facecolor=color, edgecolor="black",
                      linewidth=1.5).set_zorder(2)
        ax.add_patch(FancyBboxPatch((x - width/2, y - height/2), width, height,
                                    boxstyle=box_style, facecolor=color, edgecolor="black",
                                    linewidth=1.5))
        ax.text(x, y, text, ha="center", va="center", fontsize=10,
               fontweight="bold", color=text_color, zorder=3)

    def draw_arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
                   zorder=1)

    draw_box(6, 13, box_width, box_height, "Assessed for Eligibility (n=1112)", "#D5E8D4")

    draw_arrow(6, 12.6, 6, 11.8)

    draw_box(6, 11.3, box_width, box_height, "Not Eligible (n=97)", "#F8C8C8")
    ax.text(6, 10.7, "• Phantoms: 24", ha="center", va="top", fontsize=8)
    ax.text(6, 10.3, "• Corrupted volumes: 73", ha="center", va="top", fontsize=8)

    draw_arrow(6, 10.1, 6, 9.3)

    draw_box(6, 8.8, box_width, box_height, "Analyzed (n=1015)", "#D5E8D4")

    draw_arrow(6, 8.4, 4, 7.6)

    draw_box(4, 7.1, 4, box_height, "Training + Validation (n=707)", "#DAE8FC")
    ax.text(4, 6.5, "ASD: 423, Control: 284", ha="center", va="top", fontsize=8)
    ax.text(4, 6.1, "13 sites", ha="center", va="top", fontsize=8)

    draw_arrow(8, 7.6, 8, 7.1)
    draw_box(8, 6.6, 3.5, box_height, "Test Set (n=308)", "#FCF3CF")
    ax.text(8, 6.0, "ASD: 175, Control: 133", ha="center", va="top", fontsize=8)
    ax.text(8, 5.6, "13 sites", ha="center", va="top", fontsize=8)

    draw_arrow(4, 6.7, 2.5, 5.9)
    draw_box(2.5, 5.4, 3, box_height, "Training (n=566)", "#E1D5E7")
    ax.text(2.5, 4.8, "ASD: 341, Control: 225", ha="center", va="top", fontsize=8)

    draw_arrow(4, 6.7, 5.5, 5.9)
    draw_box(5.5, 5.4, 3, box_height, "Validation (n=141)", "#E1D5E7")
    ax.text(5.5, 4.8, "ASD: 82, Control: 59", ha="center", va="top", fontsize=8)

    ax.text(6, 1.0, "5-Fold Cross-Validation (Training + Validation)", ha="center", fontsize=9,
           fontstyle="italic", color="#666666")

    ax.text(6, 0.5, "Final model evaluated on held-out test set", ha="center", fontsize=9,
           fontstyle="italic", color="#666666")

    ax.set_title("Subject Flow Diagram (CONSORT Style)", fontsize=14, fontweight="bold", pad=20)

    legend_elements = [
        mpatches.Patch(facecolor="#D5E8D4", edgecolor="black", label="Initial/Analyzed"),
        mpatches.Patch(facecolor="#F8C8C8", edgecolor="black", label="Excluded"),
        mpatches.Patch(facecolor="#DAE8FC", edgecolor="black", label="Train/Val Split"),
        mpatches.Patch(facecolor="#FCF3CF", edgecolor="black", label="Test Set"),
        mpatches.Patch(facecolor="#E1D5E7", edgecolor="black", label="Final Splits"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()

    output_path = output_dir / "consort_flow.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"CONSORT flow diagram saved to: {output_path}")
    return output_path


def generate_mermaid_consort() -> str:
    """Generate Mermaid diagram for CONSORT flow."""

    mermaid = """
```mermaid
flowchart TD
    A[ABIDE-I Assessed<br/>n=1112] -->|Excluded| B[Not Eligible<br/>n=97]
    B -->|Reason| B1[Phantoms: 24]
    B -->|Reason| B2[Corrupted: 73]
    A -->|Eligible| C[Analyzed<br/>n=1015]
    C -->|Train+Val| D[Train+Val<br/>n=707]
    C -->|Test| E[Test Set<br/>n=308]
    D -->|Training| F[Training<br/>n=566]
    D -->|Validation| G[Validation<br/>n=141]

    style A fill:#D5E8D4,stroke:#333
    style B fill:#F8C8C8,stroke:#333
    style C fill:#D5E8D4,stroke:#333
    style D fill:#DAE8FC,stroke:#333
    style E fill:#FCF3CF,stroke:#333
    style F fill:#E1D5E7,stroke:#333
    style G fill:#E1D5E7,stroke:#333
```
"""
    return mermaid


if __name__ == "__main__":
    generate_consort_diagram()
