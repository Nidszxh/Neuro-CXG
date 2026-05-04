"""
Phase 8.4 — Clinical / Literature Validation
==============================================
Cross-references model-identified important brain regions against established
ASD neuroscience literature to assess biological plausibility.

Known ASD-relevant Networks (LOBE_NAMES index → region name)
-------------------------------------------------------------
Default Mode Network (DMN)
    Regions: Frontal_Superior (0), Frontal_Orbital (1), Cingulate (4),
             Parietal (7), Temporal (8)
    ASD finding: Hypo-connectivity within DMN; reduced anti-correlation with
    task-positive networks (Kennedy & Courchesne 2008; Monk et al. 2009).

Social Brain Network
    Regions: Temporal (8), Limbic (5), Insula (3), Frontal_Orbital (1)
    ASD finding: Reduced activity during social cognition tasks
    (Dichter et al. 2012; Pelphrey et al. 2011).

Salience Network
    Regions: Insula (3), Cingulate (4), Motor_Premotor (2), Subcortical (9)
    ASD finding: Reduced salience network connectivity; impaired interoception
    (Uddin et al. 2013; Paulus & Stein 2006).

Sensorimotor Network
    Regions: Motor_Premotor (2), Parietal (7)
    ASD finding: Atypical motor cortex organisation (Mostofsky et al. 2009).

Visual Network
    Regions: Occipital (6), Parietal (7)
    ASD finding: Enhanced low-level visual processing; altered V1 responses
    (Simmons et al. 2009).

Subcortical / Thalamo-Cortical
    Regions: Subcortical (9), Brainstem (11), Cerebellum (10)
    ASD finding: Thalamic hypo-connectivity to frontal/parietal regions;
    cerebellar differences in timing (Belmonte et al. 2004).

Public Functions
----------------
validate_important_regions(top_region_indices, top_n) -> Dict
    Cross-reference a ranked list of regions against all networks above.

generate_report(results, output_path) -> Path
    Write a structured JSON + human-readable text report.

generate_validation_figure(results, output_path) -> Path
    Heatmap of network membership vs. importance rank.
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import LOBE_NAMES, NUM_LOBES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REGION_LABELS: list[str] = [LOBE_NAMES[i] for i in range(NUM_LOBES)]

# ── Network Definitions ────────────────────────────────────────────────────────

# Each entry: { "name": str, "regions": List[int], "finding": str, "refs": List[str] }
KNOWN_NETWORKS: list[dict] = [
    {
        "name": "Default Mode Network (DMN)",
        "short": "DMN",
        "regions": [0, 1, 4, 7, 8],   # Frontal_Superior, Frontal_Orbital, Cingulate, Parietal, Temporal
        "asd_finding": (
            "Hypo-connectivity within DMN; reduced anti-correlation with task-positive "
            "networks; persistent DMN activation during task performance."
        ),
        "refs": [
            "Kennedy & Courchesne (2008), Biol Psychiatry",
            "Monk et al. (2009), NeuroImage",
            "Assaf et al. (2010), NeuroImage",
        ],
    },
    {
        "name": "Social Brain Network",
        "short": "Social",
        "regions": [1, 3, 5, 8],       # Frontal_Orbital, Insula, Limbic, Temporal
        "asd_finding": (
            "Reduced activity and connectivity during social cognition, theory-of-mind tasks, "
            "and face perception; atypical amygdala-temporal coupling."
        ),
        "refs": [
            "Dichter et al. (2012), JAMA Psychiatry",
            "Pelphrey et al. (2011), Trends Cogn Sci",
            "Schultz et al. (2000), Nature",
        ],
    },
    {
        "name": "Salience Network",
        "short": "Salience",
        "regions": [2, 3, 4, 9],       # Motor_Premotor, Insula, Cingulate, Subcortical
        "asd_finding": (
            "Reduced salience network connectivity; impaired interoceptive awareness "
            "via anterior insula; hypo-connectivity between insula and ACC."
        ),
        "refs": [
            "Uddin et al. (2013), JAMA Psychiatry",
            "Paulus & Stein (2006), Trends Cogn Sci",
            "Sridharan et al. (2008), PNAS",
        ],
    },
    {
        "name": "Sensorimotor Network",
        "short": "Sensorimotor",
        "regions": [2, 7],             # Motor_Premotor, Parietal
        "asd_finding": (
            "Atypical motor cortex organisation; reduced parietal involvement in "
            "action observation; mirror neuron system differences."
        ),
        "refs": [
            "Mostofsky et al. (2009), Cortex",
            "Dinstein et al. (2010), Neuron",
        ],
    },
    {
        "name": "Visual Network",
        "short": "Visual",
        "regions": [6, 7],             # Occipital, Parietal
        "asd_finding": (
            "Enhanced low-level visual processing with reduced top-down modulation; "
            "altered V1 surround suppression; local processing bias."
        ),
        "refs": [
            "Simmons et al. (2009), NeuroImage",
            "Dakin & Frith (2005), Neuron",
        ],
    },
    {
        "name": "Subcortical / Thalamo-Cortical",
        "short": "Subcortical",
        "regions": [9, 10, 11],        # Subcortical, Cerebellum, Brainstem
        "asd_finding": (
            "Thalamic hypo-connectivity to frontal and parietal regions; cerebellar "
            "differences in timing and prediction; enlarged amygdala in young children."
        ),
        "refs": [
            "Belmonte et al. (2004), J Neurosci",
            "Allen et al. (2004), Trends Neuropsci",
            "Courchesne et al. (2001), Neurology",
        ],
    },
    {
        "name": "Frontoparietal Control Network",
        "short": "FPC",
        "regions": [0, 2, 7],          # Frontal_Superior, Motor_Premotor, Parietal
        "asd_finding": (
            "Reduced cognitive control network efficiency; weaker fronto-parietal "
            "coupling during executive function tasks."
        ),
        "refs": [
            "Solomon et al. (2009), NeuroImage",
            "Just et al. (2004), Brain",
        ],
    },
]

# Build a region → networks lookup
REGION_TO_NETWORKS: dict[int, list[str]] = {i: [] for i in range(NUM_LOBES)}
for net in KNOWN_NETWORKS:
    for r in net["regions"]:
        REGION_TO_NETWORKS[r].append(net["short"])


# ── Core validation logic ──────────────────────────────────────────────────────

def validate_important_regions(
    top_region_indices: list[int],
    top_n: int = 5,
) -> dict:
    """
    Cross-reference the top-N most important regions against known ASD networks.

    Parameters
    ----------
    top_region_indices : List[int]
        Brain region indices ranked from most to least important (e.g. from
        GradCAM or attention analysis).
    top_n : int
        Number of top regions to analyse.

    Returns
    -------
    results : Dict
        {
          "top_regions": [{"index": int, "name": str, "networks": List[str]}, ...],
          "network_coverage": {network_short: {"hit": bool, "regions_found": List[str]}},
          "overlap_scores": {network_short: float},   # Jaccard similarity
          "summary": str,
        }
    """
    top_n = min(top_n, len(top_region_indices))
    top_regions = top_region_indices[:top_n]

    # Per-region info
    region_info = []
    for idx in top_regions:
        region_info.append(
            {
                "index": int(idx),
                "name": REGION_LABELS[idx],
                "networks": REGION_TO_NETWORKS.get(int(idx), []),
            }
        )

    # Network coverage
    top_set = {int(i) for i in top_regions}
    coverage: dict[str, dict] = {}
    overlap_scores: dict[str, float] = {}
    for net in KNOWN_NETWORKS:
        s = net["short"]
        net_set = set(net["regions"])
        intersection = top_set & net_set
        union        = top_set | net_set
        jaccard      = len(intersection) / len(union) if union else 0.0
        coverage[s] = {
            "hit": len(intersection) > 0,
            "regions_found": [REGION_LABELS[i] for i in sorted(intersection)],
            "total_in_network": len(net_set),
            "network_regions": [REGION_LABELS[i] for i in sorted(net_set)],
        }
        overlap_scores[s] = round(jaccard, 4)

    # Summary sentence
    hit_networks = [s for s, v in coverage.items() if v["hit"]]
    top_names    = [REGION_LABELS[i] for i in top_regions]
    summary      = (
        f"Top-{top_n} important regions: {', '.join(top_names)}.\n"
        f"These overlap with {len(hit_networks)}/{len(KNOWN_NETWORKS)} known ASD-relevant networks: "
        f"{', '.join(hit_networks) if hit_networks else 'none'}."
    )
    logger.info(summary)

    return {
        "top_regions":    region_info,
        "network_coverage": coverage,
        "overlap_scores": overlap_scores,
        "summary":        summary,
        "top_n":          top_n,
    }


def generate_report(results: dict, output_dir: Path) -> Path:
    """
    Write a JSON + human-readable text report to *output_dir*.

    Parameters
    ----------
    results : Dict
        Output of ``validate_important_regions()``.
    output_dir : Path

    Returns
    -------
    text_path : Path
        Path to the generated ``.txt`` report.
    """
    import textwrap

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON dump
    json_path = output_dir / "literature_validation.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("JSON report saved → %s", json_path)

    # Human-readable text
    lines = [
        "=" * 72,
        "NEURO-CXG  –  PHASE 8.4 CLINICAL VALIDATION REPORT",
        "=" * 72,
        "",
        results["summary"],
        "",
        "-" * 72,
        f"TOP-{results['top_n']} REGIONS RANKED BY MODEL IMPORTANCE",
        "-" * 72,
    ]
    for rank, r in enumerate(results["top_regions"], start=1):
        nets = ", ".join(r["networks"]) if r["networks"] else "—"
        lines.append(f"  {rank}. {r['name']:25s}  (networks: {nets})")

    lines += ["", "-" * 72, "NETWORK COVERAGE ANALYSIS", "-" * 72]
    for net in KNOWN_NETWORKS:
        s    = net["short"]
        cov  = results["network_coverage"][s]
        jac  = results["overlap_scores"][s]
        hit  = "✓" if cov["hit"] else "✗"
        lines.append(f"\n{hit}  {net['name']}")
        lines.append(f"   Jaccard similarity : {jac:.4f}")
        lines.append(f"   Network regions    : {', '.join(cov['network_regions'])}")
        if cov["hit"]:
            lines.append(f"   Matched by model   : {', '.join(cov['regions_found'])}")
        lines.append(f"   ASD finding        : {textwrap.fill(net['asd_finding'], 70, subsequent_indent='     ')}")
        lines.append(f"   Key refs           : {'; '.join(net['refs'])}")

    lines += ["", "=" * 72, "END OF REPORT", "=" * 72]

    text_path = output_dir / "literature_validation.txt"
    with open(text_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Text report saved → %s", text_path)

    # Print to stdout (captured by pipeline runner)
    print("\n".join(lines))
    return text_path


def generate_validation_figure(results: dict, output_path: Path) -> Path:
    """
    Heatmap: top-N regions (rows) × known networks (cols), colour = network membership.
    A second colour overlay shows importance rank.

    Parameters
    ----------
    results : Dict  — output of validate_important_regions()
    output_path : Path

    Returns
    -------
    output_path : Path
    """
    import seaborn as sns

    top_regions = results["top_regions"]
    network_shorts = [n["short"] for n in KNOWN_NETWORKS]
    region_names   = [r["name"] for r in top_regions]

    # Membership matrix (binary)
    mat = np.zeros((len(top_regions), len(KNOWN_NETWORKS)), dtype=float)
    for ri, r in enumerate(top_regions):
        for ni, s in enumerate(network_shorts):
            if s in r["networks"]:
                mat[ri, ni] = 1.0
            # Fade non-members to 0.2 so the heatmap has contrast
        # Encode rank as intensity: rank 1 = 1.0, rank N = 1/N
        importance = 1.0 / (ri + 1)
        for ni in range(len(KNOWN_NETWORKS)):
            if mat[ri, ni] > 0:
                mat[ri, ni] = importance

    fig, ax = plt.subplots(figsize=(14, max(4, len(top_regions) * 0.7 + 2)))
    sns.heatmap(
        mat,
        xticklabels=network_shorts,
        yticklabels=region_names,
        cmap="YlOrRd",
        vmin=0, vmax=1,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Importance (1/rank) × membership"},
    )
    ax.set_title(
        "Literature Validation: Top Regions vs Known ASD Networks\n"
        "(colour intensity = importance rank; blank = not in network)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("ASD-Relevant Network", fontsize=11)
    ax.set_ylabel("Brain Region (ranked by model)", fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Validation figure saved → %s", out)
    return out


# ── Convenience wrapper ────────────────────────────────────────────────────────

def run_literature_validation(
    gradcam_asd_scores:     np.ndarray | None = None,
    attention_asd_scores:   np.ndarray | None = None,
    output_dir: Path = Path("results/explainability/literature"),
    top_n: int = 6,
) -> dict:
    """
    Convenience function called by ``run_explainability.py``.

    Takes GradCAM and/or attention scores (one value per region, length=NUM_LOBES),
    averages them if both are provided, ranks regions, then runs the full
    validate → report → figure pipeline.

    Parameters
    ----------
    gradcam_asd_scores : np.ndarray (NUM_LOBES,), optional
    attention_asd_scores : np.ndarray (NUM_LOBES,), optional
    output_dir : Path
    top_n : int

    Returns
    -------
    Dict — validation results
    """
    # Combine scores
    score_arrays = [s for s in [gradcam_asd_scores, attention_asd_scores] if s is not None]
    if not score_arrays:
        logger.warning("No importance scores provided to literature validation; using random order")
        combined = np.arange(NUM_LOBES, dtype=float)
        np.random.shuffle(combined)
    elif len(score_arrays) == 1:
        combined = score_arrays[0]
    else:
        # Normalise to [0,1] then average
        def _norm(a: np.ndarray) -> np.ndarray:
            rng = a.max() - a.min()
            return (a - a.min()) / (rng + 1e-9)
        combined = np.mean([_norm(s) for s in score_arrays], axis=0)

    ranked_indices: list[int] = np.argsort(combined)[::-1].tolist()

    results = validate_important_regions(ranked_indices, top_n=top_n)
    report_path = generate_report(results, output_dir)
    fig_path    = generate_validation_figure(
        results, Path(output_dir) / "literature_validation_heatmap.png"
    )
    results["report_path"] = str(report_path)
    results["figure_path"] = str(fig_path)
    return results
