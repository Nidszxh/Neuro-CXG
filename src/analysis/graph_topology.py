import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import mannwhitneyu
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import LOBE_NAMES, NUM_LOBES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CausalGraphAnalyzer:
    """
    Analyze structural properties of causal brain graphs.

    Computes network metrics and compares topology between ASD and Control groups.
    """

    def __init__(self, graphs_dir: Path, manifest: pd.DataFrame):
        """
        Initialize analyzer.

        Args:
            graphs_dir: Directory containing *_graph.pt files
            manifest: DataFrame with subject_id, DX_GROUP, SITE_ID columns
        """
        self.graphs_dir = Path(graphs_dir)
        self.manifest = manifest.copy()
        self.manifest["subject_id"] = self.manifest["subject_id"].astype(str)

        self.lobe_names = [LOBE_NAMES[i] for i in range(NUM_LOBES)]

        logger.info("CausalGraphAnalyzer initialized")
        logger.info(f"  Graphs directory: {graphs_dir}")
        logger.info(f"  Manifest subjects: {len(manifest)}")

    def compute_graph_properties(self, max_graphs: Optional[int] = None) -> pd.DataFrame:
        """
        Compute standard graph metrics for each subject.
        """
        logger.info("Computing graph properties...")

        graph_files = list(self.graphs_dir.glob("*_graph.pt"))

        if max_graphs is not None:
            graph_files = np.random.choice(graph_files, min(max_graphs, len(graph_files)), replace=False)

        logger.info(f"  Analyzing {len(graph_files)} graphs")

        results = []

        for graph_file in tqdm(graph_files, desc="Computing properties"):
            try:
                graph_data = torch.load(graph_file, weights_only=False)
                subject_id = graph_file.stem.replace("_graph", "")

                sub_manifest = self.manifest[self.manifest["subject_id"] == subject_id]
                if len(sub_manifest) == 0:
                    continue

                dx_group = sub_manifest.iloc[0]["DX_GROUP"]
                site_id = sub_manifest.iloc[0]["SITE_ID"]

                adj = graph_data["adj"].numpy()
                G = nx.DiGraph(adj)

                num_nodes = G.number_of_nodes()
                num_edges = G.number_of_edges()
                density = nx.density(G)

                G_undirected = G.to_undirected()
                try:
                    avg_clustering = nx.average_clustering(G_undirected)
                except Exception:
                    avg_clustering = 0.0

                in_degree = dict(G.in_degree())
                out_degree = dict(G.out_degree())

                try:
                    betweenness = nx.betweenness_centrality(G)
                except Exception:
                    betweenness = {i: 0.0 for i in range(NUM_LOBES)}

                result = {
                    "subject_id": subject_id,
                    "dx_group": dx_group,
                    "site_id": site_id,
                    "num_nodes": num_nodes,
                    "num_edges": num_edges,
                    "density": density,
                    "avg_clustering": avg_clustering,
                }

                for lobe_id, lobe_name in enumerate(self.lobe_names):
                    result[f"{lobe_name.lower()}_in_degree"] = in_degree.get(lobe_id, 0)
                    result[f"{lobe_name.lower()}_out_degree"] = out_degree.get(lobe_id, 0)
                    result[f"{lobe_name.lower()}_betweenness"] = betweenness.get(lobe_id, 0.0)

                results.append(result)

            except Exception as e:
                logger.warning(f"Error processing {graph_file.name}: {e}")
                continue

        results_df = pd.DataFrame(results)

        logger.info(f"Computed properties for {len(results_df)} graphs")
        return results_df

    def compare_asd_vs_control(self, graph_metrics: pd.DataFrame, output_dir: Optional[Path] = None) -> Dict[str, Dict]:
        """
        Statistical comparison of graph properties between ASD and Control.
        """
        logger.info("Comparing ASD vs Control graph topology...")

        asd = graph_metrics[graph_metrics["dx_group"] == 1]
        control = graph_metrics[graph_metrics["dx_group"] == 2]

        logger.info(f"  ASD: {len(asd)} subjects")
        logger.info(f"  Control: {len(control)} subjects")

        metrics = [
            "num_edges",
            "density",
            "avg_clustering",
            "frontal_in_degree",
            "frontal_out_degree",
            "temporal_in_degree",
            "temporal_out_degree",
            "parietal_in_degree",
            "parietal_out_degree",
            "occipital_in_degree",
            "occipital_out_degree",
            "limbic_in_degree",
            "limbic_out_degree",
        ]

        comparison_results = {}

        print("\n" + "=" * 70)
        print("GRAPH TOPOLOGY COMPARISON (ASD vs CONTROL)")
        print("=" * 70)

        for metric in metrics:
            if metric not in graph_metrics.columns:
                continue

            asd_vals = asd[metric].dropna().values
            control_vals = control[metric].dropna().values

            if len(asd_vals) == 0 or len(control_vals) == 0:
                continue

            u_stat, p_value = mannwhitneyu(asd_vals, control_vals, alternative="two-sided")

            pooled_std = np.sqrt((asd_vals.std() ** 2 + control_vals.std() ** 2) / 2)
            cohens_d = (asd_vals.mean() - control_vals.mean()) / pooled_std if pooled_std > 0 else 0

            comparison_results[metric] = {
                "asd_mean": asd_vals.mean(),
                "asd_std": asd_vals.std(),
                "control_mean": control_vals.mean(),
                "control_std": control_vals.std(),
                "u_statistic": u_stat,
                "p_value": p_value,
                "cohens_d": cohens_d,
                "significant": p_value < 0.05,
            }

            print(f"\n{metric.replace('_', ' ').title()}:")
            print(f"  ASD:     {asd_vals.mean():.4f} ± {asd_vals.std():.4f}")
            print(f"  Control: {control_vals.mean():.4f} ± {control_vals.std():.4f}")
            print(f"  Mann-Whitney U: {u_stat:.2f}, p={p_value:.4f}, d={cohens_d:.3f}")

            if p_value < 0.05:
                direction = "higher" if asd_vals.mean() > control_vals.mean() else "lower"
                effect = "large" if abs(cohens_d) > 0.8 else ("medium" if abs(cohens_d) > 0.5 else "small")
                print(f"  ASD has significantly {direction} {metric} ({effect} effect)")

        print("=" * 70 + "\n")

        if output_dir is not None:
            self._plot_topology_comparison(graph_metrics, comparison_results, output_dir)

        return comparison_results

    def _plot_topology_comparison(self, graph_metrics: pd.DataFrame, comparison_results: Dict, output_dir: Path) -> None:
        """Create visualization of topology differences."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        significant_metrics = [k for k, v in comparison_results.items() if v["significant"]][:6]
        if not significant_metrics:
            significant_metrics = list(comparison_results.keys())[:6]

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        asd = graph_metrics[graph_metrics["dx_group"] == 1]
        control = graph_metrics[graph_metrics["dx_group"] == 2]

        for idx, metric in enumerate(significant_metrics):
            ax = axes[idx]
            sns.boxplot(
                data=graph_metrics,
                x="dx_group",
                y=metric,
                ax=ax,
                palette={1: "#e74c3c", 2: "#3498db"},
            )
            ax.set_title(metric.replace("_", " ").title())
            ax.set_xlabel("Diagnosis (1=ASD, 2=Control)")
            ax.set_ylabel(metric.replace("_", " ").title())

        plt.tight_layout()
        plt.savefig(output_dir / "topology_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Topology comparison saved to {output_dir / 'topology_comparison.png'}")
