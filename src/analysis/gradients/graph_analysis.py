import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pathlib import Path
from typing import Dict, Optional
import logging
from tqdm import tqdm
from scipy.stats import mannwhitneyu
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
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
        self.manifest['subject_id'] = self.manifest['subject_id'].astype(str)
        
        # Region names from config (convert dict to ordered list for indexing)
        # LOBE_NAMES is {0: 'name', 1: 'name', ...} so we get values in order
        self.lobe_names = [LOBE_NAMES[i] for i in range(NUM_LOBES)]
        
        logger.info(f"CausalGraphAnalyzer initialized")
        logger.info(f"  Graphs directory: {graphs_dir}")
        logger.info(f"  Manifest subjects: {len(manifest)}")
    
    def compute_graph_properties(self, max_graphs: Optional[int] = None) -> pd.DataFrame:
        """
        Compute standard graph metrics for each subject.
        
        Metrics computed:
        - Number of edges
        - Graph density
        - Average clustering coefficient
        - In/out degree per lobe
        - Hub identification (high betweenness centrality)
        
        Args:
            max_graphs: Maximum number of graphs to analyze (None = all)
        
        Returns:
            DataFrame with graph properties per subject
        """
        logger.info("Computing graph properties...")
        
        graph_files = list(self.graphs_dir.glob("*_graph.pt"))
        
        if max_graphs is not None:
            graph_files = np.random.choice(graph_files, min(max_graphs, len(graph_files)), replace=False)
        
        logger.info(f"  Analyzing {len(graph_files)} graphs")
        
        results = []
        
        for graph_file in tqdm(graph_files, desc="Computing properties"):
            try:
                # Load graph
                graph_data = torch.load(graph_file, weights_only=False)
                subject_id = graph_file.stem.replace('_graph', '')
                
                # Get subject info from manifest
                sub_manifest = self.manifest[self.manifest['subject_id'] == subject_id]
                if len(sub_manifest) == 0:
                    continue
                
                dx_group = sub_manifest.iloc[0]['DX_GROUP']
                site_id = sub_manifest.iloc[0]['SITE_ID']
                
                # Convert to NetworkX directed graph
                adj = graph_data['adj'].numpy()
                G = nx.DiGraph(adj)
                
                # Basic metrics
                num_nodes = G.number_of_nodes()
                num_edges = G.number_of_edges()
                density = nx.density(G)
                
                # Clustering (use undirected version)
                G_undirected = G.to_undirected()
                try:
                    avg_clustering = nx.average_clustering(G_undirected)
                except:
                    avg_clustering = 0.0
                
                # Degree centrality
                in_degree = dict(G.in_degree())
                out_degree = dict(G.out_degree())
                
                # Betweenness centrality (hub identification)
                try:
                    betweenness = nx.betweenness_centrality(G)
                except:
                    betweenness = {i: 0.0 for i in range(NUM_LOBES)}
                
                # Store results
                result = {
                    'subject_id': subject_id,
                    'dx_group': dx_group,
                    'site_id': site_id,
                    'num_nodes': num_nodes,
                    'num_edges': num_edges,
                    'density': density,
                    'avg_clustering': avg_clustering
                }
                
                # Per-lobe metrics
                for lobe_id, lobe_name in enumerate(self.lobe_names):
                    result[f'{lobe_name.lower()}_in_degree'] = in_degree.get(lobe_id, 0)
                    result[f'{lobe_name.lower()}_out_degree'] = out_degree.get(lobe_id, 0)
                    result[f'{lobe_name.lower()}_betweenness'] = betweenness.get(lobe_id, 0.0)
                
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Error processing {graph_file.name}: {e}")
                continue
        
        results_df = pd.DataFrame(results)
        
        logger.info(f"✓ Computed properties for {len(results_df)} graphs")
        
        return results_df
    
    def compare_asd_vs_control(
        self,
        graph_metrics: pd.DataFrame,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Dict]:
        """
        Statistical comparison of graph properties between ASD and Control.
        
        Tests hypothesis: ASD subjects have altered causal connectivity.
        
        Args:
            graph_metrics: DataFrame from compute_graph_properties()
            output_dir: Optional directory to save visualizations
        
        Returns:
            Dictionary with comparison statistics
        """
        logger.info("Comparing ASD vs Control graph topology...")
        
        asd = graph_metrics[graph_metrics['dx_group'] == 1]
        control = graph_metrics[graph_metrics['dx_group'] == 2]
        
        logger.info(f"  ASD: {len(asd)} subjects")
        logger.info(f"  Control: {len(control)} subjects")
        
        # Metrics to compare
        metrics = [
            'num_edges', 'density', 'avg_clustering',
            'frontal_in_degree', 'frontal_out_degree',
            'temporal_in_degree', 'temporal_out_degree',
            'parietal_in_degree', 'parietal_out_degree',
            'occipital_in_degree', 'occipital_out_degree',
            'limbic_in_degree', 'limbic_out_degree'
        ]
        
        comparison_results = {}
        
        print("\n" + "="*70)
        print("GRAPH TOPOLOGY COMPARISON (ASD vs CONTROL)")
        print("="*70)
        
        for metric in metrics:
            if metric not in graph_metrics.columns:
                continue
            
            asd_vals = asd[metric].dropna().values
            control_vals = control[metric].dropna().values
            
            if len(asd_vals) == 0 or len(control_vals) == 0:
                continue
            
            # Mann-Whitney U test (non-parametric, robust to non-normal distributions)
            u_stat, p_value = mannwhitneyu(asd_vals, control_vals, alternative='two-sided')
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((asd_vals.std()**2 + control_vals.std()**2) / 2)
            cohens_d = (asd_vals.mean() - control_vals.mean()) / pooled_std if pooled_std > 0 else 0
            
            comparison_results[metric] = {
                'asd_mean': asd_vals.mean(),
                'asd_std': asd_vals.std(),
                'control_mean': control_vals.mean(),
                'control_std': control_vals.std(),
                'u_statistic': u_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'significant': p_value < 0.05
            }
            
            # Print results
            print(f"\n{metric.replace('_', ' ').title()}:")
            print(f"  ASD:     {asd_vals.mean():.4f} ± {asd_vals.std():.4f}")
            print(f"  Control: {control_vals.mean():.4f} ± {control_vals.std():.4f}")
            print(f"  Mann-Whitney U: {u_stat:.2f}, p={p_value:.4f}, d={cohens_d:.3f}")
            
            if p_value < 0.05:
                direction = "higher" if asd_vals.mean() > control_vals.mean() else "lower"
                effect = "large" if abs(cohens_d) > 0.8 else ("medium" if abs(cohens_d) > 0.5 else "small")
                print(f"  ✓ ASD has significantly {direction} {metric} ({effect} effect)")
        
        print("="*70 + "\n")
        
        # Visualize if output directory provided
        if output_dir is not None:
            self._plot_topology_comparison(graph_metrics, comparison_results, output_dir)
        
        return comparison_results
    
    def _plot_topology_comparison(
        self,
        graph_metrics: pd.DataFrame,
        comparison_results: Dict,
        output_dir: Path
    ):
        """Create visualization of topology differences."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Select top significant metrics
        significant_metrics = [
            k for k, v in comparison_results.items() 
            if v['significant']
        ][:6]  # Top 6
        
        if not significant_metrics:
            significant_metrics = list(comparison_results.keys())[:6]
        
        # Create box plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        asd = graph_metrics[graph_metrics['dx_group'] == 1]
        control = graph_metrics[graph_metrics['dx_group'] == 2]
        
        for idx, metric in enumerate(significant_metrics):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            
            data = [asd[metric].dropna().values, control[metric].dropna().values]
            bp = ax.boxplot(data, labels=['ASD', 'Control'], patch_artist=True)
            
            # Color boxes
            bp['boxes'][0].set_facecolor('#e74c3c')
            bp['boxes'][0].set_alpha(0.7)
            bp['boxes'][1].set_facecolor('#3498db')
            bp['boxes'][1].set_alpha(0.7)
            
            # Add significance marker
            if comparison_results[metric]['significant']:
                y_max = max(data[0].max(), data[1].max())
                ax.text(1.5, y_max * 1.1, '*', ha='center', fontsize=20, fontweight='bold')
                ax.text(1.5, y_max * 1.15, f"p={comparison_results[metric]['p_value']:.3f}", 
                       ha='center', fontsize=9)
            
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            ax.grid(alpha=0.3, axis='y')
        
        plt.suptitle('Graph Topology: ASD vs Control (Most Significant Differences)', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_dir / 'topology_comparison_significant.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Topology comparison plot saved to {output_dir}")
    
    def visualize_average_causal_graph(
        self,
        output_path: Path,
        max_graphs: Optional[int] = 500
    ):
        """
        Create average causal graph for ASD vs Control.
        
        Shows which connections are stronger in each group.
        
        Args:
            output_path: Path to save figure
            max_graphs: Maximum graphs to average (None = all)
        """
        logger.info("Computing average causal graphs...")
        
        graph_files = list(self.graphs_dir.glob("*_graph.pt"))
        
        if max_graphs is not None:
            graph_files = graph_files[:max_graphs]
        
        asd_graphs = []
        control_graphs = []
        
        for graph_file in tqdm(graph_files, desc="Loading graphs"):
            try:
                graph_data = torch.load(graph_file, weights_only=False)
                subject_id = graph_file.stem.replace('_graph', '')
                
                sub_manifest = self.manifest[self.manifest['subject_id'] == subject_id]
                if len(sub_manifest) == 0:
                    continue
                
                dx_group = sub_manifest.iloc[0]['DX_GROUP']
                adj = graph_data['adj'].numpy()
                
                if dx_group == 1:
                    asd_graphs.append(adj)
                else:
                    control_graphs.append(adj)
                    
            except Exception as e:
                logger.warning(f"Error loading {graph_file.name}: {e}")
                continue
        
        logger.info(f"  ASD graphs: {len(asd_graphs)}")
        logger.info(f"  Control graphs: {len(control_graphs)}")
        
        # Compute averages
        avg_asd = np.mean(asd_graphs, axis=0)
        avg_control = np.mean(control_graphs, axis=0)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # ASD average
        sns.heatmap(
            avg_asd, 
            annot=True, 
            fmt='.2f', 
            cmap='RdBu_r', 
            center=0,
            xticklabels=self.lobe_names, 
            yticklabels=self.lobe_names,
            ax=axes[0], 
            vmin=-0.5, 
            vmax=0.5, 
            cbar_kws={'label': 'Correlation'},
            linewidths=0.5
        )
        axes[0].set_title(f'ASD Average Causal Graph (n={len(asd_graphs)})', 
                         fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Target Lobe (t)', fontsize=12)
        axes[0].set_ylabel('Source Lobe (t-1)', fontsize=12)
        
        # Control average
        sns.heatmap(
            avg_control, 
            annot=True, 
            fmt='.2f', 
            cmap='RdBu_r', 
            center=0,
            xticklabels=self.lobe_names, 
            yticklabels=self.lobe_names,
            ax=axes[1], 
            vmin=-0.5, 
            vmax=0.5, 
            cbar_kws={'label': 'Correlation'},
            linewidths=0.5
        )
        axes[1].set_title(f'Control Average Causal Graph (n={len(control_graphs)})', 
                         fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Target Lobe (t)', fontsize=12)
        axes[1].set_ylabel('Source Lobe (t-1)', fontsize=12)
        
        # Difference (ASD - Control)
        diff = avg_asd - avg_control
        
        # Identify significant differences (simple threshold for visualization)
        # In production, use bootstrapping or permutation tests
        threshold = 0.05  # Arbitrary threshold for visualization
        significant_mask = np.abs(diff) > threshold
        
        sns.heatmap(
            diff, 
            annot=True, 
            fmt='.2f', 
            cmap='RdBu_r', 
            center=0,
            xticklabels=self.lobe_names, 
            yticklabels=self.lobe_names,
            ax=axes[2], 
            vmin=-0.3, 
            vmax=0.3, 
            cbar_kws={'label': 'Difference'},
            linewidths=0.5,
            mask=~significant_mask  # Highlight significant differences
        )
        axes[2].set_title('Difference (ASD - Control)', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Target Lobe (t)', fontsize=12)
        axes[2].set_ylabel('Source Lobe (t-1)', fontsize=12)
        
        plt.suptitle('Average Causal Connectivity Patterns', 
                    fontsize=16, fontweight='bold', y=1.0)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Average causal graphs saved to {output_path}")
        
        # Print key findings
        print("\n" + "="*70)
        print("AVERAGE CAUSAL GRAPH DIFFERENCES")
        print("="*70)
        
        # Find strongest differences
        diff_flat = []
        for i in range(NUM_LOBES):
            for j in range(NUM_LOBES):
                if i != j:  # Skip diagonal
                    diff_flat.append({
                        'connection': f'{self.lobe_names[i]} → {self.lobe_names[j]}',
                        'difference': diff[i, j],
                        'asd': avg_asd[i, j],
                        'control': avg_control[i, j]
                    })
        
        diff_df = pd.DataFrame(diff_flat).sort_values('difference', key=abs, ascending=False)
        
        print("\nTop 5 Strongest Differences (ASD - Control):")
        print(diff_df.head(5).to_string(index=False))
        print("="*70 + "\n")
    
    def identify_hubs(
        self,
        graph_metrics: pd.DataFrame,
        threshold_percentile: float = 75
    ) -> pd.DataFrame:
        """
        Identify hub lobes based on betweenness centrality.
        
        Args:
            graph_metrics: DataFrame from compute_graph_properties()
            threshold_percentile: Percentile for hub classification
        
        Returns:
            DataFrame with hub analysis
        """
        logger.info("Identifying hub lobes...")
        
        # Extract betweenness centrality columns
        betweenness_cols = [col for col in graph_metrics.columns if 'betweenness' in col]
        
        hub_results = []
        
        for lobe_col in betweenness_cols:
            lobe_name = lobe_col.replace('_betweenness', '').title()
            
            # Separate by diagnosis
            asd_vals = graph_metrics[graph_metrics['dx_group'] == 1][lobe_col].values
            control_vals = graph_metrics[graph_metrics['dx_group'] == 2][lobe_col].values
            
            # Compute threshold
            threshold = np.percentile(graph_metrics[lobe_col], threshold_percentile)
            
            # Count hubs
            asd_hubs = (asd_vals > threshold).sum()
            control_hubs = (control_vals > threshold).sum()
            
            hub_results.append({
                'lobe': lobe_name,
                'asd_hub_count': asd_hubs,
                'control_hub_count': control_hubs,
                'asd_mean_betweenness': asd_vals.mean(),
                'control_mean_betweenness': control_vals.mean()
            })
        
        hub_df = pd.DataFrame(hub_results)
        
        print("\n" + "="*70)
        print("HUB LOBE ANALYSIS")
        print("="*70)
        print(hub_df.to_string(index=False))
        print("="*70 + "\n")
        
        return hub_df


# Standalone execution
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add project root to path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    
    from src.core.config import CAUSAL_GRAPHS_DIR, MASTER_MANIFEST
    
    logger.info("="*70)
    logger.info("CAUSAL GRAPH STRUCTURE ANALYSIS")
    logger.info("="*70)
    
    # Load manifest
    manifest = pd.read_csv(MASTER_MANIFEST)
    
    # Initialize analyzer
    analyzer = CausalGraphAnalyzer(
        graphs_dir=CAUSAL_GRAPHS_DIR,
        manifest=manifest
    )
    
    # Compute graph properties
    graph_metrics = analyzer.compute_graph_properties(max_graphs=500)
    
    # Create output directory
    output_dir = Path("results/analysis/graph_structure")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    graph_metrics.to_csv(output_dir / 'graph_metrics.csv', index=False)
    logger.info(f"✓ Graph metrics saved to {output_dir / 'graph_metrics.csv'}")
    
    # Compare ASD vs Control
    comparison_results = analyzer.compare_asd_vs_control(graph_metrics, output_dir)
    
    # Visualize average graphs
    analyzer.visualize_average_causal_graph(output_dir / 'average_causal_graphs.png')
    
    # Identify hubs
    hub_analysis = analyzer.identify_hubs(graph_metrics)
    hub_analysis.to_csv(output_dir / 'hub_analysis.csv', index=False)
    
    logger.info("="*70)
    logger.info("✓ Graph structure analysis complete")
    logger.info(f"  Results saved to: {output_dir}")
    logger.info("="*70)