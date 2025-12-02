"""
Evaluation and visualization module for synthetic brand data.
Compares original vs synthetic data and evaluates clustering improvements.
Enhanced with academic-quality visualizations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, silhouette_samples
import warnings
warnings.filterwarnings('ignore')


class BrandDataEvaluator:
    """
    Evaluate synthetic brand data quality and clustering improvements.
    """

    def __init__(self):
        """Initialize the evaluator."""
        self.results = {}

    def compare_distributions(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame,
                             numerical_cols: List[str]) -> Dict:
        """
        Compare distributions of numerical features.

        Args:
            real_data: Original dataset
            synthetic_data: Synthetic dataset
            numerical_cols: List of numerical column names

        Returns:
            Dictionary of KS test statistics
        """
        print("\n=== Comparing Distributions ===")
        ks_results = {}

        for col in numerical_cols:
            if col in real_data.columns and col in synthetic_data.columns:
                # Kolmogorov-Smirnov test
                statistic, pvalue = stats.ks_2samp(
                    real_data[col].dropna(),
                    synthetic_data[col].dropna()
                )
                ks_results[col] = {'statistic': statistic, 'pvalue': pvalue}

                if pvalue > 0.05:
                    print(f"  {col}: PASS (p={pvalue:.4f}, stat={statistic:.4f})")
                else:
                    print(f"  {col}: DIFF (p={pvalue:.4f}, stat={statistic:.4f})")

        self.results['ks_tests'] = ks_results
        return ks_results

    def compare_correlations(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame,
                            numerical_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compare correlation matrices.

        Args:
            real_data: Original dataset
            synthetic_data: Synthetic dataset
            numerical_cols: List of numerical column names

        Returns:
            Tuple of (real_corr, synthetic_corr)
        """
        print("\n=== Comparing Correlations ===")

        common_cols = [col for col in numerical_cols
                      if col in real_data.columns and col in synthetic_data.columns]

        real_corr = real_data[common_cols].corr()
        synth_corr = synthetic_data[common_cols].corr()

        # Compute correlation difference
        corr_diff = np.abs(real_corr - synth_corr).mean().mean()
        print(f"Mean absolute correlation difference: {corr_diff:.4f}")

        self.results['correlation_diff'] = corr_diff
        return real_corr, synth_corr

    def plot_distribution_comparison(self, real_data: pd.DataFrame,
                                     synthetic_data: pd.DataFrame,
                                     features: List[str],
                                     n_cols: int = 3,
                                     figsize: Tuple[int, int] = (15, 10)):
        """
        Plot distribution comparisons for multiple features.

        Args:
            real_data: Original dataset
            synthetic_data: Synthetic dataset
            features: List of features to plot
            n_cols: Number of columns in subplot grid
            figsize: Figure size
        """
        n_features = len(features)
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_features > 1 else [axes]

        for idx, feature in enumerate(features):
            ax = axes[idx]

            if feature in real_data.columns and feature in synthetic_data.columns:
                # Plot distributions
                ax.hist(real_data[feature].dropna(), alpha=0.5, label='Real', bins=30, density=True)
                ax.hist(synthetic_data[feature].dropna(), alpha=0.5, label='Synthetic', bins=30, density=True)
                ax.set_xlabel(feature)
                ax.set_ylabel('Density')
                ax.legend()
                ax.set_title(f'{feature} Distribution')

        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.show()

    def plot_correlation_heatmaps(self, real_corr: np.ndarray, synth_corr: np.ndarray,
                                 figsize: Tuple[int, int] = (16, 6)):
        """
        Plot correlation heatmaps side by side.

        Args:
            real_corr: Real data correlation matrix
            synth_corr: Synthetic data correlation matrix
            figsize: Figure size
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Real data correlation
        sns.heatmap(real_corr, ax=ax1, cmap='coolwarm', center=0, vmin=-1, vmax=1,
                   square=True, cbar_kws={'label': 'Correlation'})
        ax1.set_title('Real Data Correlations')

        # Synthetic data correlation
        sns.heatmap(synth_corr, ax=ax2, cmap='coolwarm', center=0, vmin=-1, vmax=1,
                   square=True, cbar_kws={'label': 'Correlation'})
        ax2.set_title('Synthetic Data Correlations')

        plt.tight_layout()
        plt.show()

    def evaluate_clustering(self, data: pd.DataFrame, numerical_cols: List[str],
                          n_clusters: int = None, method: str = 'ward') -> Dict:
        """
        Perform hierarchical clustering and evaluate quality.

        Args:
            data: Dataset to cluster
            numerical_cols: Columns to use for clustering
            n_clusters: Number of clusters (None = auto)
            method: Linkage method for hierarchical clustering

        Returns:
            Dictionary of clustering results
        """
        print(f"\n=== Hierarchical Clustering ===")

        # Select features
        X = data[numerical_cols].fillna(0).values

        # Perform clustering
        if n_clusters is None:
            # Auto-determine clusters using dendrogram
            from scipy.cluster.hierarchy import dendrogram, linkage
            Z = linkage(X, method=method)
            # Use simple heuristic: find natural break in dendrogram
            n_clusters = 5  # Default

        clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage=method)
        labels = clusterer.fit_predict(X)

        # Compute metrics
        silhouette = silhouette_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)

        # Cluster distribution
        unique, counts = np.unique(labels, return_counts=True)
        cluster_dist = dict(zip(unique, counts))

        results = {
            'n_clusters': n_clusters,
            'silhouette_score': silhouette,
            'davies_bouldin_score': davies_bouldin,
            'cluster_distribution': cluster_dist,
            'labels': labels
        }

        print(f"Number of clusters: {n_clusters}")
        print(f"Silhouette score: {silhouette:.4f} (higher is better)")
        print(f"Davies-Bouldin score: {davies_bouldin:.4f} (lower is better)")
        print(f"Cluster sizes: {cluster_dist}")

        return results

    def compare_clustering(self, original_data: pd.DataFrame,
                          augmented_data: pd.DataFrame,
                          numerical_cols: List[str]) -> Dict:
        """
        Compare clustering results before and after augmentation.

        Args:
            original_data: Original dataset
            augmented_data: Original + synthetic combined
            numerical_cols: Columns for clustering

        Returns:
            Dictionary with comparison results
        """
        print("\n" + "="*60)
        print("CLUSTERING COMPARISON: Original vs Augmented")
        print("="*60)

        print("\n--- ORIGINAL DATA ---")
        original_results = self.evaluate_clustering(original_data, numerical_cols)

        print("\n--- AUGMENTED DATA (Original + Synthetic) ---")
        augmented_results = self.evaluate_clustering(augmented_data, numerical_cols)

        # Compute improvements
        silhouette_improvement = augmented_results['silhouette_score'] - original_results['silhouette_score']
        db_improvement = original_results['davies_bouldin_score'] - augmented_results['davies_bouldin_score']

        print("\n" + "="*60)
        print("IMPROVEMENTS")
        print("="*60)
        print(f"Silhouette improvement: {silhouette_improvement:+.4f}")
        print(f"Davies-Bouldin improvement: {db_improvement:+.4f} (negative means worse)")

        comparison = {
            'original': original_results,
            'augmented': augmented_results,
            'silhouette_improvement': silhouette_improvement,
            'davies_bouldin_improvement': db_improvement
        }

        self.results['clustering_comparison'] = comparison
        return comparison

    def plot_pca_comparison(self, original_data: pd.DataFrame,
                           synthetic_data: pd.DataFrame,
                           numerical_cols: List[str],
                           figsize: Tuple[int, int] = (14, 6)):
        """
        Plot PCA visualization of original vs synthetic data.

        Args:
            original_data: Original dataset
            synthetic_data: Synthetic dataset
            numerical_cols: Columns for PCA
            figsize: Figure size
        """
        # Prepare data
        X_orig = original_data[numerical_cols].fillna(0).values
        X_synth = synthetic_data[numerical_cols].fillna(0).values

        # PCA
        pca = PCA(n_components=2)
        X_orig_pca = pca.fit_transform(X_orig)
        X_synth_pca = pca.transform(X_synth)

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Original data
        ax1.scatter(X_orig_pca[:, 0], X_orig_pca[:, 1], alpha=0.5, label='Original')
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        ax1.set_title('Original Data (PCA)')
        ax1.legend()

        # Combined
        ax2.scatter(X_orig_pca[:, 0], X_orig_pca[:, 1], alpha=0.5, label='Original', s=30)
        ax2.scatter(X_synth_pca[:, 0], X_synth_pca[:, 1], alpha=0.5, label='Synthetic', s=30)
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        ax2.set_title('Original vs Synthetic (PCA)')
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def generate_report(self) -> str:
        """
        Generate a text summary report of all evaluations.

        Returns:
            Report string
        """
        report = []
        report.append("="*60)
        report.append("SYNTHETIC DATA EVALUATION REPORT")
        report.append("="*60)

        if 'ks_tests' in self.results:
            report.append("\n--- Distribution Similarity (KS Tests) ---")
            passing = sum(1 for v in self.results['ks_tests'].values() if v['pvalue'] > 0.05)
            total = len(self.results['ks_tests'])
            report.append(f"Passing features: {passing}/{total} ({passing/total*100:.1f}%)")

        if 'correlation_diff' in self.results:
            report.append("\n--- Correlation Preservation ---")
            report.append(f"Mean absolute difference: {self.results['correlation_diff']:.4f}")

        if 'clustering_comparison' in self.results:
            comp = self.results['clustering_comparison']
            report.append("\n--- Clustering Quality ---")
            report.append(f"Original silhouette: {comp['original']['silhouette_score']:.4f}")
            report.append(f"Augmented silhouette: {comp['augmented']['silhouette_score']:.4f}")
            report.append(f"Improvement: {comp['silhouette_improvement']:+.4f}")

        report.append("\n" + "="*60)
        return "\n".join(report)

    # ==================== ENHANCED ACADEMIC VISUALIZATIONS ====================

    def plot_dendrogram(self, data: pd.DataFrame, numerical_cols: List[str],
                       method: str = 'ward', figsize: Tuple[int, int] = (16, 8),
                       max_display: int = 50):
        """
        Plot hierarchical clustering dendrogram.

        Args:
            data: Dataset to cluster
            numerical_cols: Columns for clustering
            method: Linkage method
            figsize: Figure size
            max_display: Maximum number of leaf nodes to display
        """
        X = data[numerical_cols].fillna(0).values
        Z = linkage(X, method=method)

        plt.figure(figsize=figsize)
        dendrogram(Z, truncate_mode='lastp', p=max_display)
        plt.title(f'Hierarchical Clustering Dendrogram ({method} linkage)', fontsize=14, fontweight='bold')
        plt.xlabel('Sample Index or (Cluster Size)', fontsize=12)
        plt.ylabel('Distance', fontsize=12)
        plt.tight_layout()
        plt.show()

    def plot_dendrogram_comparison(self, original_data: pd.DataFrame,
                                   augmented_data: pd.DataFrame,
                                   numerical_cols: List[str],
                                   method: str = 'ward',
                                   figsize: Tuple[int, int] = (18, 7)):
        """
        Side-by-side dendrogram comparison.

        Args:
            original_data: Original dataset
            augmented_data: Augmented dataset
            numerical_cols: Columns for clustering
            method: Linkage method
            figsize: Figure size
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Original
        X_orig = original_data[numerical_cols].fillna(0).values
        Z_orig = linkage(X_orig, method=method)
        dendrogram(Z_orig, ax=ax1, truncate_mode='lastp', p=30)
        ax1.set_title(f'Original Data Dendrogram\n({len(original_data)} brands)', fontsize=13, fontweight='bold')
        ax1.set_xlabel('Sample Index or (Cluster Size)')
        ax1.set_ylabel('Distance')

        # Augmented
        X_aug = augmented_data[numerical_cols].fillna(0).values
        Z_aug = linkage(X_aug, method=method)
        dendrogram(Z_aug, ax=ax2, truncate_mode='lastp', p=30)
        ax2.set_title(f'Augmented Data Dendrogram\n({len(augmented_data)} brands)', fontsize=13, fontweight='bold')
        ax2.set_xlabel('Sample Index or (Cluster Size)')
        ax2.set_ylabel('Distance')

        plt.tight_layout()
        plt.show()

    def plot_silhouette_analysis(self, data: pd.DataFrame, numerical_cols: List[str],
                                 labels: np.ndarray, n_clusters: int,
                                 figsize: Tuple[int, int] = (10, 7)):
        """
        Plot silhouette analysis for clusters.

        Args:
            data: Dataset
            numerical_cols: Columns used for clustering
            labels: Cluster labels
            n_clusters: Number of clusters
            figsize: Figure size
        """
        X = data[numerical_cols].fillna(0).values

        fig, ax = plt.subplots(figsize=figsize)

        # Silhouette scores for each sample
        silhouette_vals = silhouette_samples(X, labels)
        silhouette_avg = silhouette_score(X, labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate silhouette scores for samples in cluster i
            cluster_silhouette_vals = silhouette_vals[labels == i]
            cluster_silhouette_vals.sort()

            size_cluster_i = cluster_silhouette_vals.shape[0]
            y_upper = y_lower + size_cluster_i

            color = plt.cm.nipy_spectral(float(i) / n_clusters)
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals,
                            facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, f'C{i}')
            y_lower = y_upper + 10

        ax.set_title(f'Silhouette Plot for {n_clusters} Clusters', fontsize=14, fontweight='bold')
        ax.set_xlabel('Silhouette Coefficient Values', fontsize=12)
        ax.set_ylabel('Cluster Label', fontsize=12)

        # Average silhouette score line
        ax.axvline(x=silhouette_avg, color="red", linestyle="--", linewidth=2,
                  label=f'Average: {silhouette_avg:.3f}')
        ax.legend()

        ax.set_yticks([])
        ax.set_xlim([-0.1, 1])
        plt.tight_layout()
        plt.show()

    def plot_tsne_comparison(self, original_data: pd.DataFrame,
                            synthetic_data: pd.DataFrame,
                            numerical_cols: List[str],
                            perplexity: int = 30,
                            figsize: Tuple[int, int] = (14, 6)):
        """
        Plot t-SNE visualization of original vs synthetic data.

        Args:
            original_data: Original dataset
            synthetic_data: Synthetic dataset
            numerical_cols: Columns for t-SNE
            perplexity: t-SNE perplexity parameter
            figsize: Figure size
        """
        print(f"Computing t-SNE (perplexity={perplexity})... This may take a moment.")

        X_orig = original_data[numerical_cols].fillna(0).values
        X_synth = synthetic_data[numerical_cols].fillna(0).values

        # t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        X_orig_tsne = tsne.fit_transform(X_orig)

        # Transform synthetic data (need to fit again for synthetic)
        tsne_synth = TSNE(n_components=2, perplexity=min(perplexity, len(X_synth)-1), random_state=42)
        X_synth_tsne = tsne_synth.fit_transform(X_synth)

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Original
        ax1.scatter(X_orig_tsne[:, 0], X_orig_tsne[:, 1], alpha=0.6, s=50, c='steelblue')
        ax1.set_title('Original Data (t-SNE)', fontsize=13, fontweight='bold')
        ax1.set_xlabel('t-SNE Component 1')
        ax1.set_ylabel('t-SNE Component 2')

        # Synthetic
        ax2.scatter(X_synth_tsne[:, 0], X_synth_tsne[:, 1], alpha=0.6, s=50, c='coral')
        ax2.set_title('Synthetic Data (t-SNE)', fontsize=13, fontweight='bold')
        ax2.set_xlabel('t-SNE Component 1')
        ax2.set_ylabel('t-SNE Component 2')

        plt.tight_layout()
        plt.show()

    def plot_feature_importance_pca(self, data: pd.DataFrame, numerical_cols: List[str],
                                    n_components: int = 5, figsize: Tuple[int, int] = (14, 8)):
        """
        Plot PCA component loadings to understand feature importance.

        Args:
            data: Dataset
            numerical_cols: Numerical features
            n_components: Number of PCA components to analyze
            figsize: Figure size
        """
        X = data[numerical_cols].fillna(0).values

        pca = PCA(n_components=n_components)
        pca.fit(X)

        # Plot explained variance
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Explained variance ratio
        ax1.bar(range(1, n_components + 1), pca.explained_variance_ratio_)
        ax1.set_xlabel('Principal Component', fontsize=12)
        ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
        ax1.set_title('PCA Explained Variance', fontsize=13, fontweight='bold')
        ax1.set_xticks(range(1, n_components + 1))

        # Cumulative explained variance
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        ax2.plot(range(1, n_components + 1), cumsum, marker='o', linewidth=2)
        ax2.set_xlabel('Number of Components', fontsize=12)
        ax2.set_ylabel('Cumulative Explained Variance', fontsize=12)
        ax2.set_title('Cumulative Explained Variance', fontsize=13, fontweight='bold')
        ax2.set_xticks(range(1, n_components + 1))
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Component loadings heatmap
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=numerical_cols
        )

        plt.figure(figsize=(10, max(8, len(numerical_cols) * 0.3)))
        sns.heatmap(loadings, cmap='coolwarm', center=0, annot=False,
                   cbar_kws={'label': 'Loading'})
        plt.title('PCA Component Loadings', fontsize=14, fontweight='bold')
        plt.xlabel('Principal Component', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        plt.show()

    def plot_metric_comparison_bars(self, clustering_comparison: Dict,
                                   figsize: Tuple[int, int] = (12, 5)):
        """
        Bar plot comparing clustering metrics.

        Args:
            clustering_comparison: Results from compare_clustering()
            figsize: Figure size
        """
        orig = clustering_comparison['original']
        aug = clustering_comparison['augmented']

        metrics = ['Silhouette Score', 'Davies-Bouldin Score', 'Number of Clusters']
        original_vals = [orig['silhouette_score'], orig['davies_bouldin_score'], orig['n_clusters']]
        augmented_vals = [aug['silhouette_score'], aug['davies_bouldin_score'], aug['n_clusters']]

        x = np.arange(len(metrics))
        width = 0.35

        fig, ax = plt.subplots(figsize=figsize)
        bars1 = ax.bar(x - width/2, original_vals, width, label='Original', color='steelblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, augmented_vals, width, label='Augmented', color='coral', alpha=0.8)

        ax.set_ylabel('Value', fontsize=12)
        ax.set_title('Clustering Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.show()

    def plot_cluster_size_distribution(self, clustering_comparison: Dict,
                                      figsize: Tuple[int, int] = (14, 5)):
        """
        Visualize cluster size distributions before/after augmentation.

        Args:
            clustering_comparison: Results from compare_clustering()
            figsize: Figure size
        """
        orig_dist = clustering_comparison['original']['cluster_distribution']
        aug_dist = clustering_comparison['augmented']['cluster_distribution']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Original
        clusters = list(orig_dist.keys())
        sizes = list(orig_dist.values())
        colors = plt.cm.Set3(range(len(clusters)))
        ax1.bar(clusters, sizes, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_xlabel('Cluster ID', fontsize=12)
        ax1.set_ylabel('Number of Brands', fontsize=12)
        ax1.set_title(f'Original Data: Cluster Sizes\n(Total: {sum(sizes)} brands)', fontsize=13, fontweight='bold')
        for i, (cluster, size) in enumerate(zip(clusters, sizes)):
            ax1.text(cluster, size, str(size), ha='center', va='bottom', fontweight='bold')

        # Augmented
        clusters = list(aug_dist.keys())
        sizes = list(aug_dist.values())
        colors = plt.cm.Set3(range(len(clusters)))
        ax2.bar(clusters, sizes, color=colors, alpha=0.8, edgecolor='black')
        ax2.set_xlabel('Cluster ID', fontsize=12)
        ax2.set_ylabel('Number of Brands', fontsize=12)
        ax2.set_title(f'Augmented Data: Cluster Sizes\n(Total: {sum(sizes)} brands)', fontsize=13, fontweight='bold')
        for i, (cluster, size) in enumerate(zip(clusters, sizes)):
            ax2.text(cluster, size, str(size), ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.show()

    def plot_comprehensive_evaluation(self, original_data: pd.DataFrame,
                                      synthetic_data: pd.DataFrame,
                                      augmented_data: pd.DataFrame,
                                      numerical_cols: List[str],
                                      clustering_comparison: Dict):
        """
        Generate a comprehensive multi-panel evaluation figure.

        Args:
            original_data: Original dataset
            synthetic_data: Synthetic dataset
            augmented_data: Combined dataset
            numerical_cols: Numerical columns
            clustering_comparison: Clustering comparison results
        """
        print("Generating comprehensive evaluation visualization...")

        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. PCA comparison (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        X_orig = original_data[numerical_cols].fillna(0).values
        X_synth = synthetic_data[numerical_cols].fillna(0).values
        pca = PCA(n_components=2)
        X_orig_pca = pca.fit_transform(X_orig)
        X_synth_pca = pca.transform(X_synth)
        ax1.scatter(X_orig_pca[:, 0], X_orig_pca[:, 1], alpha=0.5, label='Original', s=20, c='steelblue')
        ax1.scatter(X_synth_pca[:, 0], X_synth_pca[:, 1], alpha=0.5, label='Synthetic', s=20, c='coral')
        ax1.set_title('PCA: Original vs Synthetic', fontweight='bold')
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax1.legend()

        # 2. Cluster size comparison (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        orig_sizes = list(clustering_comparison['original']['cluster_distribution'].values())
        aug_sizes = list(clustering_comparison['augmented']['cluster_distribution'].values())
        x = np.arange(max(len(orig_sizes), len(aug_sizes)))
        width = 0.35
        orig_sizes_padded = orig_sizes + [0] * (len(x) - len(orig_sizes))
        aug_sizes_padded = aug_sizes + [0] * (len(x) - len(aug_sizes))
        ax2.bar(x - width/2, orig_sizes_padded, width, label='Original', color='steelblue', alpha=0.8)
        ax2.bar(x + width/2, aug_sizes_padded, width, label='Augmented', color='coral', alpha=0.8)
        ax2.set_title('Cluster Size Comparison', fontweight='bold')
        ax2.set_xlabel('Cluster ID')
        ax2.set_ylabel('Count')
        ax2.legend()

        # 3. Metrics comparison (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        metrics = ['Silhouette', 'Davies-Bouldin']
        orig_metrics = [clustering_comparison['original']['silhouette_score'],
                       clustering_comparison['original']['davies_bouldin_score']]
        aug_metrics = [clustering_comparison['augmented']['silhouette_score'],
                      clustering_comparison['augmented']['davies_bouldin_score']]
        x_pos = np.arange(len(metrics))
        ax3.bar(x_pos - 0.2, orig_metrics, 0.4, label='Original', color='steelblue', alpha=0.8)
        ax3.bar(x_pos + 0.2, aug_metrics, 0.4, label='Augmented', color='coral', alpha=0.8)
        ax3.set_title('Quality Metrics', fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(metrics, rotation=15, ha='right')
        ax3.set_ylabel('Score')
        ax3.legend()

        # 4-6. Distribution comparisons (middle row)
        sample_features = numerical_cols[:3]
        for idx, feature in enumerate(sample_features):
            ax = fig.add_subplot(gs[1, idx])
            if feature in original_data.columns and feature in synthetic_data.columns:
                ax.hist(original_data[feature].dropna(), bins=30, alpha=0.5,
                       label='Original', density=True, color='steelblue')
                ax.hist(synthetic_data[feature].dropna(), bins=30, alpha=0.5,
                       label='Synthetic', density=True, color='coral')
                ax.set_title(f'{feature[:20]}...', fontweight='bold', fontsize=10)
                ax.set_ylabel('Density')
                ax.legend(fontsize=8)

        # 7. Sample counts (bottom left)
        ax7 = fig.add_subplot(gs[2, 0])
        categories = ['Original', 'Synthetic', 'Total']
        counts = [len(original_data), len(synthetic_data), len(augmented_data)]
        colors_bar = ['steelblue', 'coral', 'green']
        ax7.bar(categories, counts, color=colors_bar, alpha=0.7)
        ax7.set_title('Dataset Sizes', fontweight='bold')
        ax7.set_ylabel('Number of Brands')
        for i, count in enumerate(counts):
            ax7.text(i, count, str(count), ha='center', va='bottom', fontweight='bold')

        # 8. Improvements (bottom middle)
        ax8 = fig.add_subplot(gs[2, 1])
        improvements = ['Silhouette\nImprovement', 'DB\nImprovement']
        values = [clustering_comparison['silhouette_improvement'],
                 clustering_comparison['davies_bouldin_improvement']]
        colors_imp = ['green' if v > 0 else 'red' for v in values]
        ax8.bar(improvements, values, color=colors_imp, alpha=0.7)
        ax8.set_title('Clustering Improvements', fontweight='bold')
        ax8.set_ylabel('Change')
        ax8.axhline(y=0, color='black', linestyle='--', linewidth=1)
        for i, val in enumerate(values):
            ax8.text(i, val, f'{val:+.3f}', ha='center',
                    va='bottom' if val > 0 else 'top', fontweight='bold')

        # 9. Text summary (bottom right)
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        summary_text = f"""
        EVALUATION SUMMARY

        Original Brands: {len(original_data)}
        Synthetic Brands: {len(synthetic_data)}
        Augmentation: +{len(synthetic_data)/len(original_data)*100:.1f}%

        Silhouette Score:
          Before: {clustering_comparison['original']['silhouette_score']:.4f}
          After:  {clustering_comparison['augmented']['silhouette_score']:.4f}
          Change: {clustering_comparison['silhouette_improvement']:+.4f}

        Davies-Bouldin:
          Before: {clustering_comparison['original']['davies_bouldin_score']:.4f}
          After:  {clustering_comparison['augmented']['davies_bouldin_score']:.4f}
          Change: {clustering_comparison['davies_bouldin_improvement']:+.4f}
        """
        ax9.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.suptitle('Comprehensive Synthetic Data Evaluation', fontsize=16, fontweight='bold', y=0.995)
        plt.show()
