### 5.9 Augmented Data Clustering Analysis

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Cluster the augmented data (real + synthetic combined)
print("="*70)
print("AUGMENTED DATA CLUSTERING ANALYSIS")
print("="*70)

# Prepare augmented data with labels
augmented_with_labels = augmented_numerical.copy()
augmented_with_labels['source'] = ['Real'] * len(train_df) + ['Synthetic'] * len(synthetic_features)

# Standardize for clustering
X_augmented = scaler.fit_transform(augmented_numerical.fillna(0))

# Perform clustering
n_clusters = 5
clustering = AgglomerativeClustering(n_clusters=n_clusters)
cluster_labels = clustering.fit_predict(X_augmented)

# Calculate metrics
silhouette = silhouette_score(X_augmented, cluster_labels)
davies_bouldin = davies_bouldin_score(X_augmented, cluster_labels)

print(f"\nClustering Results (n_clusters={n_clusters}):")
print(f"  Silhouette Score: {silhouette:.4f} (higher is better, range: -1 to 1)")
print(f"  Davies-Bouldin Index: {davies_bouldin:.4f} (lower is better)")

# Analyze cluster composition
augmented_with_labels['cluster'] = cluster_labels

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Cluster composition (Real vs Synthetic distribution per cluster)
ax1 = axes[0]
cluster_composition = pd.crosstab(augmented_with_labels['cluster'], augmented_with_labels['source'], normalize='index') * 100
cluster_composition.plot(kind='bar', ax=ax1, color=['steelblue', 'coral'], edgecolor='black', alpha=0.8)
ax1.set_xlabel('Cluster', fontsize=12)
ax1.set_ylabel('Percentage (%)', fontsize=12)
ax1.set_title('Cluster Composition: Real vs Synthetic\n(Balanced = Good Integration)', fontsize=14, fontweight='bold')
ax1.legend(title='Source', fontsize=10)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)

# Add cluster sizes as text
cluster_sizes = augmented_with_labels['cluster'].value_counts().sort_index()
for i, (idx, size) in enumerate(cluster_sizes.items()):
    ax1.annotate(f'n={size}', xy=(i, 105), ha='center', fontsize=9, fontweight='bold')

# PCA visualization with clusters
ax2 = axes[1]
X_augmented_pca = pca.fit_transform(X_augmented)
scatter = ax2.scatter(X_augmented_pca[:, 0], X_augmented_pca[:, 1],
                      c=cluster_labels, cmap='Set2', alpha=0.6, s=30, edgecolor='white', linewidth=0.5)

# Mark synthetic points with different marker
synthetic_mask = np.array([False] * len(train_df) + [True] * len(synthetic_features))
ax2.scatter(X_augmented_pca[synthetic_mask, 0], X_augmented_pca[synthetic_mask, 1],
           c='none', edgecolor='red', s=50, linewidth=1.5, marker='o', label='Synthetic', alpha=0.5)

ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
ax2.set_title(f'Augmented Data Clusters (PCA View)\nSilhouette: {silhouette:.3f}', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right', fontsize=10)

plt.colorbar(scatter, ax=ax2, label='Cluster')
plt.suptitle('Augmented Data (Real + Synthetic) Clustering', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'augmented_clustering.png'), dpi=150, bbox_inches='tight')
plt.show()

# Print cluster composition summary
print(f"\nCluster Composition Summary:")
print(cluster_composition.round(1).to_string())
print(f"\nInterpretation: If synthetic data integrates well, each cluster should have")
print(f"roughly proportional representation ({100*len(synthetic_features)/len(augmented_numerical):.1f}% synthetic expected)")