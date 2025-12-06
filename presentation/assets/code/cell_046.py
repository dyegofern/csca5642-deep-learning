### 5.6 PCA & t-SNE Dimensionality Reduction

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Prepare data
X_real = train_df[eval_numerical_cols].fillna(0).values
X_synth = synthetic_features[eval_numerical_cols].fillna(0).values

# Standardize
scaler = StandardScaler()
X_real_scaled = scaler.fit_transform(X_real)
X_synth_scaled = scaler.transform(X_synth)

# PCA
pca = PCA(n_components=2)
X_real_pca = pca.fit_transform(X_real_scaled)
X_synth_pca = pca.transform(X_synth_scaled)

# t-SNE (on combined data for fair comparison)
print("Computing t-SNE projection (this may take a moment)...")
X_combined = np.vstack([X_real_scaled, X_synth_scaled])
labels = np.array(['Real'] * len(X_real) + ['Synthetic'] * len(X_synth))

tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
X_combined_tsne = tsne.fit_transform(X_combined)
X_real_tsne = X_combined_tsne[:len(X_real)]
X_synth_tsne = X_combined_tsne[len(X_real):]

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# PCA plot
ax1 = axes[0]
ax1.scatter(X_real_pca[:, 0], X_real_pca[:, 1], alpha=0.5, s=30, c='steelblue', label='Real', edgecolor='white', linewidth=0.5)
ax1.scatter(X_synth_pca[:, 0], X_synth_pca[:, 1], alpha=0.5, s=30, c='coral', label='Synthetic', edgecolor='white', linewidth=0.5)
ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
ax1.set_title('PCA Projection\nReal vs Synthetic Data', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11, loc='upper right')
ax1.grid(True, alpha=0.3)

# t-SNE plot
ax2 = axes[1]
ax2.scatter(X_real_tsne[:, 0], X_real_tsne[:, 1], alpha=0.5, s=30, c='steelblue', label='Real', edgecolor='white', linewidth=0.5)
ax2.scatter(X_synth_tsne[:, 0], X_synth_tsne[:, 1], alpha=0.5, s=30, c='coral', label='Synthetic', edgecolor='white', linewidth=0.5)
ax2.set_xlabel('t-SNE Dimension 1', fontsize=12)
ax2.set_ylabel('t-SNE Dimension 2', fontsize=12)
ax2.set_title('t-SNE Projection\nReal vs Synthetic Data', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11, loc='upper right')
ax2.grid(True, alpha=0.3)

plt.suptitle('Dimensionality Reduction: Synthetic Data Should Overlap with Real Data', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'pca_tsne_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"\nPCA Explained Variance: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}")