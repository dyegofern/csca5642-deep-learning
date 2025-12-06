### 5.4 Correlation Structure Preservation

# Compute correlation matrices
real_corr = train_df[eval_numerical_cols].corr()
synth_corr = synthetic_features[eval_numerical_cols].corr()
corr_diff = np.abs(real_corr - synth_corr)

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Real data correlations
sns.heatmap(real_corr, ax=axes[0], cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            square=True, cbar_kws={'label': 'Correlation', 'shrink': 0.8},
            xticklabels=True, yticklabels=True)
axes[0].set_title('Real Data Correlations', fontsize=14, fontweight='bold')
axes[0].tick_params(axis='both', labelsize=8, rotation=45)

# Synthetic data correlations
sns.heatmap(synth_corr, ax=axes[1], cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            square=True, cbar_kws={'label': 'Correlation', 'shrink': 0.8},
            xticklabels=True, yticklabels=True)
axes[1].set_title('Synthetic Data Correlations', fontsize=14, fontweight='bold')
axes[1].tick_params(axis='both', labelsize=8, rotation=45)

# Correlation difference (absolute)
sns.heatmap(corr_diff, ax=axes[2], cmap='Reds', vmin=0, vmax=0.5,
            square=True, cbar_kws={'label': '|Difference|', 'shrink': 0.8},
            xticklabels=True, yticklabels=True)
axes[2].set_title('Absolute Correlation Difference\n(Lower is Better)', fontsize=14, fontweight='bold')
axes[2].tick_params(axis='both', labelsize=8, rotation=45)

# Add metrics annotation
mean_corr_diff = corr_diff.mean().mean()
max_corr_diff = corr_diff.max().max()

plt.suptitle(f'Correlation Structure Analysis\nMean Abs. Difference: {mean_corr_diff:.4f} | Max Difference: {max_corr_diff:.4f}',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_comparison_v2.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"\nCorrelation Preservation Metrics:")
print(f"  Mean absolute difference: {mean_corr_diff:.4f}")
print(f"  Max absolute difference: {max_corr_diff:.4f}")
print(f"  Correlation RMSE: {np.sqrt(np.mean((real_corr - synth_corr).values**2)):.4f}")