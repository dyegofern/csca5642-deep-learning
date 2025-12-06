### 5.5 QQ Plots - Quantile Comparison

# QQ plots for selected features
from scipy import stats as scipy_stats

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for idx, feature in enumerate(viz_features):
    ax = axes[idx]

    real_data = np.sort(train_df[feature].dropna().values)
    synth_data = np.sort(synthetic_features[feature].dropna().values)

    # Interpolate to same length for QQ plot
    n_points = min(len(real_data), len(synth_data), 100)
    real_quantiles = np.percentile(real_data, np.linspace(0, 100, n_points))
    synth_quantiles = np.percentile(synth_data, np.linspace(0, 100, n_points))

    # Plot QQ
    ax.scatter(real_quantiles, synth_quantiles, alpha=0.6, s=30, c='steelblue', edgecolor='black')

    # Add diagonal reference line
    min_val = min(real_quantiles.min(), synth_quantiles.min())
    max_val = max(real_quantiles.max(), synth_quantiles.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Match')

    # Color based on KS result
    ks_stat = ks_results[feature]['statistic']
    result = "PASS" if ks_results[feature]['pvalue'] > 0.05 else "FAIL"
    title_color = '#27ae60' if result == "PASS" else '#c0392b'

    ax.set_title(f'{feature}\nKS={ks_stat:.3f}', fontsize=11, fontweight='bold', color=title_color)
    ax.set_xlabel('Real Quantiles', fontsize=10)
    ax.set_ylabel('Synthetic Quantiles', fontsize=10)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_aspect('equal', adjustable='box')

plt.suptitle('Q-Q Plots: Real vs Synthetic Quantiles\nPoints on diagonal = Perfect distribution match',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'qq_plots.png'), dpi=150, bbox_inches='tight')
plt.show()