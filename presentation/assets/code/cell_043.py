### 5.3 Distribution Comparison - Histograms with KDE

# Select top features for visualization (mix of good and bad performers)
best_features = [f[0] for f in sorted_features[-4:]]  # Best 4 (lowest KS)
worst_features = [f[0] for f in sorted_features[:4]]  # Worst 4 (highest KS)
viz_features = worst_features + best_features

n_features = len(viz_features)
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for idx, feature in enumerate(viz_features):
    ax = axes[idx]

    # Get data
    real_data = train_df[feature].dropna()
    synth_data = synthetic_features[feature].dropna()

    # Plot histograms with KDE
    ax.hist(real_data, bins=30, alpha=0.5, label='Real', color='steelblue', density=True, edgecolor='black')
    ax.hist(synth_data, bins=30, alpha=0.5, label='Synthetic', color='coral', density=True, edgecolor='black')

    # Add KDE curves
    try:
        real_data.plot.kde(ax=ax, color='steelblue', linewidth=2, linestyle='-')
        synth_data.plot.kde(ax=ax, color='coral', linewidth=2, linestyle='-')
    except:
        pass  # Skip KDE if it fails

    # Get KS stat for this feature
    ks_stat = ks_results[feature]['statistic']
    pvalue = ks_results[feature]['pvalue']
    result = "PASS" if pvalue > 0.05 else "FAIL"

    ax.set_title(f'{feature}\nKS={ks_stat:.3f} ({result})', fontsize=11, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Density')
    ax.legend(fontsize=9)

    # Color code title based on result
    if result == "FAIL":
        ax.title.set_color('#c0392b')
    else:
        ax.title.set_color('#27ae60')

plt.suptitle('Distribution Comparison: Real vs Synthetic Data\nTop Row: Worst Performers | Bottom Row: Best Performers',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'distribution_comparison_v2.png'), dpi=150, bbox_inches='tight')
plt.show()