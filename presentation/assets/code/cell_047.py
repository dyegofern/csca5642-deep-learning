### 5.7 Feature-wise Statistics Comparison

# Compute statistics for real and synthetic data
stats_comparison = []

for col in eval_numerical_cols:
    real_col = train_df[col].dropna()
    synth_col = synthetic_features[col].dropna()

    stats_comparison.append({
        'Feature': col,
        'Real Mean': real_col.mean(),
        'Synth Mean': synth_col.mean(),
        'Mean Diff %': abs(real_col.mean() - synth_col.mean()) / (abs(real_col.mean()) + 1e-10) * 100,
        'Real Std': real_col.std(),
        'Synth Std': synth_col.std(),
        'Std Diff %': abs(real_col.std() - synth_col.std()) / (abs(real_col.std()) + 1e-10) * 100,
        'Real Min': real_col.min(),
        'Synth Min': synth_col.min(),
        'Real Max': real_col.max(),
        'Synth Max': synth_col.max(),
        'KS Stat': ks_results[col]['statistic']
    })

stats_df = pd.DataFrame(stats_comparison)

# Display summary table
print("="*80)
print("FEATURE-WISE STATISTICS COMPARISON")
print("="*80)
display(stats_df[['Feature', 'Real Mean', 'Synth Mean', 'Mean Diff %', 'Real Std', 'Synth Std', 'Std Diff %', 'KS Stat']].round(3))

# Visualize mean and std differences
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Mean difference
ax1 = axes[0]
sorted_by_mean = stats_df.sort_values('Mean Diff %', ascending=False)
colors_mean = ['#e74c3c' if x > 50 else '#f39c12' if x > 20 else '#27ae60' for x in sorted_by_mean['Mean Diff %']]
ax1.barh(range(len(sorted_by_mean)), sorted_by_mean['Mean Diff %'], color=colors_mean, edgecolor='black', alpha=0.8)
ax1.set_yticks(range(len(sorted_by_mean)))
ax1.set_yticklabels(sorted_by_mean['Feature'], fontsize=9)
ax1.set_xlabel('Mean Difference (%)', fontsize=12)
ax1.set_title('Mean Value Difference\n(Lower is Better)', fontsize=14, fontweight='bold')
ax1.axvline(x=20, color='orange', linestyle='--', linewidth=2, alpha=0.7)
ax1.axvline(x=50, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax1.invert_yaxis()

# Std difference
ax2 = axes[1]
sorted_by_std = stats_df.sort_values('Std Diff %', ascending=False)
colors_std = ['#e74c3c' if x > 50 else '#f39c12' if x > 20 else '#27ae60' for x in sorted_by_std['Std Diff %']]
ax2.barh(range(len(sorted_by_std)), sorted_by_std['Std Diff %'], color=colors_std, edgecolor='black', alpha=0.8)
ax2.set_yticks(range(len(sorted_by_std)))
ax2.set_yticklabels(sorted_by_std['Feature'], fontsize=9)
ax2.set_xlabel('Std Deviation Difference (%)', fontsize=12)
ax2.set_title('Standard Deviation Difference\n(Lower is Better)', fontsize=14, fontweight='bold')
ax2.axvline(x=20, color='orange', linestyle='--', linewidth=2, alpha=0.7)
ax2.axvline(x=50, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'statistics_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()