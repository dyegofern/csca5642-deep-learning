### 5.2 KS Statistics Visualization

# Create a bar chart of KS statistics
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Sort features by KS statistic
sorted_features = sorted(ks_results.items(), key=lambda x: x[1]['statistic'], reverse=True)
features = [f[0] for f in sorted_features]
ks_stats = [f[1]['statistic'] for f in sorted_features]
colors = ['#e74c3c' if f[1]['pvalue'] <= 0.05 else '#27ae60' for f in sorted_features]

# Bar chart of KS statistics
ax1 = axes[0]
bars = ax1.barh(range(len(features)), ks_stats, color=colors, edgecolor='black', alpha=0.8)
ax1.set_yticks(range(len(features)))
ax1.set_yticklabels(features, fontsize=9)
ax1.set_xlabel('KS Statistic', fontsize=12)
ax1.set_title('Distribution Similarity (KS Test)\nLower is Better', fontsize=14, fontweight='bold')
ax1.axvline(x=0.1, color='orange', linestyle='--', linewidth=2, label='Good threshold (0.1)')
ax1.axvline(x=0.2, color='red', linestyle='--', linewidth=2, label='Poor threshold (0.2)')
ax1.legend(loc='lower right')
ax1.invert_yaxis()

# Add pass/fail legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#27ae60', label='Pass (p > 0.05)'),
                   Patch(facecolor='#e74c3c', label='Fail (p ≤ 0.05)')]
ax1.legend(handles=legend_elements, loc='lower right')

# Pie chart of pass/fail
ax2 = axes[1]
fail_count = len(ks_results) - passes
sizes = [passes, fail_count]
labels = [f'Pass\n({passes})', f'Fail\n({fail_count})']
colors_pie = ['#27ae60', '#e74c3c']
explode = (0.05, 0)
wedges, texts, autotexts = ax2.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
                                    autopct='%1.1f%%', shadow=True, startangle=90,
                                    textprops={'fontsize': 12, 'fontweight': 'bold'})
ax2.set_title('KS Test Results Summary', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'ks_test_results.png'), dpi=150, bbox_inches='tight')
plt.show()