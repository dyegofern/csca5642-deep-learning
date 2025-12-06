### 5.8 Synthetic Data Quality Scorecard & Radar Chart

# Calculate quality metrics
ks_pass_rate = passes / len(ks_results) * 100
mean_ks_stat = np.mean([v['statistic'] for v in ks_results.values()])
mean_mean_diff = stats_df['Mean Diff %'].mean()
mean_std_diff = stats_df['Std Diff %'].mean()

# Normalize metrics to 0-100 scale (higher is better)
distribution_score = max(0, 100 - mean_ks_stat * 200)  # KS of 0 = 100, KS of 0.5 = 0
correlation_score = max(0, 100 - mean_corr_diff * 200)  # Diff of 0 = 100, Diff of 0.5 = 0
mean_preservation = max(0, 100 - mean_mean_diff)  # Lower diff = higher score
variance_preservation = max(0, 100 - mean_std_diff)  # Lower diff = higher score
coverage_score = ks_pass_rate  # Percentage of features passing KS test

# Overall score (weighted average)
overall_score = (distribution_score * 0.3 + correlation_score * 0.2 +
                 mean_preservation * 0.2 + variance_preservation * 0.2 + coverage_score * 0.1)

# Create figure with scorecard and radar chart
fig = plt.figure(figsize=(18, 8))

# Left side: Scorecard
ax1 = fig.add_subplot(121)
ax1.axis('off')

# Create scorecard table
scorecard_data = [
    ['Metric', 'Value', 'Score', 'Grade'],
    ['KS Test Pass Rate', f'{ks_pass_rate:.1f}%', f'{coverage_score:.1f}', 'A' if coverage_score >= 70 else 'B' if coverage_score >= 50 else 'C' if coverage_score >= 30 else 'D'],
    ['Mean KS Statistic', f'{mean_ks_stat:.3f}', f'{distribution_score:.1f}', 'A' if distribution_score >= 70 else 'B' if distribution_score >= 50 else 'C' if distribution_score >= 30 else 'D'],
    ['Correlation Preservation', f'{mean_corr_diff:.3f}', f'{correlation_score:.1f}', 'A' if correlation_score >= 70 else 'B' if correlation_score >= 50 else 'C' if correlation_score >= 30 else 'D'],
    ['Mean Value Preservation', f'{mean_mean_diff:.1f}%', f'{mean_preservation:.1f}', 'A' if mean_preservation >= 70 else 'B' if mean_preservation >= 50 else 'C' if mean_preservation >= 30 else 'D'],
    ['Variance Preservation', f'{mean_std_diff:.1f}%', f'{variance_preservation:.1f}', 'A' if variance_preservation >= 70 else 'B' if variance_preservation >= 50 else 'C' if variance_preservation >= 30 else 'D'],
    ['─' * 20, '─' * 10, '─' * 10, '─' * 5],
    ['OVERALL QUALITY', '', f'{overall_score:.1f}', 'A' if overall_score >= 70 else 'B' if overall_score >= 50 else 'C' if overall_score >= 30 else 'D'],
]

table = ax1.table(cellText=scorecard_data, loc='center', cellLoc='center',
                  colWidths=[0.35, 0.2, 0.15, 0.1])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2)

# Style the table
for i in range(len(scorecard_data)):
    for j in range(4):
        cell = table[(i, j)]
        if i == 0:  # Header
            cell.set_facecolor('#34495e')
            cell.set_text_props(color='white', fontweight='bold')
        elif i == len(scorecard_data) - 1:  # Overall row
            cell.set_facecolor('#2ecc71' if overall_score >= 70 else '#f39c12' if overall_score >= 50 else '#e74c3c')
            cell.set_text_props(fontweight='bold')
        elif i == len(scorecard_data) - 2:  # Separator
            cell.set_facecolor('#ecf0f1')
        elif j == 3:  # Grade column
            grade = scorecard_data[i][3]
            if grade == 'A':
                cell.set_facecolor('#27ae60')
            elif grade == 'B':
                cell.set_facecolor('#f39c12')
            elif grade == 'C':
                cell.set_facecolor('#e67e22')
            else:
                cell.set_facecolor('#e74c3c')
            cell.set_text_props(color='white', fontweight='bold')

ax1.set_title('Synthetic Data Quality Scorecard', fontsize=16, fontweight='bold', pad=20)

# Right side: Radar chart
ax2 = fig.add_subplot(122, projection='polar')

# Radar chart data
categories = ['Distribution\nMatching', 'Correlation\nPreservation', 'Mean\nPreservation',
              'Variance\nPreservation', 'KS Test\nPass Rate']
values = [distribution_score, correlation_score, mean_preservation, variance_preservation, coverage_score]
values += values[:1]  # Close the polygon

angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

ax2.plot(angles, values, 'o-', linewidth=2, color='#3498db', markersize=8)
ax2.fill(angles, values, alpha=0.25, color='#3498db')
ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(categories, fontsize=10)
ax2.set_ylim(0, 100)
ax2.set_yticks([20, 40, 60, 80, 100])
ax2.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=9)
ax2.set_title('Quality Metrics Radar Chart\n(Higher is Better)', fontsize=14, fontweight='bold', pad=20)

# Add reference circles
for val in [30, 50, 70]:
    circle = plt.Circle((0, 0), val, transform=ax2.transData._b, fill=False,
                         linestyle='--', alpha=0.3, color='gray')

plt.suptitle('SYNTHETIC DATA QUALITY ASSESSMENT', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'quality_scorecard.png'), dpi=150, bbox_inches='tight')
plt.show()

# Print summary
print("\n" + "="*70)
print("QUALITY ASSESSMENT SUMMARY")
print("="*70)
print(f"Overall Quality Score: {overall_score:.1f}/100 ({'Excellent' if overall_score >= 70 else 'Good' if overall_score >= 50 else 'Fair' if overall_score >= 30 else 'Poor'})")
print(f"Distribution Matching: {distribution_score:.1f}/100")
print(f"Correlation Preservation: {correlation_score:.1f}/100")
print(f"Statistical Fidelity: {(mean_preservation + variance_preservation)/2:.1f}/100")
print("="*70)