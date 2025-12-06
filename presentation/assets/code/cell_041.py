### 5.1 Statistical Distribution Comparison (KS Test)

from scipy import stats

# Compute KS statistics for all features
print("="*70)
print("KOLMOGOROV-SMIRNOV TEST RESULTS")
print("="*70)
print(f"{'Feature':<35} {'KS Stat':>10} {'P-Value':>12} {'Result':>10}")
print("-"*70)

ks_results = {}
for col in eval_numerical_cols:
    if col in train_df.columns and col in synthetic_features.columns:
        stat, pvalue = stats.ks_2samp(
            train_df[col].dropna(),
            synthetic_features[col].dropna()
        )
        ks_results[col] = {'statistic': stat, 'pvalue': pvalue}
        result = "PASS" if pvalue > 0.05 else "FAIL"
        print(f"{col:<35} {stat:>10.4f} {pvalue:>12.4f} {result:>10}")

passes = sum(1 for v in ks_results.values() if v['pvalue'] > 0.05)
print("-"*70)
print(f"SUMMARY: {passes}/{len(ks_results)} features pass (p > 0.05) = {100*passes/len(ks_results):.1f}% pass rate")
print("="*70)