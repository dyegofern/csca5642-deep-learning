# Optionally optimize weights based on quality
optimized_weights = tabular_ensemble.optimize_weights(train_df, n_eval_samples=1000)
print(f"Optimized weights: {optimized_weights}")