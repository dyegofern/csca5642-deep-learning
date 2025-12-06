# Compare individual model quality
print("Evaluating individual model quality...")
comparison_df = tabular_ensemble.compare_all_models(train_df, n_samples=1000)
print("\nModel Comparison:")
display(comparison_df)