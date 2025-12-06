training_times = tabular_ensemble.train(
    data=train_df,
    discrete_columns=discrete_cols,
    binary_columns=binary_cols
)
# Save models
tabular_ensemble.save_models(os.path.join(MODEL_DIR, 'tabular_ensemble'))
# Optimize weights based on quality
optimized_weights = tabular_ensemble.optimize_weights(train_df, n_eval_samples=1000)
