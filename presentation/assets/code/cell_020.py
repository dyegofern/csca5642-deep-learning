# Create tuner instance
tuner = HyperparameterTunerV2(
    train_data=train_df,
    discrete_cols=discrete_cols,
    binary_cols=binary_cols,
    eval_sample_size=min(1000, len(train_df)),
    gen_sample_size=500,
    verbose=True
)

# Run optimization
best_hyperparams = tuner.tune(
    n_trials=N_TUNING_TRIALS,
    timeout=TUNING_TIMEOUT,
    seed=42,
    show_progress_bar=True
)

# Save the best hyperparameters
tuner.save(HYPERPARAMS_PATH)