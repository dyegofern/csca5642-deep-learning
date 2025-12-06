# Execute hyperparameter tuning if enabled and no saved params exist
if RUN_HYPERPARAMETER_TUNING and best_hyperparams is None:
    print("Initializing hyperparameter tuner...")

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

    # Plot optimization history
    tuner.plot_optimization_history(
        save_path=os.path.join(OUTPUT_DIR, 'hyperparameter_tuning_history.png')
    )

    # Update global configuration
    CTGAN_EPOCHS = best_hyperparams['ctgan_epochs']
    TVAE_EPOCHS = best_hyperparams['tvae_epochs']
    BATCH_SIZE = best_hyperparams['batch_size']
    ENSEMBLE_WEIGHTS = best_hyperparams['ensemble_weights']

    print(f"\nUpdated configuration with optimized hyperparameters:")
    print(f"  CTGAN_EPOCHS: {CTGAN_EPOCHS}")
    print(f"  TVAE_EPOCHS: {TVAE_EPOCHS}")
    print(f"  BATCH_SIZE: {BATCH_SIZE}")
    print(f"  ENSEMBLE_WEIGHTS: {ENSEMBLE_WEIGHTS}")
else:
    if best_hyperparams is not None:
        print("Using previously loaded hyperparameters.")
    else:
        print("Using default hyperparameters. Set RUN_HYPERPARAMETER_TUNING=True to optimize.")