# Check for saved hyperparameters
if os.path.exists(HYPERPARAMS_PATH):
    print("Found saved hyperparameters!")
    best_hyperparams = HyperparameterTunerV2.load(HYPERPARAMS_PATH)

    print(f"Previous best score: {best_hyperparams.get('best_score', 'N/A')}")
    print(f"Trials completed: {best_hyperparams.get('n_trials', 'N/A')}")
    print(f"Timestamp: {best_hyperparams.get('tuning_timestamp', 'N/A')}")

    # Update configuration with loaded hyperparameters
    CTGAN_EPOCHS = best_hyperparams.get('ctgan_epochs', CTGAN_EPOCHS)
    TVAE_EPOCHS = best_hyperparams.get('tvae_epochs', TVAE_EPOCHS)
    BATCH_SIZE = best_hyperparams.get('batch_size', BATCH_SIZE)
    if 'ensemble_weights' in best_hyperparams:
        ENSEMBLE_WEIGHTS = best_hyperparams['ensemble_weights']

    print(f"\nUsing optimized configuration:")
    print(f"  CTGAN_EPOCHS: {CTGAN_EPOCHS}")
    print(f"  TVAE_EPOCHS: {TVAE_EPOCHS}")
    print(f"  BATCH_SIZE: {BATCH_SIZE}")
    print(f"  ENSEMBLE_WEIGHTS: {ENSEMBLE_WEIGHTS}")
else:
    print("No saved hyperparameters found.")
    if RUN_HYPERPARAMETER_TUNING:
        print("Hyperparameter tuning will run after data preparation.")
    else:
        print("Using default configuration. Set RUN_HYPERPARAMETER_TUNING=True to optimize.")