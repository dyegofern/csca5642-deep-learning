if FROM_PRETRAINED:
    # Load pre-trained models
    print("Loading pre-trained tabular ensemble...")
    tabular_ensemble.load_models(os.path.join(MODEL_DIR, 'tabular_ensemble'))
else:
    # Train all models
    print("Training tabular ensemble (this will take ~30-60 minutes)...")
    training_times = tabular_ensemble.train(
        data=train_df,
        discrete_columns=discrete_cols,
        binary_columns=binary_cols
    )

    # Save models
    tabular_ensemble.save_models(os.path.join(MODEL_DIR, 'tabular_ensemble'))

    print(f"\nTraining times:")
    for model, time in training_times.items():
        print(f"  {model}: {time:.1f} seconds")