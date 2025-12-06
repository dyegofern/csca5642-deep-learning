if FROM_PRETRAINED:
    # Load pre-trained models
    print("Loading pre-trained LLM ensemble...")
    llm_generator.load_model(os.path.join(MODEL_DIR, 'llm_ensemble'))
else:
    # Fine-tune all models
    print(f"Fine-tuning LLM ensemble (epochs={LLM_EPOCHS})...")
    print("This will train each model sequentially to save memory.")

    llm_generator.fine_tune(
        brands_df=brands_df,
        epochs=LLM_EPOCHS,
        output_dir=os.path.join(MODEL_DIR, 'llm_ensemble')
    )

    # Save ensemble config
    llm_generator.save_model(os.path.join(MODEL_DIR, 'llm_ensemble'))