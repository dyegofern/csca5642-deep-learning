llm_generator.fine_tune(
    brands_df=brands_df,
    epochs=LLM_EPOCHS,
    output_dir=os.path.join(MODEL_DIR, 'llm_ensemble')
)
# Save ensemble config
llm_generator.save_model(os.path.join(MODEL_DIR, 'llm_ensemble'))