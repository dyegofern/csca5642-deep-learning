# Generate brand names using LLM ensemble
synthetic_with_names = llm_generator.generate_for_dataframe(
    synthetic_df=synthetic_decoded,
    temperature=DIVERSITY_TEMPERATURE,
    verbose=True
)