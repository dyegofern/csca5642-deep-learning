# Generate brand names using LLM ensemble
print("\nGenerating brand names with LLM ensemble...")
llm_generator.reset_uniqueness_tracker()

synthetic_with_names = llm_generator.generate_for_dataframe(
    synthetic_df=synthetic_decoded,
    temperature=DIVERSITY_TEMPERATURE,
    verbose=True
)

print(f"\nFinal synthetic dataset: {len(synthetic_with_names)} brands")