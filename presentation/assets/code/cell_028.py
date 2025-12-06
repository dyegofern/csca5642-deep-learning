# Initialize LLM Ensemble Generator
llm_generator = BrandNameGeneratorV2(
    models=LLM_MODELS,
    memory_efficient=True,
    verbose=True
)

print(f"LLM Ensemble initialized with models: {LLM_MODELS}")