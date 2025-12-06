# Configuration for V2
FROM_PRETRAINED = False  # Set to True to load pre-trained models

# Tabular Ensemble Config
CTGAN_EPOCHS = 300
TVAE_EPOCHS = 300
BATCH_SIZE = 500
ENSEMBLE_WEIGHTS = {
    'ctgan': 0.40,
    'tvae': 0.35,
    'gaussian_copula': 0.25
}

# LLM Ensemble Config
LLM_MODELS = ['gpt2-medium', 'flan-t5-base']  # Can add 'phi-2', 'tinyllama' if memory allows
LLM_EPOCHS = 3

# Generation Config
MIN_BRANDS_PER_COMPANY = 10
DIVERSITY_TEMPERATURE = 0.7
ADD_DIVERSITY_NOISE = True

print("Configuration:")
print(f"  FROM_PRETRAINED: {FROM_PRETRAINED}")
print(f"  CTGAN_EPOCHS: {CTGAN_EPOCHS}")
print(f"  TVAE_EPOCHS: {TVAE_EPOCHS}")
print(f"  LLM_MODELS: {LLM_MODELS}")
print(f"  ENSEMBLE_WEIGHTS: {ENSEMBLE_WEIGHTS}")