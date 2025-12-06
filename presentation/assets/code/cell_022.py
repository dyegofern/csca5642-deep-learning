# Initialize Ensemble Synthesizer
tabular_ensemble = EnsembleSynthesizer(
    ctgan_epochs=CTGAN_EPOCHS,
    ctgan_batch_size=BATCH_SIZE,
    tvae_epochs=TVAE_EPOCHS,
    tvae_batch_size=BATCH_SIZE,
    gc_default_distribution='beta',
    weights=ENSEMBLE_WEIGHTS,
    verbose=True,
    cuda=True
)

print("Tabular Ensemble initialized with:")
print(f"  - CTGAN (epochs={CTGAN_EPOCHS})")
print(f"  - TVAE (epochs={TVAE_EPOCHS})")
print(f"  - Gaussian Copula (distribution=beta)")