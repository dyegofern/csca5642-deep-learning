# Add diversity noise if enabled
if ADD_DIVERSITY_NOISE:
    print("Adding diversity noise to numerical features...")
    synthetic_features = tabular_ensemble.add_diversity_noise(
        synthetic_features,
        noise_level=0.07
    )