# Generate synthetic tabular features using ensemble
print("Generating synthetic features with ensemble...")
synthetic_features, failed_companies = tabular_ensemble.generate_stratified(
    company_distribution=generation_targets,
    verbose=True
)