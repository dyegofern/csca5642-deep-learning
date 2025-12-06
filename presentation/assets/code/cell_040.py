# Initialize evaluator and prepare data
evaluator = BrandDataEvaluator()

# Get numerical columns for evaluation
eval_numerical_cols = [col for col in numerical_cols if col in synthetic_features.columns and col in train_df.columns]

# Prepare augmented numerical data for later use
augmented_numerical = pd.concat([train_df[eval_numerical_cols], synthetic_features[eval_numerical_cols]], ignore_index=True)

print(f"Evaluation columns: {len(eval_numerical_cols)} numerical features")
print(f"Real data samples: {len(train_df)}")
print(f"Synthetic data samples: {len(synthetic_features)}")
print(f"Augmented data samples: {len(augmented_numerical)}")