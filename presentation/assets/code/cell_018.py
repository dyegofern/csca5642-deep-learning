# Prepare for GAN training
train_df, val_df = processor.prepare_for_gan(test_size=0.2)

print(f"\nTraining set: {len(train_df)} brands")
print(f"Validation set: {len(val_df)} brands")

# Get column types
discrete_cols = processor.categorical_features
binary_cols = [col for col in train_df.columns if train_df[col].nunique() == 2 and set(train_df[col].unique()).issubset({0, 1})]
numerical_cols = [col for col in train_df.columns if col not in discrete_cols and col not in binary_cols]

print(f"\nColumn types:")
print(f"  Numerical: {len(numerical_cols)}")
print(f"  Categorical: {len(discrete_cols)}")
print(f"  Binary: {len(binary_cols)}")