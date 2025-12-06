# Save synthetic data
synthetic_path = os.path.join(OUTPUT_DIR, 'synthetic_brands_v2.csv')
synthetic_with_names.to_csv(synthetic_path, index=False)
print(f"Synthetic data saved to {synthetic_path}")

# Create augmented dataset
original_decoded = processor.decode_categorical(train_df)
augmented_df = pd.concat([original_decoded, synthetic_with_names], ignore_index=True)

augmented_path = os.path.join(OUTPUT_DIR, 'augmented_brands_v2.csv')
augmented_df.to_csv(augmented_path, index=False)
print(f"Augmented data saved to {augmented_path}")
print(f"Total augmented size: {len(augmented_df)} brands")