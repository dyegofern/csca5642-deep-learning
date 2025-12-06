# Decode categorical features back to original values
print("Decoding categorical features...")
synthetic_decoded = processor.decode_categorical(synthetic_features)
print(f"Decoded {len(synthetic_decoded)} synthetic brands")