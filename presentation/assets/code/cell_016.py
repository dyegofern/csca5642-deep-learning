# Load and process data
processor = BrandDataProcessor(DATA_PATH)
raw_data = processor.load_data()
print(f"Loaded {len(raw_data)} brands with {len(raw_data.columns)} features")