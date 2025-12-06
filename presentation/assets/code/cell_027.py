# Prepare brand name training data
brands_df = processor.df[['brand_name', 'company_name', 'industry_name']].dropna()
print(f"Brand name training data: {len(brands_df)} examples")
brands_df.head()