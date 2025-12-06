# Calculate generation targets
generation_targets = calculate_generation_targets(
    data=train_df,
    company_column='company_name',
    min_brands_per_company=MIN_BRANDS_PER_COMPANY
)
generation_targets = dict(list(generation_targets.items())[:150])