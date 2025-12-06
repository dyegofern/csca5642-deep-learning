# Test LLM generation
print("Testing LLM ensemble generation...")
llm_generator.prepare_model()

test_companies = [
    ("PepsiCo", "Non-Alcoholic Beverages"),
    ("Nestle", "Processed Foods"),
    ("Mars, Incorporated", "Processed Foods")
]

for company, industry in test_companies:
    names = llm_generator.generate_brand_names(company, industry, n_names=3)
    print(f"\n{company} ({industry}): {names}")