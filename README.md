# Synthetic Brand Generation with GANs

Deep Learning Final Project - Addressing Class Imbalance in Brand Clustering using Generative Models

## Project Overview

This project tackles severe class imbalance in hierarchical clustering of ESG (Environmental, Social, and Governance) brand data by generating synthetic brands using state-of-the-art generative models. The original unsupervised learning analysis produced only 2 clusters (with one containing a single brand), indicating poor cluster balance.

### Solution: Hybrid GAN Approach

1. **CTGAN** (Conditional Tabular GAN): Generates realistic brand features (ESG metrics, demographics, business characteristics)
2. **DistilGPT2** (Fine-tuned): Generates realistic brand names conditioned on company and industry

## Dataset

- **Source**: `data/raw/brand_information.csv`
- **Size**: 3,605 brands across multiple industries
- **Features**: 80+ columns including:
  - ESG metrics (greenwashing levels, emissions, sustainability awards)
  - Demographics (age groups, income levels, lifestyle segments)
  - Business model characteristics (franchises, online sales, fleet info)
  - Financial data (revenues, market cap, R&D spend)

## Project Structure

```
final_idea_2/
├── src/
│   ├── __init__.py                 # Module initialization
│   ├── data_processor.py           # Data loading and preprocessing
│   ├── tabular_gan.py             # CTGAN wrapper for tabular features
│   ├── brand_name_generator.py    # DistilGPT2 for brand name generation
│   └── evaluator.py               # Evaluation and visualization
├── notebooks/
│   └── synthetic_brand_generation.ipynb  # Main pipeline notebook
├── data/
│   ├── raw/                       # Original datasets
│   └── generated/                 # Synthetic and augmented datasets
├── models/                        # Trained model checkpoints
├── final_idea2.md                 # Project plan and methodology
└── README.md                      # This file
```

## Requirements

### Core Libraries

```bash
pip install sdv transformers torch pandas numpy scikit-learn matplotlib seaborn plotly scipy
```

- **sdv** (v1.x): Synthetic Data Vault (includes CTGAN, TVAE, CopulaGAN)
- **transformers**: HuggingFace for GPT-2/DistilGPT2
- **torch**: PyTorch backend
- **scikit-learn**: Clustering and metrics
- **pandas/numpy**: Data manipulation
- **matplotlib/seaborn/plotly**: Visualization
- **scipy**: Statistical tests

### System Requirements

- **For Colab**: Free tier with T4 GPU (12-15GB RAM)
- **For Local**: 16GB+ RAM, CUDA-compatible GPU recommended
- **Python**: 3.8+

## Usage

### Quick Start (Jupyter Notebook)

1. **Open the main notebook**:
   ```
   notebooks/synthetic_brand_generation.ipynb
   ```

2. **For Google Colab**:
   - Upload the notebook to Colab
   - Enable GPU runtime (Runtime > Change runtime type > GPU)
   - Upload `data/raw/brand_information.csv` or mount Google Drive
   - Run all cells

3. **For Local Jupyter**:
   ```bash
   jupyter notebook notebooks/synthetic_brand_generation.ipynb
   ```

### Using the Python Classes

```python
from src.data_processor import BrandDataProcessor
from src.tabular_gan import TabularBrandGAN
from src.brand_name_generator import BrandNameGenerator
from src.evaluator import BrandDataEvaluator

# 1. Load and preprocess data
processor = BrandDataProcessor('data/raw/brand_information.csv')
train_df, val_df = processor.prepare_for_gan()

# 2. Train CTGAN
ctgan = TabularBrandGAN(epochs=300, batch_size=500)
ctgan.train(train_df)

# 3. Fine-tune brand name generator
name_gen = BrandNameGenerator(model_name='distilgpt2')
name_gen.fine_tune(brands_df, epochs=3)

# 4. Generate synthetic brands
synthetic = ctgan.generate_for_companies(companies=['PepsiCo, Inc.'], n_per_company=10)
synthetic_with_names = name_gen.generate_for_dataframe(synthetic)

# 5. Evaluate
evaluator = BrandDataEvaluator()
comparison = evaluator.compare_clustering(original_data, augmented_data, numerical_cols)
evaluator.plot_comprehensive_evaluation(...)
```

## Pipeline Phases

### Phase 1: Data Preparation & Exploration

- Load brand dataset (3,605 brands, 80+ features)
- Explore brand distribution per company
- Handle missing values
- Encode categorical features
- Train/validation split (stratified by company)

### Phase 2: CTGAN Training

- Initialize CTGAN with appropriate hyperparameters
- Train on preprocessed brand features
- Condition on `company_name` for realistic generation
- Save model checkpoint

**Training Time**: 20-40 minutes on GPU (depending on epochs)

### Phase 3: Brand Name Generation

- Fine-tune DistilGPT2 on existing brand names
- Format: `"Company: {company} | Industry: {industry} | Brand: {name}"`
- Learn company-specific naming patterns
- Save fine-tuned model

**Training Time**: 10-30 minutes on GPU

### Phase 4: Synthetic Data Generation

- Identify companies needing more brands
- Generate synthetic features using CTGAN
- Generate brand names using DistilGPT2
- Combine into complete synthetic brands
- Save: `data/generated/synthetic_brands.csv`

### Phase 5: Evaluation & Validation

**Statistical Validation**:
- Kolmogorov-Smirnov tests (distribution similarity)
- Correlation preservation analysis
- Chi-square tests for categorical features

**Clustering Evaluation**:
- Re-run hierarchical clustering on augmented dataset
- Metrics: Silhouette score, Davies-Bouldin index
- Cluster balance improvement

**Visualizations**:
- Distribution comparisons (histograms, KDE plots)
- PCA/t-SNE scatter plots (original vs synthetic)
- Dendrogram comparisons (before/after)
- Silhouette analysis plots
- Correlation heatmaps
- Comprehensive 9-panel evaluation figure

## Key Features

### Academic-Quality Visualizations

The `evaluator.py` module includes extensive visualization functions for academic presentations:

- `plot_dendrogram()`: Hierarchical clustering tree
- `plot_dendrogram_comparison()`: Side-by-side dendrogram comparison
- `plot_silhouette_analysis()`: Per-cluster silhouette scores
- `plot_tsne_comparison()`: t-SNE dimensionality reduction
- `plot_feature_importance_pca()`: PCA loadings and variance
- `plot_metric_comparison_bars()`: Clustering metric comparison
- `plot_cluster_size_distribution()`: Cluster balance visualization
- `plot_comprehensive_evaluation()`: 9-panel publication-ready figure

### Modular Design

All functionality is organized into clean, reusable Python classes:

- **BrandDataProcessor**: Complete data pipeline (load, clean, encode, split)
- **TabularBrandGAN**: CTGAN training and generation
- **BrandNameGenerator**: GPT-2 fine-tuning and name generation
- **BrandDataEvaluator**: Comprehensive evaluation and visualization

### Google Colab Optimized

- Memory-efficient processing (batch operations)
- Mixed precision training (fp16)
- Model checkpointing to Google Drive
- Fallback to CPU if GPU unavailable
- Clear progress indicators

## Expected Outcomes

1. **Synthetic Brands**: 500-1000 realistic synthetic brands
2. **Improved Clustering**: Better cluster balance (more than 2 clusters)
3. **Quality Metrics**: Statistical similarity to original data
4. **Trained Models**:
   - CTGAN model (`.pkl`)
   - DistilGPT2 brand name generator (HuggingFace format)

## Evaluation Metrics

### Clustering Quality
- **Silhouette Score**: Measures cluster cohesion (higher is better)
- **Davies-Bouldin Index**: Measures cluster separation (lower is better)
- **Cluster Balance**: Distribution of samples across clusters

### Synthetic Data Quality
- **KS Statistic**: Distribution similarity (p > 0.05 is good)
- **Correlation Preservation**: Maintain feature relationships
- **Privacy**: Ensure synthetic brands are sufficiently different from originals

## Troubleshooting

### Common Issues

**Out of Memory (OOM)**:
- Reduce `batch_size` (CTGAN: 500→256, GPT-2: 8→4)
- Use `distilgpt2` instead of `gpt2`
- Clear cache: `torch.cuda.empty_cache()`

**SDV Installation Issues**:
- Use: `pip install sdv==1.2.0` (specific version)
- If dependency conflicts, create fresh virtual environment

**Colab Session Timeout**:
- Save checkpoints frequently to Google Drive
- Use `model.save()` after training phases
- Can resume from saved models

## Citation

If you use this code for academic work, please cite:

```
Synthetic Brand Generation with GANs
Deep Learning Final Project
[Your Name], [Year]
```

## References

- **CTGAN**: "Modeling Tabular Data using Conditional GAN" (Xu et al., 2019)
- **SDV**: Synthetic Data Vault - https://sdv.dev/
- **DistilGPT2**: "DistilBERT, a distilled version of BERT" (Sanh et al., 2019)
- **HuggingFace Transformers**: https://huggingface.co/

## License

This project is for academic purposes.

## Contact

For questions or issues, please refer to the documentation in `final_idea2.md` or check the inline comments in the code.

---

**Note**: This is an academic project demonstrating the application of generative models to address data imbalance in unsupervised learning scenarios.
