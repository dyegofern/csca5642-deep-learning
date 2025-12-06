# Synthetic Brand Generation with Deep Learning

University of Colorado Boulder - CSCA 5642: Introduction to Deep Learning

**Author**: Dyego Fernandes de Sousa

## Project Overview

This project tackles class imbalance in hierarchical clustering of ESG (Environmental, Social, and Governance) brand data by generating synthetic brands using an ensemble of generative models. The solution combines tabular synthesizers with Large Language Models for complete brand generation.

### Solution: Hybrid Ensemble Approach

**Tabular Data Generation** (Ensemble of 3 models):
1. **CTGAN** (Conditional Tabular GAN): Generator/Discriminator for tabular features
2. **TVAE** (Tabular Variational Autoencoder): Encoder/Decoder for distribution learning
3. **Gaussian Copula**: Statistical method for correlation preservation

**Brand Name Generation** (LLM Ensemble):
1. **GPT-2 Medium** (355M params): Fine-tuned for brand name generation
2. **Flan-T5 Base**: Instruction-following encoder-decoder model

## Dataset

- **Source**: `data/raw/brand_information.csv`
- **Size**: 3,605 brands, 77 features
- **Features**: ESG metrics, demographics, business characteristics, financial data

## Project Structure

```
csca5642-deep-learning/
├── src/
│   ├── data_processor.py           # Data loading and preprocessing
│   ├── tabular_gan_v2.py           # Ensemble synthesizers (CTGAN, TVAE, Gaussian Copula)
│   ├── brand_name_generator_v2.py  # LLM ensemble (GPT-2, Flan-T5)
│   ├── hyperparameter_tuner_v2.py  # Optuna-based hyperparameter optimization
│   └── evaluator.py                # Evaluation and visualization
├── notebooks/
│   └── synthetic_brand_generation_v2.ipynb  # Main pipeline notebook
├── presentation/
│   └── presentation.pdf            # Project presentation
├── data/
│   ├── raw/                        # Original datasets
│   └── generated/                  # Synthetic and augmented datasets
└── models/                         # Trained model checkpoints
```

## Requirements

### Core Libraries

```bash
pip install sdv transformers torch pandas numpy scikit-learn matplotlib seaborn scipy optuna
```

- **sdv** (v1.x): Synthetic Data Vault (CTGAN, TVAE, GaussianCopula)
- **transformers**: HuggingFace for GPT-2 and Flan-T5
- **torch**: PyTorch backend
- **optuna**: Hyperparameter optimization
- **scikit-learn**: Clustering and metrics

### System Requirements

- **Google Colab**: T4 GPU recommended (12-15GB RAM)
- **Python**: 3.8+

## Pipeline Phases

### Phase 0: Hyperparameter Tuning (Optional)

Uses Optuna to find optimal hyperparameters for tabular synthesizers:
- CTGAN/TVAE epochs, batch size, embedding dimensions
- Ensemble weights optimization
- Best parameters saved to JSON

### Phase 1: Data Preparation

- Load brand dataset (3,605 brands, 77 features)
- Handle missing values and encode categorical features
- Train/validation split (stratified by company)

### Phase 2: Tabular Ensemble Training

Train three models with optimized weights:
- **CTGAN** (~10.8% weight): Mode-specific normalization, conditional vector
- **TVAE** (~58% weight): KL divergence, latent space regularization
- **Gaussian Copula** (~31.2% weight): Multivariate dependencies

**Training Time**: 20-40 minutes on GPU

### Phase 3: LLM Ensemble Training

Fine-tune language models for brand name generation:
- **GPT-2 Medium**: Causal language modeling
- **Flan-T5 Base**: Sequence-to-sequence with instruction tuning

Format: `"Company: {company} | Industry: {industry} | Brand: {name}"`

**Training Time**: 10-30 minutes on GPU

### Phase 4: Synthetic Data Generation

- Generate synthetic features using tabular ensemble (weighted average for numerical, majority voting for categorical)
- Generate brand names using LLM ensemble with quality-weighted voting
- Combine into complete synthetic brands

### Phase 5: Evaluation

**Statistical Validation**:
- Kolmogorov-Smirnov tests (distribution similarity)
- Correlation preservation analysis (MSE: 0.0197)

**Metrics Achieved**:
- Mean KS Statistic: 0.286
- LLM Success Rate: 95.4%
- Overall Score: 0.240

**Visualizations**:
- Distribution comparisons (histograms, KDE, QQ plots)
- PCA/t-SNE dimensionality reduction
- Correlation heatmaps
- Quality scorecards and radar charts

## Usage

### Quick Start (Google Colab)

1. Open `notebooks/synthetic_brand_generation_v2.ipynb` in Colab
2. Enable GPU runtime (Runtime > Change runtime type > GPU)
3. Mount Google Drive for model persistence
4. Run all cells

### Using the Python Classes

```python
from data_processor import BrandDataProcessor
from tabular_gan_v2 import EnsembleSynthesizer, CTGANSynthesizerWrapper, TVAESynthesizerWrapper, GaussianCopulaSynthesizerWrapper
from brand_name_generator_v2 import BrandNameGeneratorV2
from evaluator import BrandDataEvaluator

# 1. Load and preprocess data
processor = BrandDataProcessor('data/raw/brand_information.csv')
train_df, val_df = processor.prepare_for_gan()

# 2. Train Tabular Ensemble
tabular_ensemble = EnsembleSynthesizer(
    ctgan_wrapper=CTGANSynthesizerWrapper(...),
    tvae_wrapper=TVAESynthesizerWrapper(...),
    gc_wrapper=GaussianCopulaSynthesizerWrapper(...),
    weights={'ctgan': 0.108, 'tvae': 0.580, 'gaussian_copula': 0.312}
)
tabular_ensemble.fit(train_df)

# 3. Train LLM Ensemble
llm_generator = BrandNameGeneratorV2(models=['gpt2-medium', 'flan-t5-base'])
llm_generator.fine_tune_all(brands_df)

# 4. Generate synthetic brands
synthetic_features = tabular_ensemble.sample(n=100)
synthetic_with_names = llm_generator.generate_for_dataframe(synthetic_features)

# 5. Evaluate
evaluator = BrandDataEvaluator()
evaluator.compare_distributions(original_df, synthetic_df)
```

## Results Summary

| Category | Topic | Details |
|----------|-------|---------|
| **What Worked** | Ensemble Architecture | CTGAN, TVAE, GC complemented; TVAE ~54% weight (KS: 0.11), GC preserved correlations (MSE: 0.0197) |
| | Scalable Pipeline | Stratified generation, conditional synthesis, Google Drive persistence |
| | LLM Brand Names | GPT-2 + Flan-T5 achieved 95.4% success rate, memory-efficient |
| **Challenges** | LLM Quality Issues | Full sentences, competitor leakage, repetitive patterns, 4.6% fallback |
| **Future Work** | Tabular Synthesis | Feature preprocessing, constrained generation, TabDDPM, validation |
| | LLM Enhancement | Negative examples, stricter validation, RAG, style conditioning |
| | Architecture | Attention-based models, hierarchical pipeline, discriminator filtering |

## References

- **CTGAN**: Xu, L., et al. (2019). *Modeling Tabular Data using Conditional GAN*. NeurIPS 2019.
- **TVAE**: Same authors, variational approach for tabular data.
- **Gaussian Copula**: Statistical method for modeling dependencies.
- **GPT-2**: Radford, A., et al. (2019). *Language Models are Unsupervised Multitask Learners*.
- **Flan-T5**: Chung, H.W., et al. (2022). *Scaling Instruction-Finetuned Language Models*.
- **SDV**: Synthetic Data Vault - https://sdv.dev/
- **Optuna**: Akiba, T., et al. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework*.

## Links

- **GitHub**: https://github.com/dyegofern/csca5642-deep-learning
- **Presentation**: [presentation/presentation.pdf](presentation/presentation.pdf)

## License

This project is for academic purposes.
