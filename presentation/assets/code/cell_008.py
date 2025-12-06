# Import V2 modules
from data_processor import BrandDataProcessor
from tabular_gan_v2 import (
    EnsembleSynthesizer,
    CTGANSynthesizerWrapper,
    TVAESynthesizerWrapper,
    GaussianCopulaSynthesizerWrapper,
    calculate_generation_targets
)
from brand_name_generator_v2 import BrandNameGeneratorV2
from evaluator import BrandDataEvaluator

# Standard libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import gc

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Check GPU
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print("\nAll V2 modules loaded successfully!")