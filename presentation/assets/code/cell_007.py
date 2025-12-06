# Clone repository and install dependencies
!git clone https://github.com/dyegofern/csca5642-deep-learning.git
!pip install -q sdv transformers torch pandas numpy scikit-learn matplotlib seaborn plotly scipy
!pip install -q peft bitsandbytes accelerate sentencepiece
!pip install -q optuna

import sys
import os
from google.colab import drive

MAPPED_DIR = '/content/csca5642-deep-learning'

# Mount Google Drive
print("Mounting Google Drive...")
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')
else:
    print("Google Drive already mounted")

DATA_PATH = MAPPED_DIR + '/data/raw/brand_information.csv'

# Set output and model directories to Google Drive
DRIVE_OUTPUT_BASE = '/content/drive/MyDrive/Colab_Output/SyntheticBrandGeneration_V2'
OUTPUT_DIR = os.path.join(DRIVE_OUTPUT_BASE, 'outputs')
MODEL_DIR = os.path.join(DRIVE_OUTPUT_BASE, 'models')

# Create directories
print(f"\nCreating directories in Google Drive...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}")
print(f"Model directory: {MODEL_DIR}")

# Add src to path
src_path = MAPPED_DIR + '/src'
if src_path not in sys.path:
    sys.path.append(src_path)

print(f"\nSetup complete!")