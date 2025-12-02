"""
Tabular GAN module for generating synthetic brand features.
Uses CTGAN from the SDV library.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict
import pickle
import os


class TabularBrandGAN:
    """
    Wrapper for CTGAN to generate synthetic brand features.
    """

    def __init__(self, epochs: int = 300, batch_size: int = 500, verbose: bool = True):
        """
        Initialize the Tabular GAN.

        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Whether to print training progress
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = None
        self.metadata = None
        self.condition_column = 'company_name'

    def train(self, data: pd.DataFrame, discrete_columns: List[str] = None):
        """
        Train the CTGAN model on brand data.

        Args:
            data: Training dataframe
            discrete_columns: List of categorical column names
        """
        try:
            from sdv.single_table import CTGANSynthesizer
            from sdv.metadata import SingleTableMetadata
        except ImportError:
            print("Installing SDV library...")
            import subprocess
            subprocess.check_call(['pip', 'install', 'sdv'])
            from sdv.single_table import CTGANSynthesizer
            from sdv.metadata import SingleTableMetadata

        print("\n=== Training CTGAN ===")
        print(f"Training on {len(data)} samples with {len(data.columns)} features")

        # Create metadata
        self.metadata = SingleTableMetadata()
        self.metadata.detect_from_dataframe(data)

        # Initialize CTGAN
        self.model = CTGANSynthesizer(
            metadata=self.metadata,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose
        )

        # Train the model
        print(f"Training for {self.epochs} epochs with batch size {self.batch_size}...")
        self.model.fit(data)
        print("Training completed!")

    def generate(self, n_samples: int, condition_column: Optional[str] = None,
                 condition_value: Optional[str] = None) -> pd.DataFrame:
        """
        Generate synthetic brand data.

        Args:
            n_samples: Number of samples to generate
            condition_column: Column to condition on (e.g., 'company_name')
            condition_value: Value to condition on (e.g., 'PepsiCo, Inc.')

        Returns:
            Dataframe of synthetic brand features
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        print(f"\n=== Generating {n_samples} Synthetic Brands ===")

        if condition_column and condition_value is not None:
            print(f"Conditioning on {condition_column} = {condition_value}")
            conditions = pd.DataFrame({condition_column: [condition_value] * n_samples})
            synthetic_data = self.model.sample_from_conditions(
                conditions=conditions
            )
        else:
            print("Generating unconditionally...")
            synthetic_data = self.model.sample(num_rows=n_samples)

        print(f"Generated {len(synthetic_data)} synthetic samples")
        return synthetic_data

    def generate_for_companies(self, companies: List[str], n_per_company: int = 5) -> pd.DataFrame:
        """
        Generate synthetic brands for multiple companies.

        Args:
            companies: List of company names
            n_per_company: Number of brands to generate per company

        Returns:
            Dataframe of all synthetic brands
        """
        all_synthetic = []

        for company in companies:
            print(f"\nGenerating {n_per_company} brands for {company}...")
            synthetic = self.generate(
                n_samples=n_per_company,
                condition_column=self.condition_column,
                condition_value=company
            )
            all_synthetic.append(synthetic)

        combined = pd.concat(all_synthetic, ignore_index=True)
        print(f"\n=== Generated {len(combined)} total synthetic brands ===")
        return combined

    def save_model(self, filepath: str):
        """
        Save the trained model.

        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """
        Load a trained model.

        Args:
            filepath: Path to the saved model
        """
        try:
            from sdv.single_table import CTGANSynthesizer
        except ImportError:
            print("Installing SDV library...")
            import subprocess
            subprocess.check_call(['pip', 'install', 'sdv'])
            from sdv.single_table import CTGANSynthesizer

        self.model = CTGANSynthesizer.load(filepath)
        print(f"Model loaded from {filepath}")

    def evaluate_quality(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> Dict:
        """
        Evaluate the quality of synthetic data.

        Args:
            real_data: Original data
            synthetic_data: Generated synthetic data

        Returns:
            Dictionary of quality metrics
        """
        try:
            from sdv.evaluation.single_table import evaluate_quality
        except ImportError:
            print("SDV evaluation not available, using basic metrics")
            return self._basic_quality_metrics(real_data, synthetic_data)

        print("\n=== Evaluating Synthetic Data Quality ===")
        quality_report = evaluate_quality(
            real_data=real_data,
            synthetic_data=synthetic_data,
            metadata=self.metadata
        )

        print(f"Overall Quality Score: {quality_report.get_score():.3f}")
        return quality_report

    def _basic_quality_metrics(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> Dict:
        """Basic quality metrics if SDV evaluation unavailable."""
        metrics = {}

        # Compare means for numerical columns
        num_cols = real_data.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            real_mean = real_data[col].mean()
            synth_mean = synthetic_data[col].mean()
            metrics[f'{col}_mean_diff'] = abs(real_mean - synth_mean)

        return metrics
