"""
Tabular GAN module for generating synthetic brand features.
Uses CTGAN from the SDV library with efficient batch generation.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict
import pickle
import os


class TabularBrandGAN:
    """
    Wrapper for CTGAN to generate synthetic brand features with efficient batch operations.
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

    def train(self, data: pd.DataFrame, discrete_columns: List[str] = None,
              binary_columns: List[str] = None):
        """
        Train the CTGAN model on brand data.

        Args:
            data: Training dataframe
            discrete_columns: List of categorical column names
            binary_columns: List of binary (0/1) column names to be treated as categorical
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

        # Update binary columns to be treated as categorical (boolean)
        if binary_columns:
            print(f"Setting {len(binary_columns)} binary columns as boolean type...")
            for col in binary_columns:
                if col in data.columns:
                    self.metadata.update_column(
                        column_name=col,
                        sdtype='boolean'
                    )

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
                 condition_value: Optional[any] = None, max_tries: int = 100) -> pd.DataFrame:
        """
        Generate synthetic data in a single efficient batch.

        Args:
            n_samples: Number of samples to generate
            condition_column: Column to condition on (e.g., 'company_name')
            condition_value: Value to condition on (e.g., 156 for Nestle)
            max_tries: Maximum tries per batch for conditional sampling

        Returns:
            Dataframe of synthetic brand features

        Raises:
            ValueError: If generation fails (caller should handle this)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        print(f"\n=== Generating {n_samples} Synthetic Brands ===")

        if condition_column and condition_value is not None:
            print(f"Conditioning on {condition_column} = {condition_value}")

            # Batch conditional generation using sample_remaining_columns
            conditions_df = pd.DataFrame({
                condition_column: [condition_value] * n_samples
            })

            synthetic_data = self.model.sample_remaining_columns(
                known_columns=conditions_df,
                max_tries_per_batch=max_tries
            )
        else:
            print("Generating unconditionally...")
            synthetic_data = self.model.sample(num_rows=n_samples)

        # Remove SDV internal ID columns (they have 'sdv-id-' prefix in values)
        cols_to_drop = []
        for col in synthetic_data.columns:
            if synthetic_data[col].dtype == 'object':
                # Check if any values start with 'sdv-id-'
                sample_vals = synthetic_data[col].astype(str).head(10)
                if sample_vals.str.startswith('sdv-id-').any():
                    cols_to_drop.append(col)
        if cols_to_drop:
            synthetic_data = synthetic_data.drop(columns=cols_to_drop)
            print(f"Removed SDV internal columns: {cols_to_drop}")

        print(f"Generated {len(synthetic_data)} synthetic samples")
        return synthetic_data

    def generate_for_companies(self, companies: List[any],
                              n_per_company: int = 100,
                              verbose: bool = True) -> pd.DataFrame:
        """
        Efficiently generate synthetic brands for multiple companies using batch generation.

        Args:
            companies: List of company identifiers (encoded values)
            n_per_company: Number of brands to generate per company
            verbose: Print progress

        Returns:
            Dataframe of all synthetic brands
        """
        if verbose:
            print(f"\n=== Generating Brands for {len(companies)} Companies ===")
            print(f"  Brands per company: {n_per_company}")
            print(f"  Total brands to generate: {len(companies) * n_per_company}")

        all_synthetic = []

        for i, company in enumerate(companies):
            if verbose and (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{len(companies)} companies...")

            # Generate entire batch at once (MUCH faster than loop)
            synthetic = self.generate(
                n_samples=n_per_company,
                condition_column=self.condition_column,
                condition_value=company
            )
            all_synthetic.append(synthetic)

        combined = pd.concat(all_synthetic, ignore_index=True)

        if verbose:
            print(f"\n✓ Generated {len(combined)} total synthetic brands")
            print(f"  Companies covered: {combined[self.condition_column].nunique()}")

        return combined

    def generate_stratified(self, company_distribution: Dict[any, int],
                          verbose: bool = True, max_tries: int = 100) -> tuple[pd.DataFrame, Dict]:
        """
        Generate brands with custom distribution per company with error handling.

        Args:
            company_distribution: Dict mapping company_id -> number of brands to generate
            verbose: Print progress
            max_tries: Maximum tries per batch for conditional sampling

        Returns:
            Tuple of (dataframe of synthetic brands, dict of failed companies with error messages)
        """
        if verbose:
            print(f"\n=== Stratified Generation ===")
            print(f"  Companies: {len(company_distribution)}")
            print(f"  Total brands requested: {sum(company_distribution.values())}")

        all_synthetic = []
        failed_companies = {}
        successful_count = 0
        failed_count = 0

        for i, (company, n_brands) in enumerate(company_distribution.items()):
            if verbose and (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{len(company_distribution)} companies (Success: {successful_count}, Failed: {failed_count})...")

            if n_brands > 0:
                try:
                    synthetic = self.generate(
                        n_samples=n_brands,
                        condition_column=self.condition_column,
                        condition_value=company,
                        max_tries=max_tries
                    )
                    all_synthetic.append(synthetic)
                    successful_count += 1
                except ValueError as e:
                    failed_count += 1
                    error_msg = str(e)
                    failed_companies[company] = error_msg
                    if verbose:
                        print(f"  ⚠ WARNING: Failed to generate for company {company}: {error_msg[:80]}...")
                except Exception as e:
                    failed_count += 1
                    error_msg = f"Unexpected error: {str(e)}"
                    failed_companies[company] = error_msg
                    if verbose:
                        print(f"  ⚠ WARNING: Unexpected error for company {company}: {str(e)[:80]}...")

        if all_synthetic:
            combined = pd.concat(all_synthetic, ignore_index=True)
        else:
            # Return empty dataframe with correct structure if all failed
            combined = pd.DataFrame()

        if verbose:
            print(f"\n{'='*60}")
            print(f"Generation Summary:")
            print(f"  ✓ Successful: {successful_count} companies")
            print(f"  ✗ Failed: {failed_count} companies")
            print(f"  Total brands generated: {len(combined)}")
            if failed_companies:
                print(f"\n⚠ Failed companies will be logged for review")
            print(f"{'='*60}")

        return combined, failed_companies

    def add_diversity_noise(self, synthetic_data: pd.DataFrame,
                           noise_level: float = 0.02,
                           numerical_cols: List[str] = None) -> pd.DataFrame:
        """
        Add small random noise to increase diversity in generated data.

        Args:
            synthetic_data: Generated synthetic data
            noise_level: Proportion of noise to add (0.01 = 1% noise)
            numerical_cols: List of numerical columns to add noise to

        Returns:
            Data with added noise
        """
        if numerical_cols is None:
            numerical_cols = synthetic_data.select_dtypes(include=[np.number]).columns

        noisy_data = synthetic_data.copy()

        for col in numerical_cols:
            if col in noisy_data.columns:
                # Add Gaussian noise proportional to the std dev
                std = noisy_data[col].std()
                if std > 0:
                    noise = np.random.normal(0, std * noise_level, len(noisy_data))
                    noisy_data[col] = noisy_data[col] + noise

                    # Clip to valid range if column has constraints
                    if (noisy_data[col] >= 0).all():  # Non-negative column
                        noisy_data[col] = noisy_data[col].clip(lower=0)

        return noisy_data

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


def calculate_generation_targets(data: pd.DataFrame,
                                 company_column: str = 'company_name',
                                 min_brands_per_company: int = 100,
                                 target_total_brands: int = None) -> Dict[any, int]:
    """
    Calculate how many brands to generate for each company to reach the target.

    Args:
        data: Original dataset with company_name column
        company_column: Name of the company column
        min_brands_per_company: Minimum brands each company should have
        target_total_brands: Target total number of brands (if set, overrides min_brands)

    Returns:
        Dictionary mapping company -> number of brands to generate
    """
    current_counts = data[company_column].value_counts().to_dict()

    if target_total_brands:
        # Distribute total brands proportionally
        total_current = len(data)
        brands_to_generate = target_total_brands - total_current

        if brands_to_generate <= 0:
            print(f"Already have {total_current} brands, target is {target_total_brands}")
            return {}

        # Proportional distribution based on current size
        generation_targets = {}
        for company, current_count in current_counts.items():
            proportion = current_count / total_current
            to_generate = int(brands_to_generate * proportion)
            generation_targets[company] = to_generate

    else:
        # Fill each company to minimum threshold
        generation_targets = {}
        for company, current_count in current_counts.items():
            if current_count < min_brands_per_company:
                to_generate = min_brands_per_company - current_count
                generation_targets[company] = to_generate
            else:
                # Still generate some for diversity, but fewer
                to_generate = max(10, int(current_count * 0.1))  # 10% more
                generation_targets[company] = to_generate

    print(f"\n=== Generation Targets ===")
    print(f"  Total companies: {len(generation_targets)}")
    print(f"  Total brands to generate: {sum(generation_targets.values())}")
    print(f"  Average per company: {sum(generation_targets.values()) / len(generation_targets):.1f}")
    print(f"\n  Top 10 companies to augment:")
    sorted_targets = sorted(generation_targets.items(), key=lambda x: x[1], reverse=True)[:10]
    for company, count in sorted_targets:
        print(f"    {company}: +{count} brands")

    return generation_targets
