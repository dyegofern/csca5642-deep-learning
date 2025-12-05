"""
Tabular GAN V2 module for generating synthetic brand features.
Implements ensemble of CTGAN, TVAE, and Gaussian Copula synthesizers.

Features:
- TVAE (Tabular Variational Autoencoder) for better continuous distributions
- Gaussian Copula for better correlation structure preservation
- Ensemble methods with voting/averaging across generators
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple, Any
from abc import ABC, abstractmethod
from collections import Counter
import os
import json
import gc


class BaseSynthesizerWrapper(ABC):
    """Abstract base class for all synthesizer wrappers."""

    def __init__(self, verbose: bool = True):
        """
        Initialize the base synthesizer wrapper.

        Args:
            verbose: Whether to print progress messages
        """
        self.verbose = verbose
        self.model = None
        self.metadata = None
        self.condition_column = 'company_name'
        self._fitted = False
        self.model_type = 'base'

    @abstractmethod
    def train(self, data: pd.DataFrame, discrete_columns: List[str] = None,
              binary_columns: List[str] = None) -> None:
        """
        Train the synthesizer on data.

        Args:
            data: Training dataframe
            discrete_columns: List of categorical column names
            binary_columns: List of binary (0/1) column names
        """
        pass

    @abstractmethod
    def generate(self, n_samples: int, condition_column: Optional[str] = None,
                 condition_value: Optional[Any] = None, max_tries: int = 100) -> pd.DataFrame:
        """
        Generate synthetic samples.

        Args:
            n_samples: Number of samples to generate
            condition_column: Column to condition on
            condition_value: Value to condition on
            max_tries: Maximum tries for conditional sampling

        Returns:
            DataFrame of synthetic samples
        """
        pass

    @abstractmethod
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        pass

    @abstractmethod
    def load_model(self, filepath: str) -> None:
        """Load a pre-trained model."""
        pass

    def generate_for_companies(self, companies: List[Any],
                               n_per_company: int = 100,
                               verbose: bool = True) -> pd.DataFrame:
        """
        Generate synthetic brands for multiple companies.

        Args:
            companies: List of company identifiers
            n_per_company: Number of brands per company
            verbose: Print progress

        Returns:
            DataFrame of all synthetic brands
        """
        if verbose:
            print(f"\n=== Generating Brands for {len(companies)} Companies ===")
            print(f"  Brands per company: {n_per_company}")

        all_synthetic = []

        for i, company in enumerate(companies):
            if verbose and (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{len(companies)} companies...")

            synthetic = self.generate(
                n_samples=n_per_company,
                condition_column=self.condition_column,
                condition_value=company
            )
            all_synthetic.append(synthetic)

        combined = pd.concat(all_synthetic, ignore_index=True)

        if verbose:
            print(f"\nGenerated {len(combined)} total synthetic brands")

        return combined

    def generate_stratified(self, company_distribution: Dict[Any, int],
                           verbose: bool = True, max_tries: int = 100) -> Tuple[pd.DataFrame, Dict]:
        """
        Generate brands with custom distribution per company.

        Args:
            company_distribution: Dict mapping company_id -> number of brands
            verbose: Print progress
            max_tries: Maximum tries for conditional sampling

        Returns:
            Tuple of (synthetic data, failed companies dict)
        """
        if verbose:
            print(f"\n=== Stratified Generation ({self.model_type}) ===")
            print(f"  Companies: {len(company_distribution)}")
            print(f"  Total brands requested: {sum(company_distribution.values())}")

        all_synthetic = []
        failed_companies = {}

        for i, (company, n_brands) in enumerate(company_distribution.items()):
            if verbose and (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{len(company_distribution)} companies...")

            if n_brands > 0:
                try:
                    synthetic = self.generate(
                        n_samples=n_brands,
                        condition_column=self.condition_column,
                        condition_value=company,
                        max_tries=max_tries
                    )
                    all_synthetic.append(synthetic)
                except Exception as e:
                    failed_companies[company] = str(e)
                    if verbose:
                        print(f"  Warning: Failed for company {company}: {str(e)[:50]}...")

        if all_synthetic:
            combined = pd.concat(all_synthetic, ignore_index=True)
        else:
            combined = pd.DataFrame()

        if verbose:
            print(f"\nGenerated {len(combined)} total synthetic brands")
            if failed_companies:
                print(f"Failed companies: {len(failed_companies)}")

        return combined, failed_companies

    def add_diversity_noise(self, synthetic_data: pd.DataFrame,
                           noise_level: float = 0.02,
                           numerical_cols: List[str] = None) -> pd.DataFrame:
        """
        Add Gaussian noise to numerical columns for diversity.

        Args:
            synthetic_data: Generated synthetic data
            noise_level: Proportion of noise (0.02 = 2%)
            numerical_cols: Columns to add noise to

        Returns:
            DataFrame with added noise
        """
        if numerical_cols is None:
            numerical_cols = synthetic_data.select_dtypes(include=[np.number]).columns.tolist()

        noisy_data = synthetic_data.copy()

        for col in numerical_cols:
            if col in noisy_data.columns:
                std = noisy_data[col].std()
                if std > 0:
                    noise = np.random.normal(0, std * noise_level, len(noisy_data))
                    noisy_data[col] = noisy_data[col] + noise

                    # Clip non-negative columns
                    if (synthetic_data[col] >= 0).all():
                        noisy_data[col] = noisy_data[col].clip(lower=0)

        return noisy_data

    def _remove_sdv_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove SDV internal ID columns."""
        cols_to_drop = []
        for col in data.columns:
            if data[col].dtype == 'object':
                sample_vals = data[col].astype(str).head(10)
                if sample_vals.str.startswith('sdv-id-').any():
                    cols_to_drop.append(col)
        if cols_to_drop:
            data = data.drop(columns=cols_to_drop)
        return data

    def _convert_binary_to_bool(self, data: pd.DataFrame, binary_columns: List[str]) -> pd.DataFrame:
        """Convert binary 0/1 columns to actual boolean True/False for SDV."""
        data = data.copy()
        if binary_columns:
            for col in binary_columns:
                if col in data.columns:
                    # Convert 0/1 to True/False
                    data[col] = data[col].astype(bool)
        return data

    def _convert_bool_to_binary(self, data: pd.DataFrame, binary_columns: List[str]) -> pd.DataFrame:
        """Convert boolean True/False back to 0/1 integers."""
        data = data.copy()
        if binary_columns:
            for col in binary_columns:
                if col in data.columns:
                    # Convert True/False back to 1/0
                    data[col] = data[col].astype(int)
        return data


class CTGANSynthesizerWrapper(BaseSynthesizerWrapper):
    """Wrapper for CTGAN from SDV library."""

    def __init__(self, epochs: int = 300, batch_size: int = 500,
                 embedding_dim: int = 128,
                 generator_dim: Tuple[int, int] = (256, 256),
                 discriminator_dim: Tuple[int, int] = (256, 256),
                 generator_lr: float = 2e-4,
                 discriminator_lr: float = 2e-4,
                 verbose: bool = True,
                 cuda: bool = True):
        """
        Initialize CTGAN Synthesizer wrapper.

        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            embedding_dim: Dimension of latent embedding
            generator_dim: Generator hidden layer sizes
            discriminator_dim: Discriminator hidden layer sizes
            generator_lr: Generator learning rate
            discriminator_lr: Discriminator learning rate
            verbose: Print training progress
            cuda: Use GPU if available
        """
        super().__init__(verbose)
        self.epochs = epochs
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr
        self.cuda = cuda
        self.model_type = 'ctgan'
        self._binary_columns = []  # Store for conversion back after generation

    def train(self, data: pd.DataFrame, discrete_columns: List[str] = None,
              binary_columns: List[str] = None) -> None:
        """Train CTGAN on data."""
        from sdv.single_table import CTGANSynthesizer
        from sdv.metadata import SingleTableMetadata

        if self.verbose:
            print(f"\n=== Training CTGAN ===")
            print(f"Training on {len(data)} samples with {len(data.columns)} features")

        # Store binary columns for later conversion
        self._binary_columns = binary_columns or []

        # Convert binary 0/1 to actual boolean True/False for SDV
        if binary_columns:
            if self.verbose:
                print(f"Converting {len(binary_columns)} binary columns to boolean...")
            data = self._convert_binary_to_bool(data, binary_columns)

        # Create metadata
        self.metadata = SingleTableMetadata()
        self.metadata.detect_from_dataframe(data)

        # Update binary columns to boolean type in metadata
        if binary_columns:
            if self.verbose:
                print(f"Setting {len(binary_columns)} binary columns as boolean type in metadata...")
            for col in binary_columns:
                if col in data.columns:
                    self.metadata.update_column(column_name=col, sdtype='boolean')

        # Initialize CTGAN
        self.model = CTGANSynthesizer(
            metadata=self.metadata,
            epochs=self.epochs,
            batch_size=self.batch_size,
            embedding_dim=self.embedding_dim,
            generator_dim=self.generator_dim,
            discriminator_dim=self.discriminator_dim,
            generator_lr=self.generator_lr,
            discriminator_lr=self.discriminator_lr,
            verbose=self.verbose,
            cuda=self.cuda
        )

        if self.verbose:
            print(f"Training for {self.epochs} epochs with batch size {self.batch_size}...")

        self.model.fit(data)
        self._fitted = True

        if self.verbose:
            print("CTGAN Training completed!")

    def generate(self, n_samples: int, condition_column: Optional[str] = None,
                 condition_value: Optional[Any] = None, max_tries: int = 100) -> pd.DataFrame:
        """Generate synthetic samples using CTGAN."""
        if not self._fitted:
            raise ValueError("Model not trained yet. Call train() first.")

        if condition_column and condition_value is not None:
            conditions_df = pd.DataFrame({
                condition_column: [condition_value] * n_samples
            })
            synthetic_data = self.model.sample_remaining_columns(
                known_columns=conditions_df,
                max_tries_per_batch=max_tries
            )
        else:
            synthetic_data = self.model.sample(num_rows=n_samples)

        # Remove SDV internal columns
        synthetic_data = self._remove_sdv_columns(synthetic_data)

        # Convert boolean columns back to 0/1 integers
        if self._binary_columns:
            synthetic_data = self._convert_bool_to_binary(synthetic_data, self._binary_columns)

        return synthetic_data

    def save_model(self, filepath: str) -> None:
        """Save the trained CTGAN model."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        self.model.save(filepath)
        if self.verbose:
            print(f"CTGAN model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load a trained CTGAN model."""
        from sdv.single_table import CTGANSynthesizer
        self.model = CTGANSynthesizer.load(filepath)
        self._fitted = True
        if self.verbose:
            print(f"CTGAN model loaded from {filepath}")


class TVAESynthesizerWrapper(BaseSynthesizerWrapper):
    """Wrapper for TVAE (Tabular Variational Autoencoder) from SDV library."""

    def __init__(self, epochs: int = 300, batch_size: int = 500,
                 embedding_dim: int = 128,
                 compress_dims: Tuple[int, int] = (128, 128),
                 decompress_dims: Tuple[int, int] = (128, 128),
                 l2scale: float = 1e-5,
                 loss_factor: int = 2,
                 verbose: bool = True,
                 cuda: bool = True):
        """
        Initialize TVAE Synthesizer wrapper.

        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            embedding_dim: Dimension of latent embedding
            compress_dims: Encoder hidden layer sizes
            decompress_dims: Decoder hidden layer sizes
            l2scale: L2 regularization weight
            loss_factor: Reconstruction loss multiplier
            verbose: Print training progress
            cuda: Use GPU if available
        """
        super().__init__(verbose)
        self.epochs = epochs
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims
        self.l2scale = l2scale
        self.loss_factor = loss_factor
        self.cuda = cuda
        self.model_type = 'tvae'
        self._binary_columns = []  # Store for conversion back after generation

    def train(self, data: pd.DataFrame, discrete_columns: List[str] = None,
              binary_columns: List[str] = None) -> None:
        """Train TVAE on data."""
        from sdv.single_table import TVAESynthesizer
        from sdv.metadata import SingleTableMetadata

        if self.verbose:
            print(f"\n=== Training TVAE ===")
            print(f"Training on {len(data)} samples with {len(data.columns)} features")

        # Store binary columns for later conversion
        self._binary_columns = binary_columns or []

        # Convert binary 0/1 to actual boolean True/False for SDV
        if binary_columns:
            if self.verbose:
                print(f"Converting {len(binary_columns)} binary columns to boolean...")
            data = self._convert_binary_to_bool(data, binary_columns)

        # Create metadata
        self.metadata = SingleTableMetadata()
        self.metadata.detect_from_dataframe(data)

        # Update binary columns to boolean type in metadata
        if binary_columns:
            if self.verbose:
                print(f"Setting {len(binary_columns)} binary columns as boolean type in metadata...")
            for col in binary_columns:
                if col in data.columns:
                    self.metadata.update_column(column_name=col, sdtype='boolean')

        # Initialize TVAE
        self.model = TVAESynthesizer(
            metadata=self.metadata,
            epochs=self.epochs,
            batch_size=self.batch_size,
            embedding_dim=self.embedding_dim,
            compress_dims=self.compress_dims,
            decompress_dims=self.decompress_dims,
            l2scale=self.l2scale,
            loss_factor=self.loss_factor,
            verbose=self.verbose,
            cuda=self.cuda
        )

        if self.verbose:
            print(f"Training for {self.epochs} epochs with batch size {self.batch_size}...")

        self.model.fit(data)
        self._fitted = True

        if self.verbose:
            print("TVAE Training completed!")

    def generate(self, n_samples: int, condition_column: Optional[str] = None,
                 condition_value: Optional[Any] = None, max_tries: int = 100) -> pd.DataFrame:
        """Generate synthetic samples using TVAE."""
        if not self._fitted:
            raise ValueError("Model not trained yet. Call train() first.")

        if condition_column and condition_value is not None:
            conditions_df = pd.DataFrame({
                condition_column: [condition_value] * n_samples
            })
            synthetic_data = self.model.sample_remaining_columns(
                known_columns=conditions_df,
                max_tries_per_batch=max_tries
            )
        else:
            synthetic_data = self.model.sample(num_rows=n_samples)

        # Remove SDV internal columns
        synthetic_data = self._remove_sdv_columns(synthetic_data)

        # Convert boolean columns back to 0/1 integers
        if self._binary_columns:
            synthetic_data = self._convert_bool_to_binary(synthetic_data, self._binary_columns)

        return synthetic_data

    def save_model(self, filepath: str) -> None:
        """Save the trained TVAE model."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        self.model.save(filepath)
        if self.verbose:
            print(f"TVAE model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load a trained TVAE model."""
        from sdv.single_table import TVAESynthesizer
        self.model = TVAESynthesizer.load(filepath)
        self._fitted = True
        if self.verbose:
            print(f"TVAE model loaded from {filepath}")


class GaussianCopulaSynthesizerWrapper(BaseSynthesizerWrapper):
    """Wrapper for Gaussian Copula from SDV library."""

    def __init__(self,
                 default_distribution: str = 'beta',
                 numerical_distributions: Dict[str, str] = None,
                 verbose: bool = True):
        """
        Initialize Gaussian Copula Synthesizer wrapper.

        Args:
            default_distribution: Default distribution for numerical columns.
                Options: 'norm', 'beta', 'truncnorm', 'uniform', 'gamma', 'gaussian_kde'
            numerical_distributions: Dict mapping column names to distribution types
            verbose: Print progress

        Note: Gaussian Copula is MUCH faster than CTGAN/TVAE but may not capture
              complex nonlinear relationships. Best for data with clear marginal
              distributions and linear correlations.
        """
        super().__init__(verbose)
        self.default_distribution = default_distribution
        self.numerical_distributions = numerical_distributions or {}
        self.model_type = 'gaussian_copula'
        self._binary_columns = []  # Store for conversion back after generation

    def train(self, data: pd.DataFrame, discrete_columns: List[str] = None,
              binary_columns: List[str] = None) -> None:
        """
        Train Gaussian Copula on data.

        Note: Gaussian Copula trains MUCH faster than neural network methods.
        Expected training time: seconds vs 10-30 minutes for CTGAN/TVAE.
        """
        from sdv.single_table import GaussianCopulaSynthesizer
        from sdv.metadata import SingleTableMetadata

        if self.verbose:
            print(f"\n=== Training Gaussian Copula ===")
            print(f"Training on {len(data)} samples with {len(data.columns)} features")

        # Store binary columns for later conversion
        self._binary_columns = binary_columns or []

        # Convert binary 0/1 to actual boolean True/False for SDV
        if binary_columns:
            if self.verbose:
                print(f"Converting {len(binary_columns)} binary columns to boolean...")
            data = self._convert_binary_to_bool(data, binary_columns)

        # Create metadata
        self.metadata = SingleTableMetadata()
        self.metadata.detect_from_dataframe(data)

        # Update binary columns to boolean type in metadata
        if binary_columns:
            if self.verbose:
                print(f"Setting {len(binary_columns)} binary columns as boolean type in metadata...")
            for col in binary_columns:
                if col in data.columns:
                    self.metadata.update_column(column_name=col, sdtype='boolean')

        # Initialize Gaussian Copula
        self.model = GaussianCopulaSynthesizer(
            metadata=self.metadata,
            default_distribution=self.default_distribution,
            numerical_distributions=self.numerical_distributions if self.numerical_distributions else None
        )

        if self.verbose:
            print(f"Fitting Gaussian Copula with '{self.default_distribution}' distribution...")

        self.model.fit(data)
        self._fitted = True

        if self.verbose:
            print("Gaussian Copula Training completed!")

    def generate(self, n_samples: int, condition_column: Optional[str] = None,
                 condition_value: Optional[Any] = None, max_tries: int = 100) -> pd.DataFrame:
        """Generate synthetic samples using Gaussian Copula."""
        if not self._fitted:
            raise ValueError("Model not trained yet. Call train() first.")

        if condition_column and condition_value is not None:
            conditions_df = pd.DataFrame({
                condition_column: [condition_value] * n_samples
            })
            synthetic_data = self.model.sample_remaining_columns(
                known_columns=conditions_df,
                max_tries_per_batch=max_tries
            )
        else:
            synthetic_data = self.model.sample(num_rows=n_samples)

        # Remove SDV internal columns
        synthetic_data = self._remove_sdv_columns(synthetic_data)

        # Convert boolean columns back to 0/1 integers
        if self._binary_columns:
            synthetic_data = self._convert_bool_to_binary(synthetic_data, self._binary_columns)

        return synthetic_data

    def get_learned_distributions(self) -> Dict:
        """Get the learned marginal distributions for each column."""
        if not self._fitted:
            raise ValueError("Model not fitted. Call train() first.")
        return self.model.get_learned_distributions()

    def save_model(self, filepath: str) -> None:
        """Save the trained Gaussian Copula model."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        self.model.save(filepath)
        if self.verbose:
            print(f"Gaussian Copula model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load a trained Gaussian Copula model."""
        from sdv.single_table import GaussianCopulaSynthesizer
        self.model = GaussianCopulaSynthesizer.load(filepath)
        self._fitted = True
        if self.verbose:
            print(f"Gaussian Copula model loaded from {filepath}")


class EnsembleSynthesizer:
    """
    Ensemble synthesizer combining CTGAN, TVAE, and Gaussian Copula.

    Combines outputs via:
    - Numerical columns: Weighted average
    - Categorical columns: Majority voting
    - Binary columns: Weighted average, rounded to 0/1

    This approach leverages:
    - CTGAN: Best at capturing complex conditional distributions
    - TVAE: Good at maintaining overall data structure with VAE latent space
    - Gaussian Copula: Fast, excellent at preserving marginal distributions
    """

    def __init__(self,
                 # CTGAN parameters
                 ctgan_epochs: int = 300,
                 ctgan_batch_size: int = 500,
                 # TVAE parameters
                 tvae_epochs: int = 300,
                 tvae_batch_size: int = 500,
                 # Gaussian Copula parameters
                 gc_default_distribution: str = 'beta',
                 # Ensemble parameters
                 weights: Dict[str, float] = None,
                 enable_models: Dict[str, bool] = None,
                 verbose: bool = True,
                 cuda: bool = True):
        """
        Initialize Ensemble Synthesizer.

        Args:
            ctgan_epochs: Training epochs for CTGAN
            ctgan_batch_size: Batch size for CTGAN
            tvae_epochs: Training epochs for TVAE
            tvae_batch_size: Batch size for TVAE
            gc_default_distribution: Default distribution for Gaussian Copula
            weights: Dict of model weights for averaging.
                Default: {'ctgan': 0.4, 'tvae': 0.35, 'gaussian_copula': 0.25}
            enable_models: Dict to enable/disable specific models.
                Default: {'ctgan': True, 'tvae': True, 'gaussian_copula': True}
            verbose: Print progress
            cuda: Use GPU for neural network models
        """
        self.verbose = verbose
        self.cuda = cuda
        self.condition_column = 'company_name'

        # Default weights
        self.weights = weights or {
            'ctgan': 0.40,
            'tvae': 0.35,
            'gaussian_copula': 0.25
        }

        # Enable/disable models
        self.enable_models = enable_models or {
            'ctgan': True,
            'tvae': True,
            'gaussian_copula': True
        }

        # Initialize synthesizers
        self.synthesizers: Dict[str, BaseSynthesizerWrapper] = {}

        if self.enable_models.get('ctgan', True):
            self.synthesizers['ctgan'] = CTGANSynthesizerWrapper(
                epochs=ctgan_epochs,
                batch_size=ctgan_batch_size,
                verbose=verbose,
                cuda=cuda
            )

        if self.enable_models.get('tvae', True):
            self.synthesizers['tvae'] = TVAESynthesizerWrapper(
                epochs=tvae_epochs,
                batch_size=tvae_batch_size,
                verbose=verbose,
                cuda=cuda
            )

        if self.enable_models.get('gaussian_copula', True):
            self.synthesizers['gaussian_copula'] = GaussianCopulaSynthesizerWrapper(
                default_distribution=gc_default_distribution,
                verbose=verbose
            )

        # Track column types for combination logic
        self.numerical_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.binary_columns: List[str] = []

        # Quality metrics for each model
        self.model_quality_scores: Dict[str, Dict] = {}

        self._fitted = False

    def train(self, data: pd.DataFrame, discrete_columns: List[str] = None,
              binary_columns: List[str] = None) -> Dict[str, float]:
        """
        Train all enabled synthesizers on the data.

        Args:
            data: Training dataframe
            discrete_columns: List of categorical column names
            binary_columns: List of binary (0/1) column names

        Returns:
            Dict with training times for each model
        """
        import time

        if self.verbose:
            print("\n" + "="*60)
            print("ENSEMBLE SYNTHESIZER: TRAINING ALL MODELS")
            print("="*60)
            print(f"Enabled models: {[k for k, v in self.enable_models.items() if v]}")

        # Store column types for combination logic
        self.binary_columns = binary_columns or []
        self.categorical_columns = discrete_columns or []
        self.numerical_columns = [
            col for col in data.columns
            if col not in self.categorical_columns
            and col not in self.binary_columns
            and data[col].dtype in ['int64', 'float64', 'int32', 'float32']
        ]

        training_times = {}

        # Train each model sequentially (to manage GPU memory)
        for name, synthesizer in self.synthesizers.items():
            if self.verbose:
                print(f"\n--- Training {name.upper()} ---")

            start_time = time.time()
            synthesizer.train(data, discrete_columns, binary_columns)
            elapsed = time.time() - start_time
            training_times[name] = elapsed

            if self.verbose:
                print(f"{name} trained in {elapsed:.2f} seconds")

            # Clear GPU cache between models
            self._clear_gpu_cache()

        self._fitted = True

        if self.verbose:
            print("\n" + "="*60)
            print("TRAINING COMPLETE")
            print(f"Total time: {sum(training_times.values()):.2f} seconds")
            print("="*60)

        return training_times

    def _clear_gpu_cache(self):
        """Clear GPU memory cache."""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def generate(self, n_samples: int, condition_column: Optional[str] = None,
                 condition_value: Optional[Any] = None, max_tries: int = 100) -> pd.DataFrame:
        """
        Generate synthetic samples by combining outputs from all models.

        Args:
            n_samples: Number of samples to generate
            condition_column: Column to condition on
            condition_value: Value to condition on
            max_tries: Max tries for conditional sampling

        Returns:
            Combined synthetic dataframe
        """
        if not self._fitted:
            raise ValueError("Ensemble not trained. Call train() first.")

        if self.verbose:
            print(f"\n=== Generating {n_samples} Samples (Ensemble) ===")

        # Generate from each model
        individual_samples = {}
        for name, synthesizer in self.synthesizers.items():
            if self.verbose:
                print(f"  Generating from {name}...")
            try:
                samples = synthesizer.generate(
                    n_samples=n_samples,
                    condition_column=condition_column,
                    condition_value=condition_value,
                    max_tries=max_tries
                )
                individual_samples[name] = samples
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: {name} generation failed: {str(e)[:50]}...")
                continue

        if not individual_samples:
            raise ValueError("All models failed to generate samples!")

        # Combine samples
        combined = self._combine_samples(individual_samples)

        if self.verbose:
            print(f"  Combined {len(combined)} samples from {len(individual_samples)} models")

        return combined

    def _combine_samples(self, individual_samples: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine samples from multiple models.

        Strategy:
        - Numerical columns: Weighted average
        - Categorical columns: Majority voting
        - Binary columns: Weighted average, rounded to 0/1
        """
        model_names = list(individual_samples.keys())

        if len(model_names) == 1:
            return individual_samples[model_names[0]]

        # Get reference structure from first model
        reference = individual_samples[model_names[0]]
        combined = pd.DataFrame(index=reference.index)

        # Normalize weights for available models
        available_weights = {k: v for k, v in self.weights.items() if k in model_names}
        total_weight = sum(available_weights.values())
        normalized_weights = {k: v/total_weight for k, v in available_weights.items()}

        for col in reference.columns:
            if col in self.categorical_columns:
                # Majority voting for categoricals
                combined[col] = self._majority_vote(individual_samples, col)
            elif col in self.binary_columns:
                # Weighted average, then round to 0/1
                combined[col] = self._weighted_average(
                    individual_samples, col, normalized_weights
                ).round().astype(int)
            elif col in self.numerical_columns:
                # Weighted average for numerical
                combined[col] = self._weighted_average(
                    individual_samples, col, normalized_weights
                )
            else:
                # Default: take from first model
                combined[col] = reference[col]

        return combined

    def _weighted_average(self, samples: Dict[str, pd.DataFrame],
                          column: str, weights: Dict[str, float]) -> pd.Series:
        """Compute weighted average for a numerical column."""
        result = pd.Series(0.0, index=list(samples.values())[0].index)
        for name, df in samples.items():
            if column in df.columns:
                result += df[column].fillna(0) * weights.get(name, 0)
        return result

    def _majority_vote(self, samples: Dict[str, pd.DataFrame], column: str) -> pd.Series:
        """Compute majority vote for a categorical column."""
        n_samples = len(list(samples.values())[0])
        votes = []

        for i in range(n_samples):
            row_votes = []
            for name, df in samples.items():
                if column in df.columns:
                    row_votes.append(df[column].iloc[i])

            if row_votes:
                counter = Counter(row_votes)
                votes.append(counter.most_common(1)[0][0])
            else:
                votes.append(None)

        return pd.Series(votes, index=list(samples.values())[0].index)

    def generate_stratified(self, company_distribution: Dict[Any, int],
                           verbose: bool = True, max_tries: int = 100) -> Tuple[pd.DataFrame, Dict]:
        """
        Generate brands with custom distribution per company using ensemble.

        Args:
            company_distribution: Dict mapping company_id -> number of brands
            verbose: Print progress
            max_tries: Max tries for conditional sampling

        Returns:
            Tuple of (combined dataframe, dict of failed companies)
        """
        if verbose:
            print(f"\n=== Ensemble Stratified Generation ===")
            print(f"  Companies: {len(company_distribution)}")
            print(f"  Total brands requested: {sum(company_distribution.values())}")

        all_synthetic = []
        failed_companies = {}

        for i, (company, n_brands) in enumerate(company_distribution.items()):
            if verbose and (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{len(company_distribution)} companies...")

            if n_brands > 0:
                try:
                    synthetic = self.generate(
                        n_samples=n_brands,
                        condition_column=self.condition_column,
                        condition_value=company,
                        max_tries=max_tries
                    )
                    all_synthetic.append(synthetic)
                except Exception as e:
                    failed_companies[company] = str(e)
                    if verbose:
                        print(f"  Warning: Failed for company {company}: {str(e)[:50]}...")

        if all_synthetic:
            combined = pd.concat(all_synthetic, ignore_index=True)
        else:
            combined = pd.DataFrame()

        if verbose:
            print(f"\nGenerated {len(combined)} total synthetic brands")
            if failed_companies:
                print(f"Failed companies: {len(failed_companies)}")

        return combined, failed_companies

    def evaluate_individual_quality(self, real_data: pd.DataFrame,
                                    n_eval_samples: int = 1000) -> Dict[str, Dict]:
        """
        Evaluate quality of each model individually.

        Returns dict with KS statistics, correlation preservation, etc.
        """
        from scipy import stats

        if self.verbose:
            print("\n=== Evaluating Individual Model Quality ===")

        quality_results = {}

        for name, synthesizer in self.synthesizers.items():
            if self.verbose:
                print(f"\nEvaluating {name}...")

            try:
                synthetic = synthesizer.generate(n_samples=n_eval_samples)
            except Exception as e:
                quality_results[name] = {'error': str(e)}
                continue

            model_metrics = {
                'ks_statistics': {},
                'mean_ks': 0.0,
                'correlation_mse': 0.0
            }

            # KS test for numerical columns
            ks_values = []
            for col in self.numerical_columns:
                if col in real_data.columns and col in synthetic.columns:
                    try:
                        stat, _ = stats.ks_2samp(
                            real_data[col].dropna(),
                            synthetic[col].dropna()
                        )
                        model_metrics['ks_statistics'][col] = stat
                        ks_values.append(stat)
                    except Exception:
                        pass

            model_metrics['mean_ks'] = np.mean(ks_values) if ks_values else 1.0

            # Correlation matrix comparison
            common_num = [c for c in self.numerical_columns
                         if c in real_data.columns and c in synthetic.columns]
            if len(common_num) > 1:
                try:
                    real_corr = real_data[common_num].corr().values
                    synth_corr = synthetic[common_num].corr().values
                    model_metrics['correlation_mse'] = np.nanmean((real_corr - synth_corr)**2)
                except Exception:
                    model_metrics['correlation_mse'] = 1.0

            quality_results[name] = model_metrics

            if self.verbose:
                print(f"  Mean KS statistic: {model_metrics['mean_ks']:.4f}")
                print(f"  Correlation MSE: {model_metrics['correlation_mse']:.4f}")

        self.model_quality_scores = quality_results
        return quality_results

    def optimize_weights(self, real_data: pd.DataFrame,
                        n_eval_samples: int = 1000) -> Dict[str, float]:
        """
        Optimize ensemble weights based on model quality metrics.

        Uses inverse of KS statistic and correlation MSE as quality indicators.
        """
        quality = self.evaluate_individual_quality(real_data, n_eval_samples)

        # Compute quality scores (lower KS and correlation MSE = better)
        scores = {}
        for name, metrics in quality.items():
            if 'error' in metrics:
                scores[name] = 0.0
            else:
                combined_error = metrics['mean_ks'] + metrics.get('correlation_mse', 0)
                scores[name] = 1.0 / (combined_error + 0.01)

        # Normalize to weights
        total = sum(scores.values())
        if total > 0:
            self.weights = {k: v/total for k, v in scores.items()}

        if self.verbose:
            print(f"\nOptimized weights: {self.weights}")

        return self.weights

    def compare_all_models(self, real_data: pd.DataFrame,
                          n_samples: int = 1000) -> pd.DataFrame:
        """
        Compare quality of all models plus ensemble output.

        Returns comparison table as DataFrame.
        """
        from scipy import stats

        results = {}

        # Evaluate each individual model
        for name, synthesizer in self.synthesizers.items():
            try:
                synthetic = synthesizer.generate(n_samples=n_samples)
                results[name] = self._compute_quality_metrics(real_data, synthetic)
            except Exception as e:
                results[name] = {'error': str(e)}

        # Evaluate ensemble output
        try:
            ensemble_synthetic = self.generate(n_samples=n_samples)
            results['ensemble'] = self._compute_quality_metrics(real_data, ensemble_synthetic)
        except Exception as e:
            results['ensemble'] = {'error': str(e)}

        # Create comparison dataframe
        comparison_data = {}
        for model, metrics in results.items():
            if 'error' not in metrics:
                comparison_data[model] = {
                    'Mean KS Statistic': metrics.get('mean_ks', np.nan),
                    'KS Pass Rate': metrics.get('ks_pass_rate', np.nan),
                    'Correlation MSE': metrics.get('correlation_mse', np.nan),
                    'Mean Relative Error': metrics.get('mean_relative_error', np.nan),
                }

        return pd.DataFrame(comparison_data).T

    def _compute_quality_metrics(self, real_data: pd.DataFrame,
                                 synthetic_data: pd.DataFrame) -> Dict:
        """Compute comprehensive quality metrics."""
        from scipy import stats

        metrics = {}

        # KS tests
        ks_values = []
        ks_pass = 0
        for col in self.numerical_columns:
            if col in real_data.columns and col in synthetic_data.columns:
                try:
                    stat, pvalue = stats.ks_2samp(
                        real_data[col].dropna(),
                        synthetic_data[col].dropna()
                    )
                    ks_values.append(stat)
                    if pvalue > 0.05:
                        ks_pass += 1
                except Exception:
                    pass

        metrics['mean_ks'] = np.mean(ks_values) if ks_values else 1.0
        metrics['ks_pass_rate'] = ks_pass / len(ks_values) if ks_values else 0.0

        # Correlation MSE
        common_num = [c for c in self.numerical_columns
                     if c in real_data.columns and c in synthetic_data.columns]
        if len(common_num) > 1:
            try:
                real_corr = real_data[common_num].corr().values
                synth_corr = synthetic_data[common_num].corr().values
                metrics['correlation_mse'] = np.nanmean((real_corr - synth_corr)**2)
            except Exception:
                metrics['correlation_mse'] = 1.0

        # Mean relative error
        mean_errors = []
        for col in self.numerical_columns:
            if col in real_data.columns and col in synthetic_data.columns:
                real_mean = real_data[col].mean()
                synth_mean = synthetic_data[col].mean()
                if real_mean != 0:
                    mean_errors.append(abs(real_mean - synth_mean) / abs(real_mean))

        metrics['mean_relative_error'] = np.mean(mean_errors) if mean_errors else 1.0

        return metrics

    def add_diversity_noise(self, synthetic_data: pd.DataFrame,
                           noise_level: float = 0.02,
                           numerical_cols: List[str] = None) -> pd.DataFrame:
        """Add Gaussian noise to numerical columns for diversity."""
        if numerical_cols is None:
            numerical_cols = self.numerical_columns

        noisy_data = synthetic_data.copy()

        for col in numerical_cols:
            if col in noisy_data.columns:
                std = noisy_data[col].std()
                if std > 0:
                    noise = np.random.normal(0, std * noise_level, len(noisy_data))
                    noisy_data[col] = noisy_data[col] + noise

                    if (synthetic_data[col] >= 0).all():
                        noisy_data[col] = noisy_data[col].clip(lower=0)

        return noisy_data

    def save_models(self, base_path: str) -> None:
        """Save all models to a directory."""
        os.makedirs(base_path, exist_ok=True)

        for name, synthesizer in self.synthesizers.items():
            model_path = os.path.join(base_path, f'{name}_model.pkl')
            synthesizer.save_model(model_path)

        # Save config
        config = {
            'weights': self.weights,
            'numerical_columns': self.numerical_columns,
            'categorical_columns': self.categorical_columns,
            'binary_columns': self.binary_columns,
            'enable_models': self.enable_models
        }
        config_path = os.path.join(base_path, 'ensemble_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        if self.verbose:
            print(f"Ensemble models saved to {base_path}")

    def load_models(self, base_path: str) -> None:
        """Load all models from a directory."""
        # Load config
        config_path = os.path.join(base_path, 'ensemble_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)

        self.weights = config['weights']
        self.numerical_columns = config['numerical_columns']
        self.categorical_columns = config['categorical_columns']
        self.binary_columns = config['binary_columns']

        # Load each model
        for name, synthesizer in self.synthesizers.items():
            model_path = os.path.join(base_path, f'{name}_model.pkl')
            if os.path.exists(model_path):
                synthesizer.load_model(model_path)

        self._fitted = True

        if self.verbose:
            print(f"Ensemble models loaded from {base_path}")


def calculate_generation_targets(data: pd.DataFrame,
                                 company_column: str = 'company_name',
                                 min_brands_per_company: int = 100,
                                 target_total_brands: int = None) -> Dict[Any, int]:
    """
    Calculate how many brands to generate for each company.

    Args:
        data: Original dataset
        company_column: Name of the company column
        min_brands_per_company: Minimum brands each company should have
        target_total_brands: Target total (overrides min_brands if set)

    Returns:
        Dictionary mapping company -> number of brands to generate
    """
    current_counts = data[company_column].value_counts().to_dict()

    if target_total_brands:
        total_current = len(data)
        brands_to_generate = target_total_brands - total_current

        if brands_to_generate <= 0:
            print(f"Already have {total_current} brands, target is {target_total_brands}")
            return {}

        generation_targets = {}
        for company, current_count in current_counts.items():
            proportion = current_count / total_current
            to_generate = int(brands_to_generate * proportion)
            generation_targets[company] = to_generate
    else:
        generation_targets = {}
        for company, current_count in current_counts.items():
            if current_count < min_brands_per_company:
                to_generate = min_brands_per_company - current_count
                generation_targets[company] = to_generate
            else:
                to_generate = max(10, int(current_count * 0.1))
                generation_targets[company] = to_generate

    print(f"\n=== Generation Targets ===")
    print(f"  Total companies: {len(generation_targets)}")
    print(f"  Total brands to generate: {sum(generation_targets.values())}")
    print(f"  Average per company: {sum(generation_targets.values()) / len(generation_targets):.1f}")

    return generation_targets
