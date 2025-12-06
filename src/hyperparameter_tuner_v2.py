"""
Hyperparameter Tuner V2 - Optuna-based optimization for ensemble synthesizers.

This module provides advanced hyperparameter tuning using Optuna's TPE sampler
for the tabular ensemble synthesizers (CTGAN, TVAE, Gaussian Copula).

Features:
- Optuna-based Bayesian optimization (TPE sampler)
- Multi-objective optimization support
- Automatic hyperparameter persistence (save/load)
- Comprehensive evaluation metrics (KS test, correlation preservation)
- Progress tracking and visualization
"""

import os
import json
import gc
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats


class HyperparameterTunerV2:
    """
    Optuna-based hyperparameter tuner for tabular ensemble synthesizers.

    This class provides automated hyperparameter optimization for CTGAN, TVAE,
    and Gaussian Copula ensemble synthesizers using Optuna's TPE (Tree-structured
    Parzen Estimator) sampler.

    Attributes:
        train_data: Training DataFrame
        discrete_cols: List of categorical column names
        binary_cols: List of binary column names
        numerical_cols: List of numerical column names
        best_params: Best hyperparameters found (after tuning)
        study: Optuna study object (after tuning)

    Example:
        >>> tuner = HyperparameterTunerV2(train_df, discrete_cols, binary_cols)
        >>> best_params = tuner.tune(n_trials=20, timeout=3600)
        >>> tuner.save(filepath='best_hyperparameters.json')
    """

    def __init__(
        self,
        train_data: pd.DataFrame,
        discrete_cols: List[str],
        binary_cols: List[str],
        eval_sample_size: int = 1000,
        gen_sample_size: int = 500,
        verbose: bool = True
    ):
        """
        Initialize the hyperparameter tuner.

        Args:
            train_data: Training DataFrame with all features
            discrete_cols: List of categorical/discrete column names
            binary_cols: List of binary (0/1) column names
            eval_sample_size: Number of samples to use for training during tuning
            gen_sample_size: Number of samples to generate for evaluation
            verbose: Whether to print progress messages
        """
        self.train_data = train_data
        self.discrete_cols = discrete_cols
        self.binary_cols = binary_cols
        self.eval_sample_size = min(eval_sample_size, len(train_data))
        self.gen_sample_size = gen_sample_size
        self.verbose = verbose

        # Identify numerical columns
        self.numerical_cols = [
            col for col in train_data.columns
            if col not in discrete_cols and col not in binary_cols
        ]

        # Results storage
        self.best_params: Optional[Dict] = None
        self.study = None
        self.all_trials: List[Dict] = []

        if self.verbose:
            print(f"HyperparameterTunerV2 initialized:")
            print(f"  Training samples: {len(train_data)}")
            print(f"  Eval sample size: {self.eval_sample_size}")
            print(f"  Numerical columns: {len(self.numerical_cols)}")
            print(f"  Discrete columns: {len(discrete_cols)}")
            print(f"  Binary columns: {len(binary_cols)}")

    @staticmethod
    def get_search_space() -> Dict[str, Any]:
        """
        Define the hyperparameter search space.

        Returns:
            Dictionary describing the search space for each hyperparameter
        """
        return {
            'ctgan_epochs': {'type': 'int', 'low': 100, 'high': 500, 'step': 50},
            'tvae_epochs': {'type': 'int', 'low': 100, 'high': 500, 'step': 50},
            'batch_size': {'type': 'categorical', 'choices': [250, 500, 750, 1000]},
            'embedding_dim': {'type': 'categorical', 'choices': [64, 128, 256]},
            'generator_dim': {'type': 'categorical', 'choices': [(128, 128), (256, 256), (128, 256), (256, 128)]},
            'discriminator_dim': {'type': 'categorical', 'choices': [(128, 128), (256, 256), (128, 256), (256, 128)]},
            'weight_ctgan': {'type': 'float', 'low': 0.1, 'high': 0.6},
            'weight_tvae': {'type': 'float', 'low': 0.1, 'high': 0.6},
            # Gaussian Copula weight is computed as 1 - ctgan - tvae
        }

    def _compute_quality_score(
        self,
        synthetic_data: pd.DataFrame,
        real_data: pd.DataFrame
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute a comprehensive quality score for synthetic data.

        Args:
            synthetic_data: Generated synthetic DataFrame
            real_data: Real training DataFrame

        Returns:
            Tuple of (overall_score, metrics_dict)
        """
        metrics = {}

        # 1. KS Test - Distribution matching
        ks_scores = []
        for col in self.numerical_cols:
            if col in synthetic_data.columns and col in real_data.columns:
                try:
                    real_vals = real_data[col].dropna()
                    synth_vals = synthetic_data[col].dropna()
                    if len(real_vals) > 0 and len(synth_vals) > 0:
                        stat, _ = stats.ks_2samp(real_vals, synth_vals)
                        ks_scores.append(stat)
                except Exception:
                    pass

        mean_ks = np.mean(ks_scores) if ks_scores else 1.0
        metrics['mean_ks_statistic'] = mean_ks

        # 2. Correlation preservation
        common_cols = [c for c in self.numerical_cols
                      if c in synthetic_data.columns and c in real_data.columns]

        if len(common_cols) > 1:
            try:
                real_corr = real_data[common_cols].corr().values
                synth_corr = synthetic_data[common_cols].corr().values

                # Handle NaN in correlation matrices
                real_corr = np.nan_to_num(real_corr, nan=0.0)
                synth_corr = np.nan_to_num(synth_corr, nan=0.0)

                corr_mse = np.mean((real_corr - synth_corr) ** 2)
            except Exception:
                corr_mse = 1.0
        else:
            corr_mse = 1.0

        metrics['correlation_mse'] = corr_mse

        # 3. Mean preservation
        mean_diffs = []
        for col in self.numerical_cols:
            if col in synthetic_data.columns and col in real_data.columns:
                try:
                    real_mean = real_data[col].mean()
                    synth_mean = synthetic_data[col].mean()
                    if abs(real_mean) > 1e-10:
                        diff = abs(real_mean - synth_mean) / abs(real_mean)
                        mean_diffs.append(min(diff, 2.0))  # Cap at 200%
                except Exception:
                    pass

        mean_mean_diff = np.mean(mean_diffs) if mean_diffs else 1.0
        metrics['mean_preservation_error'] = mean_mean_diff

        # 4. Variance preservation
        std_diffs = []
        for col in self.numerical_cols:
            if col in synthetic_data.columns and col in real_data.columns:
                try:
                    real_std = real_data[col].std()
                    synth_std = synthetic_data[col].std()
                    if abs(real_std) > 1e-10:
                        diff = abs(real_std - synth_std) / abs(real_std)
                        std_diffs.append(min(diff, 2.0))  # Cap at 200%
                except Exception:
                    pass

        mean_std_diff = np.mean(std_diffs) if std_diffs else 1.0
        metrics['variance_preservation_error'] = mean_std_diff

        # Combined score (lower is better)
        # Weights: KS (40%), Correlation (30%), Mean (15%), Variance (15%)
        overall_score = (
            mean_ks * 0.40 +
            corr_mse * 0.30 +
            mean_mean_diff * 0.15 +
            mean_std_diff * 0.15
        )

        metrics['overall_score'] = overall_score

        return overall_score, metrics

    def _create_objective(self) -> Callable:
        """
        Create the Optuna objective function.

        Returns:
            Objective function for Optuna optimization
        """
        # Import here to avoid circular imports and allow optional dependency
        try:
            from tabular_gan_v2 import EnsembleSynthesizer
        except ImportError:
            raise ImportError(
                "tabular_gan_v2 module not found. Ensure it's in your Python path."
            )

        try:
            import torch
            cuda_available = torch.cuda.is_available()
        except ImportError:
            cuda_available = False

        def objective(trial) -> float:
            """Optuna objective function."""
            # Sample hyperparameters
            ctgan_epochs = trial.suggest_int('ctgan_epochs', 100, 500, step=50)
            tvae_epochs = trial.suggest_int('tvae_epochs', 100, 500, step=50)
            batch_size = trial.suggest_categorical('batch_size', [250, 500, 750, 1000])
            embedding_dim = trial.suggest_categorical('embedding_dim', [64, 128, 256])
            generator_dim = trial.suggest_categorical(
                'generator_dim',
                ['(128, 128)', '(256, 256)', '(128, 256)', '(256, 128)']
            )
            discriminator_dim = trial.suggest_categorical(
                'discriminator_dim',
                ['(128, 128)', '(256, 256)', '(128, 256)', '(256, 128)']
            )

            # Ensemble weights
            w_ctgan = trial.suggest_float('weight_ctgan', 0.1, 0.6)
            w_tvae = trial.suggest_float('weight_tvae', 0.1, 0.6)
            w_gc = 1.0 - w_ctgan - w_tvae

            # Validate weights
            if w_gc < 0.05 or w_gc > 0.5:
                return float('inf')  # Invalid configuration

            # Parse dimension tuples
            gen_dim = eval(generator_dim)
            disc_dim = eval(discriminator_dim)

            try:
                # Create ensemble with sampled hyperparameters
                ensemble = EnsembleSynthesizer(
                    ctgan_epochs=ctgan_epochs,
                    ctgan_batch_size=batch_size,
                    tvae_epochs=tvae_epochs,
                    tvae_batch_size=batch_size,
                    gc_default_distribution='beta',
                    weights={
                        'ctgan': w_ctgan,
                        'tvae': w_tvae,
                        'gaussian_copula': w_gc
                    },
                    verbose=False,
                    cuda=cuda_available
                )

                # Train on sample for speed
                train_sample = self.train_data.sample(
                    n=self.eval_sample_size,
                    random_state=42
                )

                ensemble.train(
                    data=train_sample,
                    discrete_columns=self.discrete_cols,
                    binary_columns=self.binary_cols
                )

                # Generate synthetic data
                synthetic = ensemble.generate(n_samples=self.gen_sample_size)

                # Evaluate quality
                score, metrics = self._compute_quality_score(synthetic, train_sample)

                # Store trial info
                trial.set_user_attr('metrics', metrics)

                # Cleanup
                del ensemble
                gc.collect()
                if cuda_available:
                    import torch
                    torch.cuda.empty_cache()

                return score

            except Exception as e:
                if self.verbose:
                    print(f"Trial failed: {e}")
                return float('inf')

        return objective

    def tune(
        self,
        n_trials: int = 20,
        timeout: Optional[int] = 3600,
        seed: int = 42,
        show_progress_bar: bool = True
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization using Optuna.

        Args:
            n_trials: Maximum number of trials to run
            timeout: Maximum time in seconds (None for no limit)
            seed: Random seed for reproducibility
            show_progress_bar: Whether to show progress bar

        Returns:
            Dictionary of best hyperparameters found
        """
        try:
            import optuna
            from optuna.samplers import TPESampler
        except ImportError:
            raise ImportError(
                "Optuna not found. Install with: pip install optuna"
            )

        # Suppress Optuna's default logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        if self.verbose:
            print("\n" + "=" * 60)
            print("HYPERPARAMETER TUNING WITH OPTUNA")
            print("=" * 60)
            print(f"Trials: {n_trials}")
            print(f"Timeout: {timeout}s" if timeout else "Timeout: None")
            print(f"Seed: {seed}")
            print("=" * 60)

        # Create study
        sampler = TPESampler(seed=seed)
        self.study = optuna.create_study(
            direction='minimize',
            sampler=sampler,
            study_name='ensemble_synthesizer_tuning'
        )

        # Create objective
        objective = self._create_objective()

        # Run optimization
        self.study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=show_progress_bar,
            catch=(Exception,)
        )

        # Extract best parameters
        best_trial = self.study.best_trial
        best_params = best_trial.params

        # Parse and structure the best parameters
        self.best_params = {
            'ctgan_epochs': best_params['ctgan_epochs'],
            'tvae_epochs': best_params['tvae_epochs'],
            'batch_size': best_params['batch_size'],
            'embedding_dim': best_params['embedding_dim'],
            'generator_dim': list(eval(best_params['generator_dim'])),
            'discriminator_dim': list(eval(best_params['discriminator_dim'])),
            'ensemble_weights': {
                'ctgan': best_params['weight_ctgan'],
                'tvae': best_params['weight_tvae'],
                'gaussian_copula': 1.0 - best_params['weight_ctgan'] - best_params['weight_tvae']
            },
            'best_score': best_trial.value,
            'n_trials': len(self.study.trials),
            'tuning_timestamp': datetime.now().isoformat()
        }

        # Add metrics if available
        if 'metrics' in best_trial.user_attrs:
            self.best_params['best_metrics'] = best_trial.user_attrs['metrics']

        if self.verbose:
            print("\n" + "=" * 60)
            print("BEST HYPERPARAMETERS FOUND")
            print("=" * 60)
            print(f"Best score: {self.best_params['best_score']:.4f}")
            print(f"CTGAN epochs: {self.best_params['ctgan_epochs']}")
            print(f"TVAE epochs: {self.best_params['tvae_epochs']}")
            print(f"Batch size: {self.best_params['batch_size']}")
            print(f"Embedding dim: {self.best_params['embedding_dim']}")
            print(f"Generator dim: {self.best_params['generator_dim']}")
            print(f"Discriminator dim: {self.best_params['discriminator_dim']}")
            print(f"Ensemble weights: {self.best_params['ensemble_weights']}")
            print("=" * 60)

        return self.best_params

    def save(self, filepath: str) -> None:
        """
        Save the best hyperparameters to a JSON file.

        Args:
            filepath: Path to save the hyperparameters
        """
        if self.best_params is None:
            raise ValueError("No hyperparameters to save. Run tune() first.")

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

        # Convert numpy types to Python types
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert(i) for i in obj]
            return obj

        params_clean = convert(self.best_params)

        with open(filepath, 'w') as f:
            json.dump(params_clean, f, indent=2)

        if self.verbose:
            print(f"Hyperparameters saved to: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> Dict[str, Any]:
        """
        Load hyperparameters from a JSON file.

        Args:
            filepath: Path to the hyperparameters file

        Returns:
            Dictionary of hyperparameters
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Hyperparameters file not found: {filepath}")

        with open(filepath, 'r') as f:
            params = json.load(f)

        return params

    def get_optimization_history(self) -> pd.DataFrame:
        """
        Get the optimization history as a DataFrame.

        Returns:
            DataFrame with trial history
        """
        if self.study is None:
            raise ValueError("No study available. Run tune() first.")

        trials_data = []
        for trial in self.study.trials:
            if trial.state.name == 'COMPLETE':
                row = {
                    'trial_number': trial.number,
                    'value': trial.value,
                    **trial.params
                }
                if 'metrics' in trial.user_attrs:
                    row.update(trial.user_attrs['metrics'])
                trials_data.append(row)

        return pd.DataFrame(trials_data)

    def plot_optimization_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot the optimization history.

        Args:
            save_path: Optional path to save the figure
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            raise ImportError("matplotlib and seaborn required for plotting")

        if self.study is None:
            raise ValueError("No study available. Run tune() first.")

        history_df = self.get_optimization_history()

        if len(history_df) == 0:
            print("No completed trials to plot.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Optimization history
        ax1 = axes[0, 0]
        ax1.plot(history_df['trial_number'], history_df['value'], 'o-', alpha=0.7)
        ax1.axhline(y=history_df['value'].min(), color='r', linestyle='--',
                    label=f'Best: {history_df["value"].min():.4f}')
        ax1.set_xlabel('Trial Number')
        ax1.set_ylabel('Objective Value (lower is better)')
        ax1.set_title('Optimization History')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Parameter importance (epochs)
        ax2 = axes[0, 1]
        if 'ctgan_epochs' in history_df.columns and 'tvae_epochs' in history_df.columns:
            ax2.scatter(history_df['ctgan_epochs'], history_df['value'],
                       label='CTGAN', alpha=0.6, s=50)
            ax2.scatter(history_df['tvae_epochs'], history_df['value'],
                       label='TVAE', alpha=0.6, s=50)
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('Objective Value')
            ax2.set_title('Epochs vs Performance')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # 3. Ensemble weights distribution
        ax3 = axes[1, 0]
        if 'weight_ctgan' in history_df.columns and 'weight_tvae' in history_df.columns:
            scatter = ax3.scatter(
                history_df['weight_ctgan'],
                history_df['weight_tvae'],
                c=history_df['value'],
                cmap='viridis_r',
                alpha=0.7,
                s=80
            )
            plt.colorbar(scatter, ax=ax3, label='Objective Value')
            ax3.set_xlabel('CTGAN Weight')
            ax3.set_ylabel('TVAE Weight')
            ax3.set_title('Ensemble Weights vs Performance')
            ax3.grid(True, alpha=0.3)

        # 4. Best trials comparison
        ax4 = axes[1, 1]
        top_n = min(10, len(history_df))
        best_trials = history_df.nsmallest(top_n, 'value')
        ax4.barh(range(len(best_trials)), best_trials['value'], color='steelblue', alpha=0.8)
        ax4.set_yticks(range(len(best_trials)))
        ax4.set_yticklabels([f"Trial {int(t)}" for t in best_trials['trial_number']])
        ax4.set_xlabel('Objective Value')
        ax4.set_title(f'Top {top_n} Trials')
        ax4.invert_yaxis()
        ax4.grid(True, alpha=0.3, axis='x')

        plt.suptitle('Hyperparameter Tuning Results', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if self.verbose:
                print(f"Plot saved to: {save_path}")

        plt.show()


def tune_ensemble_hyperparameters(
    train_data: pd.DataFrame,
    discrete_cols: List[str],
    binary_cols: List[str],
    n_trials: int = 20,
    timeout: int = 3600,
    save_path: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to tune hyperparameters for ensemble synthesizers.

    Args:
        train_data: Training DataFrame
        discrete_cols: List of categorical column names
        binary_cols: List of binary column names
        n_trials: Number of optimization trials
        timeout: Maximum time in seconds
        save_path: Optional path to save best hyperparameters
        verbose: Whether to print progress

    Returns:
        Dictionary of best hyperparameters

    Example:
        >>> best_params = tune_ensemble_hyperparameters(
        ...     train_df, discrete_cols, binary_cols,
        ...     n_trials=20, save_path='best_params.json'
        ... )
    """
    tuner = HyperparameterTunerV2(
        train_data=train_data,
        discrete_cols=discrete_cols,
        binary_cols=binary_cols,
        verbose=verbose
    )

    best_params = tuner.tune(n_trials=n_trials, timeout=timeout)

    if save_path:
        tuner.save(save_path)

    return best_params
