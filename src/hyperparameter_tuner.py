"""
Hyperparameter tuning module for CTGAN and DistilGPT2.
Uses grid search and random search for optimal model configuration.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import itertools
import json
import os
from datetime import datetime


class CTGANHyperparameterTuner:
    """
    Hyperparameter tuning for CTGAN model.
    Tests different combinations of epochs, batch_size, and architectural parameters.
    """

    def __init__(self, train_data: pd.DataFrame, val_data: pd.DataFrame):
        """
        Initialize the tuner.

        Args:
            train_data: Training dataset
            val_data: Validation dataset
        """
        self.train_data = train_data
        self.val_data = val_data
        self.results = []

    def get_search_space(self, search_type: str = 'grid') -> Dict:
        """
        Define hyperparameter search space.

        Args:
            search_type: 'grid' for grid search, 'random' for random sampling

        Returns:
            Dictionary of hyperparameter options
        """
        if search_type == 'grid':
            # Grid search: test all combinations (use fewer options for speed)
            return {
                'epochs': [100, 200, 300],
                'batch_size': [250, 500, 750],
                'generator_dim': [(128, 128), (256, 256)],
                'discriminator_dim': [(128, 128), (256, 256)],
                'generator_lr': [2e-4],
                'discriminator_lr': [2e-4],
            }
        else:  # random search
            # Random search: sample from ranges
            return {
                'epochs': [50, 100, 150, 200, 250, 300],
                'batch_size': [128, 256, 500, 750, 1000],
                'generator_dim': [(128, 128), (256, 256), (512, 512)],
                'discriminator_dim': [(128, 128), (256, 256), (512, 512)],
                'generator_lr': [1e-4, 2e-4, 5e-4],
                'discriminator_lr': [1e-4, 2e-4, 5e-4],
            }

    def evaluate_config(self, config: Dict) -> Dict:
        """
        Train and evaluate a single hyperparameter configuration.

        Args:
            config: Dictionary of hyperparameters

        Returns:
            Dictionary with configuration and evaluation metrics
        """
        from tabular_gan import TabularBrandGAN
        from evaluator import BrandDataEvaluator

        print(f"\n{'='*60}")
        print(f"Testing configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        print(f"{'='*60}")

        # Initialize model with config
        model = TabularBrandGAN(
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            verbose=False  # Reduce noise during tuning
        )

        # Train
        start_time = datetime.now()
        try:
            model.train(self.train_data)

            # Generate validation samples
            n_samples = min(500, len(self.val_data))
            synthetic_val = model.generate(n_samples=n_samples)

            # Evaluate
            evaluator = BrandDataEvaluator()

            # Get numerical columns
            numerical_cols = self.val_data.select_dtypes(include=[np.number]).columns.tolist()

            # Distribution comparison (KS tests)
            ks_results = evaluator.compare_distributions(
                self.val_data,
                synthetic_val,
                numerical_cols
            )

            # Calculate metrics
            avg_ks_pvalue = np.mean([v['pvalue'] for v in ks_results.values()])
            passing_ks = sum(1 for v in ks_results.values() if v['pvalue'] > 0.05)
            ks_pass_rate = passing_ks / len(ks_results) if ks_results else 0

            # Correlation preservation
            _, _ = evaluator.compare_correlations(
                self.val_data,
                synthetic_val,
                numerical_cols
            )
            correlation_diff = evaluator.results.get('correlation_diff', np.inf)

            # Combined score (higher is better)
            # Weight: KS pass rate (60%), correlation preservation (40%)
            quality_score = (ks_pass_rate * 0.6) + ((1 - min(correlation_diff, 1)) * 0.4)

            training_time = (datetime.now() - start_time).total_seconds()

            result = {
                'config': config,
                'quality_score': quality_score,
                'ks_pass_rate': ks_pass_rate,
                'avg_ks_pvalue': avg_ks_pvalue,
                'correlation_diff': correlation_diff,
                'training_time_seconds': training_time,
                'success': True
            }

            print(f"\nResults:")
            print(f"  Quality Score: {quality_score:.4f}")
            print(f"  KS Pass Rate: {ks_pass_rate:.4f} ({passing_ks}/{len(ks_results)})")
            print(f"  Avg KS p-value: {avg_ks_pvalue:.4f}")
            print(f"  Correlation Diff: {correlation_diff:.4f}")
            print(f"  Training Time: {training_time:.1f}s")

        except Exception as e:
            print(f"Configuration failed with error: {e}")
            result = {
                'config': config,
                'quality_score': 0.0,
                'error': str(e),
                'success': False
            }

        return result

    def grid_search(self, max_trials: Optional[int] = None) -> Tuple[Dict, List[Dict]]:
        """
        Perform grid search over hyperparameter space.

        Args:
            max_trials: Maximum number of configurations to test (None = all)

        Returns:
            Tuple of (best_config, all_results)
        """
        print("\n" + "="*60)
        print("STARTING GRID SEARCH")
        print("="*60)

        search_space = self.get_search_space('grid')

        # Generate all combinations
        keys = list(search_space.keys())
        values = list(search_space.values())
        combinations = list(itertools.product(*values))

        print(f"\nTotal configurations: {len(combinations)}")
        if max_trials and max_trials < len(combinations):
            print(f"Limiting to {max_trials} trials")
            combinations = combinations[:max_trials]

        # Test each configuration
        for idx, combo in enumerate(combinations, 1):
            config = dict(zip(keys, combo))
            print(f"\n\nConfiguration {idx}/{len(combinations)}")

            result = self.evaluate_config(config)
            self.results.append(result)

        # Find best configuration
        successful_results = [r for r in self.results if r.get('success', False)]
        if not successful_results:
            raise ValueError("No successful configurations found!")

        best_result = max(successful_results, key=lambda x: x['quality_score'])

        print("\n" + "="*60)
        print("GRID SEARCH COMPLETE")
        print("="*60)
        print(f"\nBest Configuration:")
        for key, value in best_result['config'].items():
            print(f"  {key}: {value}")
        print(f"\nBest Quality Score: {best_result['quality_score']:.4f}")

        return best_result['config'], self.results

    def random_search(self, n_trials: int = 10) -> Tuple[Dict, List[Dict]]:
        """
        Perform random search over hyperparameter space.

        Args:
            n_trials: Number of random configurations to test

        Returns:
            Tuple of (best_config, all_results)
        """
        print("\n" + "="*60)
        print("STARTING RANDOM SEARCH")
        print("="*60)
        print(f"\nNumber of trials: {n_trials}")

        search_space = self.get_search_space('random')

        # Sample random configurations
        for trial in range(n_trials):
            config = {
                key: np.random.choice(values)
                for key, values in search_space.items()
            }

            print(f"\n\nTrial {trial + 1}/{n_trials}")
            result = self.evaluate_config(config)
            self.results.append(result)

        # Find best configuration
        successful_results = [r for r in self.results if r.get('success', False)]
        if not successful_results:
            raise ValueError("No successful configurations found!")

        best_result = max(successful_results, key=lambda x: x['quality_score'])

        print("\n" + "="*60)
        print("RANDOM SEARCH COMPLETE")
        print("="*60)
        print(f"\nBest Configuration:")
        for key, value in best_result['config'].items():
            print(f"  {key}: {value}")
        print(f"\nBest Quality Score: {best_result['quality_score']:.4f}")

        return best_result['config'], self.results

    def save_results(self, filepath: str):
        """
        Save tuning results to JSON.

        Args:
            filepath: Path to save results
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Convert to JSON-serializable format
        results_to_save = []
        for result in self.results:
            result_copy = result.copy()
            # Convert tuples to lists for JSON
            if 'config' in result_copy:
                for key, value in result_copy['config'].items():
                    if isinstance(value, tuple):
                        result_copy['config'][key] = list(value)
            results_to_save.append(result_copy)

        with open(filepath, 'w') as f:
            json.dump(results_to_save, f, indent=2)

        print(f"\nResults saved to: {filepath}")

    def plot_results(self):
        """
        Visualize hyperparameter tuning results.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        successful_results = [r for r in self.results if r.get('success', False)]
        if not successful_results:
            print("No successful results to plot")
            return

        df = pd.DataFrame([{
            'epochs': r['config']['epochs'],
            'batch_size': r['config']['batch_size'],
            'quality_score': r['quality_score'],
            'ks_pass_rate': r['ks_pass_rate'],
            'correlation_diff': r['correlation_diff'],
            'training_time': r['training_time_seconds']
        } for r in successful_results])

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # 1. Quality score by epochs
        axes[0, 0].scatter(df['epochs'], df['quality_score'], alpha=0.6, s=100)
        axes[0, 0].set_xlabel('Epochs')
        axes[0, 0].set_ylabel('Quality Score')
        axes[0, 0].set_title('Quality Score vs Epochs')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Quality score by batch size
        axes[0, 1].scatter(df['batch_size'], df['quality_score'], alpha=0.6, s=100, c='coral')
        axes[0, 1].set_xlabel('Batch Size')
        axes[0, 1].set_ylabel('Quality Score')
        axes[0, 1].set_title('Quality Score vs Batch Size')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Training time vs quality
        axes[0, 2].scatter(df['training_time'], df['quality_score'], alpha=0.6, s=100, c='green')
        axes[0, 2].set_xlabel('Training Time (seconds)')
        axes[0, 2].set_ylabel('Quality Score')
        axes[0, 2].set_title('Quality vs Training Time')
        axes[0, 2].grid(True, alpha=0.3)

        # 4. KS pass rate distribution
        axes[1, 0].hist(df['ks_pass_rate'], bins=15, alpha=0.7, color='steelblue', edgecolor='black')
        axes[1, 0].set_xlabel('KS Test Pass Rate')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of KS Pass Rates')

        # 5. Correlation diff distribution
        axes[1, 1].hist(df['correlation_diff'], bins=15, alpha=0.7, color='coral', edgecolor='black')
        axes[1, 1].set_xlabel('Correlation Difference')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Correlation Diffs')

        # 6. Best configurations bar chart
        df_sorted = df.sort_values('quality_score', ascending=False).head(5)
        axes[1, 2].barh(range(len(df_sorted)), df_sorted['quality_score'], color='purple', alpha=0.7)
        axes[1, 2].set_yticks(range(len(df_sorted)))
        axes[1, 2].set_yticklabels([f"Config {i+1}" for i in range(len(df_sorted))])
        axes[1, 2].set_xlabel('Quality Score')
        axes[1, 2].set_title('Top 5 Configurations')
        axes[1, 2].invert_yaxis()

        plt.suptitle('Hyperparameter Tuning Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()


class GPT2HyperparameterTuner:
    """
    Hyperparameter tuning for DistilGPT2 brand name generation.
    """

    def __init__(self, brands_df: pd.DataFrame):
        """
        Initialize the tuner.

        Args:
            brands_df: Dataframe with brand_name, company_name, industry_name
        """
        self.brands_df = brands_df
        self.results = []

    def get_search_space(self) -> Dict:
        """
        Define hyperparameter search space for GPT-2.

        Returns:
            Dictionary of hyperparameter options
        """
        return {
            'epochs': [2, 3, 5],
            'batch_size': [4, 8, 16],
            'learning_rate': [3e-5, 5e-5, 7e-5],
            'temperature': [0.7, 0.8, 0.9, 1.0],  # For generation
        }

    def evaluate_config(self, config: Dict, n_test_samples: int = 50) -> Dict:
        """
        Train and evaluate a GPT-2 configuration.

        Args:
            config: Dictionary of hyperparameters
            n_test_samples: Number of brand names to generate for evaluation

        Returns:
            Dictionary with configuration and evaluation metrics
        """
        from brand_name_generator import BrandNameGenerator

        print(f"\n{'='*60}")
        print(f"Testing GPT-2 configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        print(f"{'='*60}")

        try:
            # Initialize and fine-tune
            generator = BrandNameGenerator(model_name='distilgpt2')
            generator.prepare_model()

            start_time = datetime.now()
            generator.fine_tune(
                self.brands_df,
                epochs=config['epochs'],
                batch_size=config['batch_size'],
                learning_rate=config['learning_rate']
            )
            training_time = (datetime.now() - start_time).total_seconds()

            # Evaluate generation quality
            test_companies = self.brands_df[['company_name', 'industry_name']].drop_duplicates().head(10)
            generated_names = []

            for _, row in test_companies.iterrows():
                names = generator.generate_brand_names(
                    company_name=row['company_name'],
                    industry_name=row['industry_name'],
                    n_names=n_test_samples // len(test_companies),
                    temperature=config['temperature']
                )
                generated_names.extend(names)

            # Metrics
            uniqueness = len(set(generated_names)) / len(generated_names) if generated_names else 0
            avg_length = np.mean([len(name) for name in generated_names]) if generated_names else 0
            success_rate = len([n for n in generated_names if n and len(n) > 0]) / n_test_samples

            # Quality score (higher is better)
            quality_score = (uniqueness * 0.4) + (success_rate * 0.4) + ((avg_length / 20) * 0.2)

            result = {
                'config': config,
                'quality_score': quality_score,
                'uniqueness': uniqueness,
                'avg_name_length': avg_length,
                'success_rate': success_rate,
                'training_time_seconds': training_time,
                'sample_names': generated_names[:10],
                'success': True
            }

            print(f"\nResults:")
            print(f"  Quality Score: {quality_score:.4f}")
            print(f"  Uniqueness: {uniqueness:.4f}")
            print(f"  Avg Name Length: {avg_length:.1f}")
            print(f"  Success Rate: {success_rate:.4f}")
            print(f"  Sample Names: {generated_names[:5]}")

        except Exception as e:
            print(f"Configuration failed: {e}")
            result = {
                'config': config,
                'quality_score': 0.0,
                'error': str(e),
                'success': False
            }

        return result

    def grid_search(self, max_trials: Optional[int] = None) -> Tuple[Dict, List[Dict]]:
        """
        Perform grid search for GPT-2 hyperparameters.

        Args:
            max_trials: Maximum trials (None = all)

        Returns:
            Tuple of (best_config, all_results)
        """
        print("\n" + "="*60)
        print("STARTING GPT-2 GRID SEARCH")
        print("="*60)

        search_space = self.get_search_space()

        # Generate combinations (excluding temperature - used at generation time)
        train_params = ['epochs', 'batch_size', 'learning_rate']
        gen_params = ['temperature']

        train_space = {k: v for k, v in search_space.items() if k in train_params}
        keys = list(train_space.keys()) + gen_params
        values = list(train_space.values()) + [search_space['temperature']]
        combinations = list(itertools.product(*values))

        print(f"\nTotal configurations: {len(combinations)}")
        if max_trials and max_trials < len(combinations):
            print(f"Limiting to {max_trials} trials")
            combinations = combinations[:max_trials]

        for idx, combo in enumerate(combinations, 1):
            config = dict(zip(keys, combo))
            print(f"\n\nConfiguration {idx}/{len(combinations)}")

            result = self.evaluate_config(config)
            self.results.append(result)

        # Find best
        successful_results = [r for r in self.results if r.get('success', False)]
        if not successful_results:
            raise ValueError("No successful configurations!")

        best_result = max(successful_results, key=lambda x: x['quality_score'])

        print("\n" + "="*60)
        print("GPT-2 GRID SEARCH COMPLETE")
        print("="*60)
        print(f"\nBest Configuration:")
        for key, value in best_result['config'].items():
            print(f"  {key}: {value}")
        print(f"\nBest Quality Score: {best_result['quality_score']:.4f}")

        return best_result['config'], self.results

    def save_results(self, filepath: str):
        """Save results to JSON."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to: {filepath}")
