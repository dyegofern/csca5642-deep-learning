"""
Configuration manager for saving and loading hyperparameters.
Automatically uses best parameters from tuning unless explicitly overridden.
"""

import json
import os
from typing import Dict, Optional
from datetime import datetime


class ConfigManager:
    """
    Manage hyperparameter configurations with automatic best-parameter loading.
    """

    def __init__(self, config_dir: str = './models/configs'):
        """
        Initialize configuration manager.

        Args:
            config_dir: Directory to store configuration files
        """
        self.config_dir = config_dir
        os.makedirs(config_dir, exist_ok=True)

        self.ctgan_config_path = os.path.join(config_dir, 'ctgan_best_params.json')
        self.gpt2_config_path = os.path.join(config_dir, 'gpt2_best_params.json')
        self.scaling_config_path = os.path.join(config_dir, 'scaling_strategy.json')

    def get_default_ctgan_params(self) -> Dict:
        """Get default CTGAN parameters."""
        return {
            'epochs': 300,
            'batch_size': 500,
            'generator_dim': (256, 256),
            'discriminator_dim': (256, 256),
            'generator_lr': 2e-4,
            'discriminator_lr': 2e-4,
            'source': 'default'
        }

    def get_default_gpt2_params(self) -> Dict:
        """Get default GPT-2 parameters."""
        return {
            'epochs': 3,
            'batch_size': 8,
            'learning_rate': 5e-5,
            'temperature': 0.8,
            'source': 'default'
        }

    def get_default_scaling_strategy(self) -> Dict:
        """Get default scaling strategy."""
        return {
            'strategy': 'robust',
            'features_to_scale': ['revenues', 'scope12_total', 'market_cap_billion_usd'],
            'source': 'default'
        }

    def save_best_ctgan_params(self, params: Dict, quality_score: float,
                               tuning_method: str = 'grid_search'):
        """
        Save best CTGAN parameters from tuning.

        Args:
            params: Best parameter configuration
            quality_score: Quality score achieved
            tuning_method: How parameters were found (grid_search/random_search/manual)
        """
        config = {
            'params': params,
            'quality_score': quality_score,
            'tuning_method': tuning_method,
            'timestamp': datetime.now().isoformat(),
            'source': 'tuning'
        }

        with open(self.ctgan_config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\n✓ Best CTGAN parameters saved to: {self.ctgan_config_path}")
        print(f"  Quality Score: {quality_score:.4f}")
        print(f"  Method: {tuning_method}")

    def save_best_gpt2_params(self, params: Dict, quality_score: float,
                              tuning_method: str = 'grid_search'):
        """
        Save best GPT-2 parameters from tuning.

        Args:
            params: Best parameter configuration
            quality_score: Quality score achieved
            tuning_method: How parameters were found
        """
        config = {
            'params': params,
            'quality_score': quality_score,
            'tuning_method': tuning_method,
            'timestamp': datetime.now().isoformat(),
            'source': 'tuning'
        }

        with open(self.gpt2_config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\n✓ Best GPT-2 parameters saved to: {self.gpt2_config_path}")
        print(f"  Quality Score: {quality_score:.4f}")
        print(f"  Method: {tuning_method}")

    def save_scaling_strategy(self, strategy: str, features: list):
        """
        Save preferred scaling strategy.

        Args:
            strategy: Scaling strategy name (robust/power/quantile/log)
            features: Features to apply scaling to
        """
        config = {
            'strategy': strategy,
            'features_to_scale': features,
            'timestamp': datetime.now().isoformat(),
            'source': 'configured'
        }

        with open(self.scaling_config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\n✓ Scaling strategy saved: {strategy}")

    def load_ctgan_params(self, force_default: bool = False) -> Dict:
        """
        Load CTGAN parameters (uses tuned params if available, else default).

        Args:
            force_default: If True, ignore tuned params and use default

        Returns:
            Dictionary of CTGAN parameters
        """
        if force_default or not os.path.exists(self.ctgan_config_path):
            params = self.get_default_ctgan_params()
            print("\n📋 Using DEFAULT CTGAN parameters")
        else:
            with open(self.ctgan_config_path, 'r') as f:
                config = json.load(f)

            params = config['params']
            params['source'] = 'tuned'

            print("\n📋 Using TUNED CTGAN parameters")
            print(f"  (Quality Score: {config['quality_score']:.4f}, Method: {config['tuning_method']})")
            print(f"  (Tuned on: {config['timestamp']})")

        print(f"\n  Configuration:")
        for key, value in params.items():
            if key != 'source':
                print(f"    {key}: {value}")

        return params

    def load_gpt2_params(self, force_default: bool = False) -> Dict:
        """
        Load GPT-2 parameters (uses tuned params if available, else default).

        Args:
            force_default: If True, ignore tuned params and use default

        Returns:
            Dictionary of GPT-2 parameters
        """
        if force_default or not os.path.exists(self.gpt2_config_path):
            params = self.get_default_gpt2_params()
            print("\n📋 Using DEFAULT GPT-2 parameters")
        else:
            with open(self.gpt2_config_path, 'r') as f:
                config = json.load(f)

            params = config['params']
            params['source'] = 'tuned'

            print("\n📋 Using TUNED GPT-2 parameters")
            print(f"  (Quality Score: {config['quality_score']:.4f}, Method: {config['tuning_method']})")
            print(f"  (Tuned on: {config['timestamp']})")

        print(f"\n  Configuration:")
        for key, value in params.items():
            if key != 'source':
                print(f"    {key}: {value}")

        return params

    def load_scaling_strategy(self) -> Dict:
        """
        Load scaling strategy.

        Returns:
            Dictionary with strategy and features
        """
        if not os.path.exists(self.scaling_config_path):
            config = self.get_default_scaling_strategy()
            print("\n📋 Using DEFAULT scaling strategy: robust")
        else:
            with open(self.scaling_config_path, 'r') as f:
                config = json.load(f)
            print(f"\n📋 Using CONFIGURED scaling strategy: {config['strategy']}")

        return config

    def reset_to_defaults(self):
        """Reset all configurations to defaults (deletes tuned params)."""
        configs_to_delete = [
            self.ctgan_config_path,
            self.gpt2_config_path,
            self.scaling_config_path
        ]

        for config_path in configs_to_delete:
            if os.path.exists(config_path):
                os.remove(config_path)
                print(f"✓ Deleted: {config_path}")

        print("\n✓ All configurations reset to defaults")

    def get_all_configs(self) -> Dict:
        """
        Get all current configurations.

        Returns:
            Dictionary with all configs
        """
        return {
            'ctgan': self.load_ctgan_params(force_default=False),
            'gpt2': self.load_gpt2_params(force_default=False),
            'scaling': self.load_scaling_strategy()
        }

    def print_status(self):
        """Print status of all configurations."""
        print("\n" + "="*60)
        print("CONFIGURATION STATUS")
        print("="*60)

        # CTGAN
        if os.path.exists(self.ctgan_config_path):
            with open(self.ctgan_config_path, 'r') as f:
                ctgan_config = json.load(f)
            print(f"\n✓ CTGAN: TUNED (score: {ctgan_config['quality_score']:.4f})")
            print(f"  - Epochs: {ctgan_config['params']['epochs']}")
            print(f"  - Batch Size: {ctgan_config['params']['batch_size']}")
            print(f"  - Tuned on: {ctgan_config['timestamp']}")
        else:
            print("\n○ CTGAN: DEFAULT (not tuned yet)")

        # GPT-2
        if os.path.exists(self.gpt2_config_path):
            with open(self.gpt2_config_path, 'r') as f:
                gpt2_config = json.load(f)
            print(f"\n✓ GPT-2: TUNED (score: {gpt2_config['quality_score']:.4f})")
            print(f"  - Epochs: {gpt2_config['params']['epochs']}")
            print(f"  - Learning Rate: {gpt2_config['params']['learning_rate']}")
            print(f"  - Temperature: {gpt2_config['params']['temperature']}")
            print(f"  - Tuned on: {gpt2_config['timestamp']}")
        else:
            print("\n○ GPT-2: DEFAULT (not tuned yet)")

        # Scaling
        if os.path.exists(self.scaling_config_path):
            with open(self.scaling_config_path, 'r') as f:
                scaling_config = json.load(f)
            print(f"\n✓ Scaling: {scaling_config['strategy'].upper()}")
        else:
            print("\n○ Scaling: DEFAULT (robust)")

        print("\n" + "="*60)


# Convenience functions for notebook usage
def get_config_manager(config_dir: str = './models/configs') -> ConfigManager:
    """
    Get a configuration manager instance.

    Args:
        config_dir: Directory for config files

    Returns:
        ConfigManager instance
    """
    return ConfigManager(config_dir)


def load_best_params(run_tuning: bool = False, config_dir: str = './models/configs') -> Dict:
    """
    Load best parameters for all models.

    Args:
        run_tuning: If True, ignore saved params (will trigger tuning in notebook)
        config_dir: Directory for config files

    Returns:
        Dictionary with all parameters
    """
    config_mgr = ConfigManager(config_dir)

    if run_tuning:
        print("\n⚠️  RUN_TUNING=True: Will use default params and run hyperparameter tuning")
        return {
            'ctgan': config_mgr.get_default_ctgan_params(),
            'gpt2': config_mgr.get_default_gpt2_params(),
            'scaling': config_mgr.get_default_scaling_strategy(),
            'run_tuning': True
        }
    else:
        print("\n📋 Loading best available parameters...")
        return {
            'ctgan': config_mgr.load_ctgan_params(),
            'gpt2': config_mgr.load_gpt2_params(),
            'scaling': config_mgr.load_scaling_strategy(),
            'run_tuning': False
        }
