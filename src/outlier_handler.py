"""
Outlier handling and normalization strategies for extreme-value companies.
Specifically addresses cases like Amazon with billion-dollar revenues that dominate clustering.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import RobustScaler, PowerTransformer, QuantileTransformer
import matplotlib.pyplot as plt
import seaborn as sns


class OutlierHandler:
    """
    Handle extreme outliers (like Amazon's revenue) that can dominate clustering.
    Provides multiple normalization strategies and validation.
    """

    def __init__(self):
        """Initialize the outlier handler."""
        self.scalers = {}
        self.outlier_companies = []
        self.feature_stats = {}

    def identify_extreme_outliers(self, data: pd.DataFrame,
                                  numerical_cols: List[str],
                                  threshold_iqr: float = 10.0) -> Dict:
        """
        Identify extreme outliers using IQR method.

        Args:
            data: Dataset
            numerical_cols: Numerical columns to check
            threshold_iqr: Number of IQRs beyond which to flag (default: 10x IQR)

        Returns:
            Dictionary of outlier information
        """
        print("\n=== Identifying Extreme Outliers ===")

        outliers_info = {
            'features': {},
            'companies': {},
            'severity': {}
        }

        for col in numerical_cols:
            if col not in data.columns:
                continue

            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1

            # Extreme outliers: values beyond threshold * IQR
            lower_bound = Q1 - threshold_iqr * IQR
            upper_bound = Q3 + threshold_iqr * IQR

            outlier_mask = (data[col] < lower_bound) | (data[col] > upper_bound)
            n_outliers = outlier_mask.sum()

            if n_outliers > 0:
                outlier_rows = data[outlier_mask]

                outliers_info['features'][col] = {
                    'n_outliers': n_outliers,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'IQR': IQR,
                    'max_value': data[col].max(),
                    'median': data[col].median()
                }

                # Track which companies are outliers
                if 'company_name' in data.columns:
                    outlier_companies = outlier_rows['company_name'].unique()
                    for company in outlier_companies:
                        if company not in outliers_info['companies']:
                            outliers_info['companies'][company] = []
                        outliers_info['companies'][company].append({
                            'feature': col,
                            'value': data[data['company_name'] == company][col].iloc[0],
                            'severity': (data[data['company_name'] == company][col].iloc[0] - upper_bound) / IQR
                        })

                print(f"\n  {col}:")
                print(f"    Outliers: {n_outliers} / {len(data)} ({n_outliers/len(data)*100:.1f}%)")
                print(f"    Range: [{lower_bound:.2f}, {upper_bound:.2f}]")
                print(f"    Max value: {data[col].max():.2f}")
                print(f"    Median: {data[col].median():.2f}")

        # Identify companies with multiple outlier features (like Amazon)
        if outliers_info['companies']:
            print("\n  Companies with multiple outlier features:")
            multi_outlier = {k: v for k, v in outliers_info['companies'].items() if len(v) >= 2}
            for company, features in sorted(multi_outlier.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
                print(f"    {company}: {len(features)} outlier features")
                for feat_info in features:
                    print(f"      - {feat_info['feature']}: {feat_info['value']:.2f} (severity: {feat_info['severity']:.1f}x IQR)")

        self.outlier_companies = list(outliers_info['companies'].keys())
        return outliers_info

    def apply_robust_scaling(self, data: pd.DataFrame,
                            numerical_cols: List[str],
                            strategy: str = 'robust') -> pd.DataFrame:
        """
        Apply robust scaling to handle outliers.

        Args:
            data: Dataset
            numerical_cols: Columns to scale
            strategy: Scaling strategy:
                - 'robust': RobustScaler (uses median and IQR, resistant to outliers)
                - 'power': PowerTransformer (makes data more Gaussian-like)
                - 'quantile': QuantileTransformer (uniform or normal distribution)
                - 'log': Log transformation (log1p)

        Returns:
            Scaled dataframe
        """
        print(f"\n=== Applying {strategy.upper()} Scaling ===")

        data_scaled = data.copy()

        for col in numerical_cols:
            if col not in data.columns:
                continue

            if strategy == 'robust':
                # RobustScaler: uses median and IQR (resistant to outliers)
                scaler = RobustScaler()
                data_scaled[col] = scaler.fit_transform(data[[col]])
                self.scalers[col] = scaler
                print(f"  {col}: RobustScaler (median={scaler.center_[0]:.2f}, IQR={scaler.scale_[0]:.2f})")

            elif strategy == 'power':
                # PowerTransformer: makes data more Gaussian (Box-Cox/Yeo-Johnson)
                scaler = PowerTransformer(method='yeo-johnson', standardize=True)
                data_scaled[col] = scaler.fit_transform(data[[col]])
                self.scalers[col] = scaler
                print(f"  {col}: PowerTransformer (lambda={scaler.lambdas_[0]:.4f})")

            elif strategy == 'quantile':
                # QuantileTransformer: maps to uniform/normal distribution
                scaler = QuantileTransformer(output_distribution='normal', random_state=42)
                data_scaled[col] = scaler.fit_transform(data[[col]])
                self.scalers[col] = scaler
                print(f"  {col}: QuantileTransformer (n_quantiles={scaler.n_quantiles_})")

            elif strategy == 'log':
                # Log transformation (handles zeros with log1p)
                data_scaled[col] = np.log1p(data[col])
                print(f"  {col}: Log transformation (log1p)")

        return data_scaled

    def create_amazon_test_cases(self, data: pd.DataFrame) -> Dict:
        """
        Create specific test cases for Amazon and similar billion-dollar companies.

        Args:
            data: Original dataset

        Returns:
            Dictionary of test cases
        """
        print("\n=== Creating Test Cases for Extreme-Value Companies ===")

        test_cases = {}

        # Identify billion-dollar companies
        if 'revenues' in data.columns and 'company_name' in data.columns:
            high_revenue_threshold = data['revenues'].quantile(0.95)
            high_revenue_companies = data[data['revenues'] > high_revenue_threshold]['company_name'].unique()

            print(f"\nHigh-revenue companies (>{high_revenue_threshold:.0f}M):")
            for company in high_revenue_companies:
                company_data = data[data['company_name'] == company].iloc[0]
                print(f"  {company}: ${company_data['revenues']:.0f}M revenue")

                test_cases[company] = {
                    'original_metrics': company_data[['revenues', 'scope12_total', 'market_cap_billion_usd']].to_dict() if 'market_cap_billion_usd' in data.columns else {},
                    'expected_behavior': 'Should cluster with similar business models, not isolate due to revenue',
                    'clustering_expectations': {
                        'should_not_be_alone': True,
                        'min_cluster_size': 10,  # Cluster should have at least 10 companies
                        'similar_companies': []  # Will be filled during analysis
                    }
                }

        # Amazon-specific test case
        if 'company_name' in data.columns:
            amazon_data = data[data['company_name'].str.contains('Amazon', case=False, na=False)]
            if not amazon_data.empty:
                test_cases['Amazon'] = {
                    'n_brands': len(amazon_data),
                    'original_metrics': amazon_data.iloc[0].to_dict() if len(amazon_data) > 0 else {},
                    'expected_behavior': 'Should cluster with other e-commerce/tech companies, not be isolated',
                    'clustering_expectations': {
                        'should_not_be_alone': True,
                        'min_cluster_size': 10,
                        'expected_cluster_mates': ['Walmart', 'Target', 'Alibaba', 'eBay'],  # Similar business models
                    },
                    'feature_dominance_check': {
                        'revenues': 'Should not dominate clustering decision',
                        'market_cap': 'Should not dominate clustering decision',
                        'business_model': 'Should be primary clustering factor (e-commerce, online_sales=1)'
                    }
                }
                print(f"\n  Amazon Test Case Created:")
                print(f"    Number of Amazon brands: {len(amazon_data)}")
                print(f"    Revenue: ${amazon_data.iloc[0]['revenues']:.0f}M")

        self.feature_stats = test_cases
        return test_cases

    def validate_clustering_for_outliers(self, data: pd.DataFrame,
                                        cluster_labels: np.ndarray,
                                        test_cases: Dict) -> Dict:
        """
        Validate that outlier companies are not isolated in their own clusters.

        Args:
            data: Dataset with clustering results
            cluster_labels: Cluster assignments
            test_cases: Test cases from create_amazon_test_cases()

        Returns:
            Validation results
        """
        print("\n=== Validating Clustering for Outlier Companies ===")

        validation_results = {
            'passed': [],
            'failed': [],
            'warnings': []
        }

        data_with_clusters = data.copy()
        data_with_clusters['cluster'] = cluster_labels

        for company, test_case in test_cases.items():
            if 'company_name' not in data.columns:
                continue

            company_rows = data_with_clusters[data_with_clusters['company_name'].str.contains(company, case=False, na=False)]

            if company_rows.empty:
                continue

            company_cluster = company_rows.iloc[0]['cluster']
            cluster_size = (data_with_clusters['cluster'] == company_cluster).sum()

            print(f"\n  {company}:")
            print(f"    Assigned to cluster: {company_cluster}")
            print(f"    Cluster size: {cluster_size}")

            # Check if company is isolated
            is_alone = cluster_size == 1

            if is_alone:
                validation_results['failed'].append({
                    'company': company,
                    'issue': 'ISOLATED IN OWN CLUSTER',
                    'cluster': company_cluster,
                    'cluster_size': cluster_size
                })
                print(f"    ⚠️  FAILED: {company} is ALONE in cluster {company_cluster}")

            elif cluster_size < test_case.get('clustering_expectations', {}).get('min_cluster_size', 5):
                validation_results['warnings'].append({
                    'company': company,
                    'issue': 'Small cluster',
                    'cluster': company_cluster,
                    'cluster_size': cluster_size,
                    'expected_min': test_case['clustering_expectations']['min_cluster_size']
                })
                print(f"    ⚠️  WARNING: Cluster size ({cluster_size}) below expected minimum")

            else:
                # Get cluster mates
                cluster_mates = data_with_clusters[data_with_clusters['cluster'] == company_cluster]['company_name'].unique()
                cluster_mates = [c for c in cluster_mates if company.lower() not in c.lower()][:10]

                validation_results['passed'].append({
                    'company': company,
                    'cluster': company_cluster,
                    'cluster_size': cluster_size,
                    'cluster_mates': cluster_mates
                })
                print(f"    ✓ PASSED: Clustered with {cluster_size-1} other companies")
                print(f"    Sample cluster mates: {', '.join(cluster_mates[:5])}")

        # Summary
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        print(f"  Passed: {len(validation_results['passed'])}")
        print(f"  Failed: {len(validation_results['failed'])}")
        print(f"  Warnings: {len(validation_results['warnings'])}")

        if validation_results['failed']:
            print("\n  ⚠️  CRITICAL: Some companies are isolated!")
            for failure in validation_results['failed']:
                print(f"    - {failure['company']}: alone in cluster {failure['cluster']}")
        else:
            print("\n  ✓ All outlier companies are properly clustered!")

        return validation_results

    def plot_outlier_analysis(self, data: pd.DataFrame,
                              numerical_cols: List[str],
                              outlier_info: Dict,
                              figsize: Tuple[int, int] = (16, 10)):
        """
        Visualize outlier distribution and impact.

        Args:
            data: Dataset
            numerical_cols: Numerical columns
            outlier_info: Output from identify_extreme_outliers()
            figsize: Figure size
        """
        outlier_features = list(outlier_info['features'].keys())[:6]  # Top 6

        if not outlier_features:
            print("No outliers to plot")
            return

        n_features = len(outlier_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_features > 1 else [axes]

        for idx, feature in enumerate(outlier_features):
            ax = axes[idx]

            # Box plot with outliers highlighted
            data_clean = data[feature].dropna()

            ax.boxplot(data_clean, vert=False)
            ax.set_xlabel(feature, fontsize=10)
            ax.set_title(f'{feature}\n({outlier_info["features"][feature]["n_outliers"]} outliers)',
                        fontsize=11, fontweight='bold')

            # Mark median and IQR bounds
            median = outlier_info['features'][feature]['median']
            upper = outlier_info['features'][feature]['upper_bound']
            ax.axvline(median, color='green', linestyle='--', label='Median', linewidth=2)
            ax.axvline(upper, color='red', linestyle='--', label='Outlier threshold', linewidth=2)
            ax.legend(fontsize=8)

        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')

        plt.suptitle('Extreme Outlier Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def compare_scaling_strategies(self, data: pd.DataFrame,
                                   numerical_cols: List[str],
                                   sample_feature: str = 'revenues') -> pd.DataFrame:
        """
        Compare different scaling strategies side-by-side.

        Args:
            data: Dataset
            numerical_cols: Numerical columns
            sample_feature: Feature to visualize (should be one with outliers)

        Returns:
            DataFrame with scaling comparison
        """
        if sample_feature not in data.columns:
            print(f"Feature {sample_feature} not found")
            return pd.DataFrame()

        print(f"\n=== Comparing Scaling Strategies for '{sample_feature}' ===")

        strategies = ['robust', 'power', 'quantile', 'log']
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        results = {}

        for idx, strategy in enumerate(strategies):
            # Apply scaling
            scaled_data = self.apply_robust_scaling(data, [sample_feature], strategy=strategy)
            scaled_values = scaled_data[sample_feature]

            # Plot
            ax = axes[idx]
            ax.hist(scaled_values, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
            ax.set_title(f'{strategy.capitalize()} Scaling', fontsize=13, fontweight='bold')
            ax.set_xlabel(f'Scaled {sample_feature}')
            ax.set_ylabel('Frequency')

            # Statistics
            results[strategy] = {
                'mean': scaled_values.mean(),
                'std': scaled_values.std(),
                'min': scaled_values.min(),
                'max': scaled_values.max(),
                'skewness': scaled_values.skew()
            }

            stats_text = f"Mean: {results[strategy]['mean']:.2f}\nStd: {results[strategy]['std']:.2f}\nSkew: {results[strategy]['skewness']:.2f}"
            ax.text(0.65, 0.95, stats_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle(f'Scaling Strategy Comparison: {sample_feature}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

        # Print comparison table
        print("\nScaling Strategy Comparison:")
        comparison_df = pd.DataFrame(results).T
        print(comparison_df.round(3))

        return comparison_df
