"""
Data loading and preprocessing module for brand dataset.
Handles loading, cleaning, and preparing data for GAN training.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


class BrandDataProcessor:
    """
    Handles all data loading, cleaning, and preprocessing operations.
    """

    def __init__(self, data_path: str):
        """
        Initialize the data processor.

        Args:
            data_path: Path to the brand_information.csv file
        """
        self.data_path = data_path
        self.df = None
        self.df_clean = None
        self.label_encoders = {}
        self.scaler = None

        # Define feature categories
        self.text_features = ['brand_name', 'company_name', 'esg_summary', 'accusation',
                             'references_and_links', 'parent_company']
        self.categorical_features = ['industry_name', 'country_of_origin',
                                    'headquarters_country', 'demographics_income_level',
                                    'demographics_geographic_reach', 'demographics_gender',
                                    'demographics_lifestyle']
        self.numerical_features = None  # Will be computed

    def load_data(self) -> pd.DataFrame:
        """Load the brand dataset."""
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_csv(self.data_path, encoding='utf-8-sig')
        print(f"Loaded {len(self.df)} brands with {len(self.df.columns)} features")
        return self.df

    def explore_data(self) -> Dict:
        """
        Perform exploratory data analysis.

        Returns:
            Dictionary with exploration statistics
        """
        if self.df is None:
            self.load_data()

        stats = {
            'total_brands': len(self.df),
            'total_features': len(self.df.columns),
            'missing_values': self.df.isnull().sum().to_dict(),
            'brands_per_company': self.df['company_name'].value_counts().to_dict(),
            'brands_per_industry': self.df['industry_name'].value_counts().to_dict(),
            'numerical_stats': self.df.describe().to_dict()
        }

        print(f"\n=== Data Exploration ===")
        print(f"Total brands: {stats['total_brands']}")
        print(f"Total features: {stats['total_features']}")
        print(f"\nTop 10 companies by brand count:")
        for company, count in list(stats['brands_per_company'].items())[:10]:
            print(f"  {company}: {count} brands")

        return stats

    def identify_feature_types(self):
        """Automatically identify numerical vs categorical features."""
        if self.df is None:
            self.load_data()

        # Get all numerical columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove ID columns and cluster assignments (we'll regenerate those)
        exclude_cols = ['id', 'brand_id', 'company_id', 'industry_id']
        self.numerical_features = [col for col in numerical_cols if col not in exclude_cols]

        print(f"\nIdentified {len(self.numerical_features)} numerical features")
        print(f"Identified {len(self.categorical_features)} categorical features")
        print(f"Identified {len(self.text_features)} text features (will be handled separately)")

    def clean_data(self, drop_text_heavy: bool = True) -> pd.DataFrame:
        """
        Clean the dataset.

        Args:
            drop_text_heavy: If True, drop text-heavy columns like esg_summary, accusation

        Returns:
            Cleaned dataframe
        """
        if self.df is None:
            self.load_data()

        print("\n=== Data Cleaning ===")
        self.df_clean = self.df.copy()

        # Identify feature types
        self.identify_feature_types()

        # Drop text-heavy columns if requested (too complex for tabular GAN)
        if drop_text_heavy:
            drop_cols = ['esg_summary', 'accusation', 'references_and_links']
            self.df_clean = self.df_clean.drop(columns=drop_cols, errors='ignore')
            print(f"Dropped text-heavy columns: {drop_cols}")

        # Handle missing values
        print("\nHandling missing values...")

        # For numerical features: fill with median
        for col in self.numerical_features:
            if col in self.df_clean.columns and self.df_clean[col].isnull().any():
                median_val = self.df_clean[col].median()
                self.df_clean[col].fillna(median_val, inplace=True)
                print(f"  Filled {col} with median: {median_val:.2f}")

        # For categorical features: fill with mode or 'Unknown'
        for col in self.categorical_features:
            if col in self.df_clean.columns and self.df_clean[col].isnull().any():
                mode_val = self.df_clean[col].mode()
                if len(mode_val) > 0:
                    self.df_clean[col].fillna(mode_val[0], inplace=True)
                else:
                    self.df_clean[col].fillna('Unknown', inplace=True)
                print(f"  Filled {col} with mode/Unknown")

        print(f"\nCleaned dataset: {len(self.df_clean)} rows, {len(self.df_clean.columns)} columns")
        return self.df_clean

    def prepare_for_gan(self, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for GAN training.

        Handles companies with only one brand by:
        1. Identifying single-brand companies
        2. Performing stratified split only on multi-brand companies
        3. Adding all single-brand companies to the training set

        Args:
            test_size: Fraction of data to use for validation

        Returns:
            Tuple of (train_df, val_df)
        """
        if self.df_clean is None:
            self.clean_data()

        print("\n=== Preparing Data for GAN ===")

        # Create a copy for GAN training (exclude brand_name, we'll generate those separately)
        gan_features = ['company_name'] + self.categorical_features + self.numerical_features
        gan_features = [col for col in gan_features if col in self.df_clean.columns]

        df_gan = self.df_clean[gan_features].copy()

        # Encode categorical features
        print("\nEncoding categorical features...")
        for col in self.categorical_features:
            if col in df_gan.columns:
                le = LabelEncoder()
                df_gan[col] = le.fit_transform(df_gan[col].astype(str))
                self.label_encoders[col] = le
                print(f"  Encoded {col}: {len(le.classes_)} unique values")

        # Encode company_name (our conditioning variable)
        le_company = LabelEncoder()
        df_gan['company_name'] = le_company.fit_transform(df_gan['company_name'].astype(str))
        self.label_encoders['company_name'] = le_company
        print(f"  Encoded company_name: {len(le_company.classes_)} companies")

        # Identify companies with only one brand (after encoding)
        company_counts = df_gan['company_name'].value_counts()
        single_brand_company_ids = company_counts[company_counts == 1].index

        # Separate single-brand companies from multi-brand companies
        df_single_brand = df_gan[df_gan['company_name'].isin(single_brand_company_ids)]
        df_multi_brand = df_gan[~df_gan['company_name'].isin(single_brand_company_ids)]

        # Initialize empty DataFrames with proper columns
        train_df = pd.DataFrame(columns=df_gan.columns)
        val_df = pd.DataFrame(columns=df_gan.columns)

        # Perform stratified split on multi-brand companies
        if not df_multi_brand.empty:
            train_multi, val_multi = train_test_split(
                df_multi_brand,
                test_size=test_size,
                random_state=42,
                stratify=df_multi_brand['company_name']
            )
            train_df = pd.concat([train_df, train_multi], ignore_index=True)
            val_df = pd.concat([val_df, val_multi], ignore_index=True)
        else:
            print("No multi-brand companies for stratified split.")

        # Add single-brand companies to the training set
        if not df_single_brand.empty:
            train_df = pd.concat([train_df, df_single_brand], ignore_index=True)
            print(f"Added {len(df_single_brand)} single-brand companies to training set.")
        else:
            print("No single-brand companies to add.")

        # Shuffle the combined training set
        train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"\nTrain set: {len(train_df)} brands")
        print(f"Validation set: {len(val_df)} brands")

        return train_df, val_df

    def get_company_brands(self, company_name: str) -> pd.DataFrame:
        """
        Get all brands for a specific company.

        Args:
            company_name: Name of the company

        Returns:
            Dataframe of brands for that company
        """
        if self.df is None:
            self.load_data()

        return self.df[self.df['company_name'] == company_name]

    def get_multi_brand_companies(self, min_brands: int = 3) -> List[str]:
        """
        Get companies with multiple brands (good for training).

        Args:
            min_brands: Minimum number of brands to consider

        Returns:
            List of company names
        """
        if self.df is None:
            self.load_data()

        brand_counts = self.df['company_name'].value_counts()
        multi_brand = brand_counts[brand_counts >= min_brands].index.tolist()

        print(f"\nFound {len(multi_brand)} companies with {min_brands}+ brands")
        return multi_brand

    def decode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Decode categorical features back to original values.

        Args:
            df: Dataframe with encoded features

        Returns:
            Dataframe with decoded features
        """
        df_decoded = df.copy()

        for col, encoder in self.label_encoders.items():
            if col in df_decoded.columns:
                # Handle out-of-range values (from GAN generation)
                df_decoded[col] = df_decoded[col].clip(0, len(encoder.classes_) - 1)
                df_decoded[col] = df_decoded[col].round().astype(int)
                df_decoded[col] = encoder.inverse_transform(df_decoded[col])

        return df_decoded
