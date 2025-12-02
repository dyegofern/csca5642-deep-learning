"""
Brand name generator using fine-tuned DistilGPT2.
Generates realistic brand names conditioned on company and industry.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
import os
import torch
from torch.utils.data import Dataset


class BrandNameDataset(Dataset):
    """Dataset for fine-tuning GPT-2 on brand names."""

    def __init__(self, brands_df: pd.DataFrame, tokenizer, max_length: int = 128):
        """
        Initialize the dataset.

        Args:
            brands_df: Dataframe with brand_name, company_name, industry_name columns
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        # Format: "Company: {company} | Industry: {industry} | Brand: {brand}"
        for _, row in brands_df.iterrows():
            text = f"Company: {row['company_name']} | Industry: {row['industry_name']} | Brand: {row['brand_name']}"
            self.examples.append(text)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }


class BrandNameGenerator:
    """
    Generate brand names using fine-tuned DistilGPT2.
    """

    def __init__(self, model_name: str = 'distilgpt2'):
        """
        Initialize the brand name generator.

        Args:
            model_name: HuggingFace model name (default: distilgpt2)
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def prepare_model(self):
        """Load the pre-trained model and tokenizer."""
        try:
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
        except ImportError:
            print("Installing transformers library...")
            import subprocess
            subprocess.check_call(['pip', 'install', 'transformers'])
            from transformers import GPT2LMHeadModel, GPT2Tokenizer

        print(f"\nLoading {self.model_name}...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)

        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

        self.model.to(self.device)
        print(f"Model loaded on {self.device}")

    def fine_tune(self, brands_df: pd.DataFrame, epochs: int = 3,
                  batch_size: int = 8, learning_rate: float = 5e-5,
                  output_dir: str = './models/brand_name_generator'):
        """
        Fine-tune the model on brand names.

        Args:
            brands_df: Dataframe with brand_name, company_name, industry_name
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            output_dir: Directory to save the model
        """
        if self.model is None:
            self.prepare_model()

        try:
            from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
        except ImportError:
            print("Installing transformers library...")
            import subprocess
            subprocess.check_call(['pip', 'install', 'transformers'])
            from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

        print("\n=== Fine-tuning Brand Name Generator ===")
        print(f"Training on {len(brands_df)} brand examples")

        # Create dataset
        dataset = BrandNameDataset(brands_df, self.tokenizer)

        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM, not masked LM
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            logging_steps=50,
            save_steps=500,
            save_total_limit=2,
            prediction_loss_only=True,
            fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
        )

        # Train
        print(f"Training for {epochs} epochs...")
        trainer.train()
        print("Fine-tuning completed!")

        # Save model
        self.save_model(output_dir)

    def generate_brand_names(self, company_name: str, industry_name: str,
                            n_names: int = 5, temperature: float = 0.8,
                            max_length: int = 50) -> List[str]:
        """
        Generate brand names for a given company and industry.

        Args:
            company_name: Name of the company
            industry_name: Industry name
            n_names: Number of names to generate
            temperature: Sampling temperature (higher = more creative)
            max_length: Maximum generation length

        Returns:
            List of generated brand names
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call prepare_model() or fine_tune() first.")

        prompt = f"Company: {company_name} | Industry: {industry_name} | Brand:"

        print(f"\nGenerating {n_names} brand names for {company_name}...")

        generated_names = []

        for i in range(n_names):
            # Encode prompt
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

            # Generate
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

            # Extract brand name (after "Brand:")
            try:
                brand_name = generated_text.split("Brand:")[-1].strip()
                # Clean up: take only the first line or until next |
                brand_name = brand_name.split('\n')[0].split('|')[0].strip()

                if brand_name and brand_name not in generated_names:
                    generated_names.append(brand_name)
            except:
                continue

        print(f"Generated {len(generated_names)} unique names: {generated_names}")
        return generated_names

    def generate_for_dataframe(self, synthetic_df: pd.DataFrame,
                              n_names_per_brand: int = 3,
                              temperature: float = 0.8) -> pd.DataFrame:
        """
        Generate brand names for a dataframe of synthetic brands.

        Args:
            synthetic_df: Dataframe with company_name and industry_name (encoded)
            n_names_per_brand: Number of candidate names to generate per brand
            temperature: Sampling temperature

        Returns:
            Dataframe with brand_name column added
        """
        print("\n=== Generating Brand Names for Synthetic Data ===")

        generated_names = []

        for idx, row in synthetic_df.iterrows():
            company = str(row['company_name'])
            industry = str(row['industry_name'])

            # Generate multiple candidates and pick the first one
            candidates = self.generate_brand_names(
                company_name=company,
                industry_name=industry,
                n_names=n_names_per_brand,
                temperature=temperature
            )

            # Pick the first valid name
            brand_name = candidates[0] if candidates else f"Brand_{idx}"
            generated_names.append(brand_name)

            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(synthetic_df)} brands...")

        synthetic_df['brand_name'] = generated_names
        print(f"\nGenerated {len(generated_names)} brand names")
        return synthetic_df

    def save_model(self, output_dir: str):
        """
        Save the fine-tuned model.

        Args:
            output_dir: Directory to save the model
        """
        if self.model is None:
            raise ValueError("No model to save.")

        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")

    def load_model(self, model_dir: str):
        """
        Load a fine-tuned model.

        Args:
            model_dir: Directory containing the saved model
        """
        try:
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
        except ImportError:
            print("Installing transformers library...")
            import subprocess
            subprocess.check_call(['pip', 'install', 'transformers'])
            from transformers import GPT2LMHeadModel, GPT2Tokenizer

        print(f"Loading model from {model_dir}...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        self.model = GPT2LMHeadModel.from_pretrained(model_dir)
        self.model.to(self.device)
        print(f"Model loaded on {self.device}")
