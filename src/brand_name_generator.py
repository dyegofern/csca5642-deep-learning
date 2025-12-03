"""
Brand name generator using fine-tuned DistilGPT2.
Generates realistic brand names conditioned on company and industry with quality controls.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Set
import os
import torch
import re
from collections import Counter
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
    Generate brand names using fine-tuned DistilGPT2 with quality controls.
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
        self.generated_names = set()  # Track uniqueness
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

    @staticmethod
    def _clean_brand_name(raw_name: str) -> Optional[str]:
        """
        Clean and validate a generated brand name.

        Args:
            raw_name: Raw generated text

        Returns:
            Cleaned brand name or None if invalid
        """
        if not raw_name:
            return None

        # Extract just the brand name part (after "Brand:")
        if "Brand:" in raw_name:
            raw_name = raw_name.split("Brand:")[-1]

        # Stop at newline, pipe, or "Company:"
        raw_name = re.split(r'[\n|]|Company:|Industry:', raw_name)[0]

        # Remove extra whitespace
        raw_name = raw_name.strip()

        # Reject if empty
        if not raw_name or len(raw_name) < 2:
            return None

        # Detect repetition patterns (like "Smart Choice Smart Choice Smart Choice...")
        words = raw_name.split()
        if len(words) > 3:
            # Check if the same word/phrase repeats more than 2 times
            word_counts = Counter(words)
            max_count = max(word_counts.values())
            if max_count > 3:  # Word appears more than 3 times
                # Try to extract the non-repetitive part
                unique_words = []
                seen = set()
                for word in words:
                    if word not in seen or len(seen) < 3:
                        unique_words.append(word)
                        seen.add(word)
                    if len(unique_words) >= 5:  # Cap at 5 words
                        break
                raw_name = ' '.join(unique_words)

        # Limit to reasonable length (2-50 characters)
        if len(raw_name) > 50:
            # Try to truncate at a natural boundary
            raw_name = raw_name[:50]
            # Cut at last space
            if ' ' in raw_name:
                raw_name = raw_name.rsplit(' ', 1)[0]

        # Remove trailing punctuation except !, ?, '
        raw_name = re.sub(r'[,.\-:;]+$', '', raw_name)

        # Reject if too short after cleaning
        if len(raw_name) < 2:
            return None

        # Reject if it's just numbers or special characters
        if not re.search(r'[a-zA-Z]', raw_name):
            return None

        # Reject if it contains too many special characters
        special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s\-\'&!]', raw_name)) / len(raw_name)
        if special_char_ratio > 0.3:
            return None

        return raw_name

    def generate_brand_names(self, company_name: str, industry_name: str,
                            n_names: int = 5, temperature: float = 0.7,
                            max_length: int = 40,
                            require_unique: bool = True) -> List[str]:
        """
        Generate brand names with quality controls.

        Args:
            company_name: Name of the company
            industry_name: Industry name
            n_names: Number of unique names to generate
            temperature: Sampling temperature (0.6-0.8 recommended, lower = more conservative)
            max_length: Maximum generation length
            require_unique: Only return names not previously generated

        Returns:
            List of generated brand names
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call prepare_model() or fine_tune() first.")

        prompt = f"Company: {company_name} | Industry: {industry_name} | Brand:"

        generated_names = []
        attempts = 0
        max_attempts = n_names * 5  # Try up to 5x to get enough unique names

        while len(generated_names) < n_names and attempts < max_attempts:
            attempts += 1

            # Encode prompt
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

            # Generate with stricter controls
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_length=len(input_ids[0]) + max_length,  # Relative to prompt length
                    min_length=len(input_ids[0]) + 3,  # At least a few tokens
                    temperature=temperature,
                    do_sample=True,
                    top_k=50,
                    top_p=0.92,  # Slightly more conservative
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.3,  # Penalize repetition
                    no_repeat_ngram_size=3,  # Prevent 3-gram repetition
                )

            # Decode
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

            # Clean the brand name
            brand_name = self._clean_brand_name(generated_text)

            if brand_name:
                # Check uniqueness
                if require_unique and brand_name.lower() in {n.lower() for n in self.generated_names}:
                    continue

                # Check not already in this batch
                if brand_name.lower() not in {n.lower() for n in generated_names}:
                    generated_names.append(brand_name)
                    if require_unique:
                        self.generated_names.add(brand_name.lower())

        return generated_names

    def generate_for_dataframe(self, synthetic_df: pd.DataFrame,
                              n_names_per_brand: int = 1,
                              temperature: float = 0.7,
                              verbose: bool = True) -> pd.DataFrame:
        """
        Generate brand names for a dataframe with quality controls.

        Args:
            synthetic_df: Dataframe with company_name and industry_name
            n_names_per_brand: Number of candidate names to try per brand (will pick best)
            temperature: Sampling temperature
            verbose: Print progress

        Returns:
            Dataframe with brand_name column added
        """
        if verbose:
            print(f"\n=== Generating Brand Names for {len(synthetic_df)} Brands ===")

        generated_names = []
        failed_count = 0

        for idx, row in synthetic_df.iterrows():
            company = str(row['company_name'])
            industry = str(row['industry_name'])

            # Generate candidate names
            candidates = self.generate_brand_names(
                company_name=company,
                industry_name=industry,
                n_names=max(3, n_names_per_brand),  # Generate at least 3 candidates
                temperature=temperature,
                require_unique=True
            )

            # Pick the first valid name, or generate a fallback
            if candidates:
                brand_name = candidates[0]
            else:
                # Fallback: simple generated name
                brand_name = f"{company[:10].strip()}-{idx % 1000:03d}"
                failed_count += 1

            generated_names.append(brand_name)

            if verbose and (idx + 1) % 100 == 0:
                print(f"  Generated {idx + 1}/{len(synthetic_df)} names... (Failed: {failed_count})")

        synthetic_df['brand_name'] = generated_names

        if verbose:
            print(f"\n✓ Generated {len(generated_names)} brand names")
            print(f"  Unique names: {len(set(generated_names))}")
            print(f"  Failed/fallback: {failed_count}")
            print(f"  Success rate: {(1 - failed_count/len(generated_names))*100:.1f}%")

        return synthetic_df

    def reset_uniqueness_tracker(self):
        """Reset the set of generated names (useful for new generation sessions)."""
        self.generated_names.clear()
        print("Uniqueness tracker reset")

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
