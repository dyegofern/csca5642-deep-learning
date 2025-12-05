"""
Brand Name Generator V2 using ensemble of local LLMs.

Implements:
- GPT-2 Medium for improved text generation
- Flan-T5 for instruction-tuned generation
- Phi-2 for state-of-the-art quality (with LoRA)
- TinyLlama for efficient high-quality generation (with LoRA)
- Ensemble voting/averaging across multiple models

Optimized for Google Colab Pro (~15GB RAM, ~16GB VRAM)
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Set, Dict, Tuple, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections import Counter
import os
import json
import re
import gc
import torch
from torch.utils.data import Dataset


@dataclass
class ModelConfig:
    """Configuration for each model in the ensemble."""
    model_name: str
    model_type: str  # 'causal' or 'seq2seq'
    weight: float = 1.0
    max_length: int = 128
    use_gradient_checkpointing: bool = True
    use_lora: bool = False
    torch_dtype: str = 'float16'
    # LoRA config
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


@dataclass
class BrandCandidate:
    """A brand name candidate with metadata."""
    name: str
    source_model: str
    confidence: float
    votes: int = 0
    quality_score: float = 0.0


class BrandNameDatasetV2(Dataset):
    """
    Enhanced dataset supporting multiple model architectures.

    Supports:
    - Causal LM format: "Company: {company} | Industry: {industry} | Brand: {brand}"
    - Seq2Seq format: Input: "Generate brand for..." Target: "{brand}"
    """

    def __init__(
        self,
        brands_df: pd.DataFrame,
        tokenizer,
        max_length: int = 128,
        model_type: str = 'causal'
    ):
        """
        Initialize dataset for specific model type.

        Args:
            brands_df: DataFrame with brand_name, company_name, industry_name
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            model_type: 'causal' or 'seq2seq'
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_type = model_type
        self.examples = []

        for _, row in brands_df.iterrows():
            company = str(row['company_name'])
            industry = str(row['industry_name'])
            brand = str(row['brand_name'])

            if model_type == 'causal':
                # Causal LM format
                text = f"Company: {company} | Industry: {industry} | Brand: {brand}"
                self.examples.append({'text': text})
            else:
                # Seq2Seq format
                input_text = f"Generate a brand name for company: {company} in industry: {industry}"
                target_text = brand
                self.examples.append({
                    'input_text': input_text,
                    'target_text': target_text
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        if self.model_type == 'causal':
            encoding = self.tokenizer(
                example['text'],
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze()
            }
        else:
            # Seq2Seq
            input_encoding = self.tokenizer(
                example['input_text'],
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            target_encoding = self.tokenizer(
                example['target_text'],
                max_length=64,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids': input_encoding['input_ids'].squeeze(),
                'attention_mask': input_encoding['attention_mask'].squeeze(),
                'labels': target_encoding['input_ids'].squeeze()
            }


class BaseBrandModel(ABC):
    """Abstract base class for brand name generation models."""

    def __init__(self, config: ModelConfig, verbose: bool = True):
        self.config = config
        self.verbose = verbose
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_fine_tuned = False

    @abstractmethod
    def load_model(self) -> None:
        """Load model and tokenizer from HuggingFace."""
        pass

    @abstractmethod
    def fine_tune(
        self,
        brands_df: pd.DataFrame,
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        output_dir: str = './models'
    ) -> None:
        """Fine-tune model on brand data."""
        pass

    @abstractmethod
    def generate(
        self,
        company_name: str,
        industry_name: str,
        n_candidates: int = 5,
        temperature: float = 0.7
    ) -> List[BrandCandidate]:
        """Generate brand name candidates with confidence scores."""
        pass

    def save(self, output_dir: str) -> None:
        """Save fine-tuned model."""
        if self.model is None:
            raise ValueError("No model to save.")
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        if self.verbose:
            print(f"Model saved to {output_dir}")

    def load(self, model_dir: str) -> None:
        """Load fine-tuned model."""
        pass

    def _clean_brand_name(self, raw_name: str) -> Optional[str]:
        """Clean and validate a generated brand name."""
        if not raw_name:
            return None

        # Extract just the brand name part
        if "Brand:" in raw_name:
            raw_name = raw_name.split("Brand:")[-1]

        # Stop at newline, pipe, or other markers
        raw_name = re.split(r'[\n|]|Company:|Industry:|<|>', raw_name)[0]
        raw_name = raw_name.strip()

        if not raw_name or len(raw_name) < 2:
            return None

        # Detect repetition patterns
        words = raw_name.split()
        if len(words) > 3:
            word_counts = Counter(words)
            max_count = max(word_counts.values())
            if max_count > 3:
                unique_words = []
                seen = set()
                for word in words:
                    if word not in seen or len(seen) < 3:
                        unique_words.append(word)
                        seen.add(word)
                    if len(unique_words) >= 5:
                        break
                raw_name = ' '.join(unique_words)

        # Limit length
        if len(raw_name) > 50:
            raw_name = raw_name[:50]
            if ' ' in raw_name:
                raw_name = raw_name.rsplit(' ', 1)[0]

        # Remove trailing punctuation
        raw_name = re.sub(r'[,.\-:;]+$', '', raw_name)

        if len(raw_name) < 2:
            return None

        if not re.search(r'[a-zA-Z]', raw_name):
            return None

        special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s\-\'&!]', raw_name)) / len(raw_name)
        if special_char_ratio > 0.3:
            return None

        return raw_name

    def _clear_gpu_cache(self):
        """Clear GPU memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class GPT2BrandModel(BaseBrandModel):
    """GPT-2 family models (distilgpt2, gpt2, gpt2-medium, gpt2-large)."""

    def load_model(self) -> None:
        """Load GPT-2 model with memory optimizations."""
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        if self.verbose:
            print(f"Loading {self.config.model_name}...")

        self.tokenizer = GPT2Tokenizer.from_pretrained(self.config.model_name)
        # Load in FP32 - let Trainer handle FP16 during training
        self.model = GPT2LMHeadModel.from_pretrained(self.config.model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

        if self.config.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(self.device)
        if self.verbose:
            print(f"Model loaded on {self.device}")

    def fine_tune(
        self,
        brands_df: pd.DataFrame,
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        output_dir: str = './models/gpt2_brand'
    ) -> None:
        """Fine-tune GPT-2 on brand data."""
        from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

        if self.model is None:
            self.load_model()

        if self.verbose:
            print(f"\n=== Fine-tuning {self.config.model_name} ===")
            print(f"Training on {len(brands_df)} examples")

        dataset = BrandNameDatasetV2(brands_df, self.tokenizer, model_type='causal')
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=2,
            learning_rate=learning_rate,
            logging_steps=50,
            save_steps=500,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            warmup_steps=100,
            prediction_loss_only=True,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
        )

        trainer.train()
        self.is_fine_tuned = True
        self.save(output_dir)

    def generate(
        self,
        company_name: str,
        industry_name: str,
        n_candidates: int = 5,
        temperature: float = 0.7
    ) -> List[BrandCandidate]:
        """Generate brand name candidates."""
        if self.model is None:
            raise ValueError("Model not loaded.")

        prompt = f"Company: {company_name} | Industry: {industry_name} | Brand:"
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

        candidates = []
        attempts = 0
        max_attempts = n_candidates * 3

        while len(candidates) < n_candidates and attempts < max_attempts:
            attempts += 1

            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_length=len(input_ids[0]) + 40,
                    min_length=len(input_ids[0]) + 3,
                    temperature=temperature,
                    do_sample=True,
                    top_k=50,
                    top_p=0.92,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.3,
                    no_repeat_ngram_size=3,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            generated_text = self.tokenizer.decode(output.sequences[0], skip_special_tokens=True)
            brand_name = self._clean_brand_name(generated_text)

            if brand_name and brand_name.lower() not in {c.name.lower() for c in candidates}:
                # Calculate confidence from scores
                confidence = 0.7  # Default confidence
                if hasattr(output, 'scores') and output.scores:
                    probs = torch.softmax(output.scores[-1], dim=-1)
                    confidence = float(probs.max())

                candidates.append(BrandCandidate(
                    name=brand_name,
                    source_model=self.config.model_name,
                    confidence=confidence
                ))

        return candidates

    def load(self, model_dir: str) -> None:
        """Load fine-tuned GPT-2 model."""
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        self.model = GPT2LMHeadModel.from_pretrained(model_dir)
        self.model.to(self.device)
        self.is_fine_tuned = True


class T5BrandModel(BaseBrandModel):
    """T5/Flan-T5 models for text-to-text brand generation."""

    def load_model(self) -> None:
        """Load T5 model."""
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        if self.verbose:
            print(f"Loading {self.config.model_name}...")

        self.tokenizer = T5Tokenizer.from_pretrained(self.config.model_name)
        # Load in FP32 - let Trainer handle FP16 during training
        self.model = T5ForConditionalGeneration.from_pretrained(self.config.model_name)

        if self.config.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(self.device)
        if self.verbose:
            print(f"Model loaded on {self.device}")

    def fine_tune(
        self,
        brands_df: pd.DataFrame,
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 3e-4,
        output_dir: str = './models/t5_brand'
    ) -> None:
        """Fine-tune T5 for conditional generation."""
        from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

        if self.model is None:
            self.load_model()

        if self.verbose:
            print(f"\n=== Fine-tuning {self.config.model_name} ===")
            print(f"Training on {len(brands_df)} examples")

        dataset = BrandNameDatasetV2(brands_df, self.tokenizer, model_type='seq2seq')
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)

        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=2,
            learning_rate=learning_rate,
            logging_steps=50,
            save_steps=500,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            warmup_steps=100,
            predict_with_generate=True,
            generation_max_length=32,
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
        )

        trainer.train()
        self.is_fine_tuned = True
        self.save(output_dir)

    def generate(
        self,
        company_name: str,
        industry_name: str,
        n_candidates: int = 5,
        temperature: float = 0.7
    ) -> List[BrandCandidate]:
        """Generate brand names using encoder-decoder."""
        if self.model is None:
            raise ValueError("Model not loaded.")

        input_text = f"Generate a brand name for company: {company_name} in industry: {industry_name}"
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)

        candidates = []

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=32,
                num_return_sequences=min(n_candidates * 2, 10),
                temperature=temperature,
                do_sample=True,
                top_k=50,
                top_p=0.92,
                repetition_penalty=1.2,
                output_scores=True,
                return_dict_in_generate=True,
            )

        for i, seq in enumerate(outputs.sequences):
            brand_name = self.tokenizer.decode(seq, skip_special_tokens=True)
            brand_name = self._clean_brand_name(brand_name)

            if brand_name and brand_name.lower() not in {c.name.lower() for c in candidates}:
                confidence = 0.75
                candidates.append(BrandCandidate(
                    name=brand_name,
                    source_model=self.config.model_name,
                    confidence=confidence
                ))

            if len(candidates) >= n_candidates:
                break

        return candidates

    def load(self, model_dir: str) -> None:
        """Load fine-tuned T5 model."""
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        self.tokenizer = T5Tokenizer.from_pretrained(model_dir)
        self.model = T5ForConditionalGeneration.from_pretrained(model_dir)
        self.model.to(self.device)
        self.is_fine_tuned = True


class PhiBrandModel(BaseBrandModel):
    """Microsoft Phi-2 model with LoRA for efficient fine-tuning."""

    def load_model(self) -> None:
        """Load Phi-2 with memory optimizations."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.verbose:
            print(f"Loading {self.config.model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map='auto'
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.verbose:
            print(f"Model loaded")

    def fine_tune(
        self,
        brands_df: pd.DataFrame,
        epochs: int = 2,
        batch_size: int = 2,
        learning_rate: float = 2e-5,
        output_dir: str = './models/phi2_brand'
    ) -> None:
        """Fine-tune Phi-2 with LoRA."""
        from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

        if self.model is None:
            self.load_model()

        # Apply LoRA
        if self.config.use_lora:
            try:
                from peft import LoraConfig, get_peft_model, TaskType

                lora_config = LoraConfig(
                    r=self.config.lora_r,
                    lora_alpha=self.config.lora_alpha,
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                    lora_dropout=self.config.lora_dropout,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM
                )
                self.model = get_peft_model(self.model, lora_config)
                if self.verbose:
                    print("LoRA applied successfully")
                    self.model.print_trainable_parameters()
            except ImportError:
                if self.verbose:
                    print("PEFT not available, training without LoRA")

        if self.verbose:
            print(f"\n=== Fine-tuning {self.config.model_name} ===")

        dataset = BrandNameDatasetV2(brands_df, self.tokenizer, model_type='causal')
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=8,
            learning_rate=learning_rate,
            logging_steps=50,
            save_steps=500,
            fp16=True,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
            warmup_ratio=0.1,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
        )

        trainer.train()
        self.is_fine_tuned = True
        self.save(output_dir)

    def generate(
        self,
        company_name: str,
        industry_name: str,
        n_candidates: int = 5,
        temperature: float = 0.7
    ) -> List[BrandCandidate]:
        """Generate with Phi-2."""
        if self.model is None:
            raise ValueError("Model not loaded.")

        prompt = f"Company: {company_name} | Industry: {industry_name} | Brand:"
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.model.device)

        candidates = []
        attempts = 0
        max_attempts = n_candidates * 3

        while len(candidates) < n_candidates and attempts < max_attempts:
            attempts += 1

            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_length=len(input_ids[0]) + 40,
                    temperature=temperature,
                    do_sample=True,
                    top_k=50,
                    top_p=0.92,
                    repetition_penalty=1.3,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            brand_name = self._clean_brand_name(generated_text)

            if brand_name and brand_name.lower() not in {c.name.lower() for c in candidates}:
                candidates.append(BrandCandidate(
                    name=brand_name,
                    source_model=self.config.model_name,
                    confidence=0.8
                ))

        return candidates


class TinyLlamaBrandModel(BaseBrandModel):
    """TinyLlama 1.1B model with LoRA."""

    def load_model(self) -> None:
        """Load TinyLlama model."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.verbose:
            print(f"Loading {self.config.model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,
            device_map='auto'
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.verbose:
            print(f"Model loaded")

    def fine_tune(
        self,
        brands_df: pd.DataFrame,
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-5,
        output_dir: str = './models/tinyllama_brand'
    ) -> None:
        """Fine-tune TinyLlama with LoRA."""
        from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

        if self.model is None:
            self.load_model()

        # Apply LoRA
        if self.config.use_lora:
            try:
                from peft import LoraConfig, get_peft_model, TaskType

                lora_config = LoraConfig(
                    r=self.config.lora_r,
                    lora_alpha=self.config.lora_alpha,
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                    lora_dropout=self.config.lora_dropout,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM
                )
                self.model = get_peft_model(self.model, lora_config)
                if self.verbose:
                    print("LoRA applied successfully")
            except ImportError:
                if self.verbose:
                    print("PEFT not available, training without LoRA")

        if self.verbose:
            print(f"\n=== Fine-tuning {self.config.model_name} ===")

        dataset = BrandNameDatasetV2(brands_df, self.tokenizer, model_type='causal')
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            logging_steps=50,
            save_steps=500,
            fp16=True,
            gradient_checkpointing=True,
            warmup_ratio=0.1,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
        )

        trainer.train()
        self.is_fine_tuned = True
        self.save(output_dir)

    def generate(
        self,
        company_name: str,
        industry_name: str,
        n_candidates: int = 5,
        temperature: float = 0.7
    ) -> List[BrandCandidate]:
        """Generate with TinyLlama using chat format."""
        if self.model is None:
            raise ValueError("Model not loaded.")

        # Use chat template if available
        prompt = f"Company: {company_name} | Industry: {industry_name} | Brand:"
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.model.device)

        candidates = []
        attempts = 0
        max_attempts = n_candidates * 3

        while len(candidates) < n_candidates and attempts < max_attempts:
            attempts += 1

            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_length=len(input_ids[0]) + 40,
                    temperature=temperature,
                    do_sample=True,
                    top_k=50,
                    top_p=0.92,
                    repetition_penalty=1.3,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            brand_name = self._clean_brand_name(generated_text)

            if brand_name and brand_name.lower() not in {c.name.lower() for c in candidates}:
                candidates.append(BrandCandidate(
                    name=brand_name,
                    source_model=self.config.model_name,
                    confidence=0.75
                ))

        return candidates


# ============================================================
# VOTING STRATEGIES
# ============================================================

def calculate_quality_score(
    name: str,
    confidence: float,
    existing_brands: Set[str]
) -> float:
    """
    Calculate composite quality score for a brand name candidate.

    Factors:
    - Model confidence (25%)
    - Length score (20%): Optimal 5-20 chars
    - Uniqueness score (25%): Distance from existing brands
    - Character diversity (15%)
    - Phonetic quality (15%)
    """
    # Confidence score
    conf_score = min(confidence, 1.0)

    # Length score
    length = len(name)
    if 5 <= length <= 20:
        length_score = 1.0
    elif length < 5:
        length_score = length / 5
    else:
        length_score = max(0, 1 - (length - 20) / 30)

    # Uniqueness score (simple Levenshtein approximation)
    if existing_brands:
        min_distance = min(
            _simple_edit_distance(name.lower(), brand.lower())
            for brand in existing_brands
        )
        uniqueness_score = min(min_distance / 5, 1.0)
    else:
        uniqueness_score = 1.0

    # Character diversity
    diversity_score = len(set(name.lower())) / len(name) if name else 0

    # Phonetic quality (consonant-vowel ratio)
    vowels = sum(1 for c in name.lower() if c in 'aeiou')
    consonants = sum(1 for c in name.lower() if c.isalpha() and c not in 'aeiou')
    ratio = vowels / (consonants + 1)
    phonetic_score = 1.0 if 0.3 <= ratio <= 0.7 else 0.5

    # Weighted combination
    return (
        0.25 * conf_score +
        0.20 * length_score +
        0.25 * uniqueness_score +
        0.15 * diversity_score +
        0.15 * phonetic_score
    )


def _simple_edit_distance(s1: str, s2: str) -> int:
    """Simple edit distance calculation."""
    if len(s1) < len(s2):
        return _simple_edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


class BrandNameEnsemble:
    """
    Ensemble of multiple LLMs for brand name generation.

    Features:
    - Multiple model architectures (GPT-2, T5, Phi-2, TinyLlama)
    - Quality-weighted voting
    - Memory-efficient sequential model loading
    """

    DEFAULT_CONFIGS = {
        'gpt2-medium': ModelConfig(
            model_name='gpt2-medium',
            model_type='causal',
            weight=1.0,
            use_gradient_checkpointing=True
        ),
        'flan-t5-base': ModelConfig(
            model_name='google/flan-t5-base',
            model_type='seq2seq',
            weight=1.2,
            use_gradient_checkpointing=True
        ),
        'phi-2': ModelConfig(
            model_name='microsoft/phi-2',
            model_type='causal',
            weight=1.5,
            use_gradient_checkpointing=True,
            use_lora=True
        ),
        'tinyllama': ModelConfig(
            model_name='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
            model_type='causal',
            weight=1.3,
            use_gradient_checkpointing=True,
            use_lora=True
        )
    }

    def __init__(
        self,
        model_names: Optional[List[str]] = None,
        memory_efficient: bool = True,
        verbose: bool = True
    ):
        """
        Initialize ensemble.

        Args:
            model_names: List of model names to use (None = auto-select)
            memory_efficient: Load models sequentially to save memory
            verbose: Print progress
        """
        self.model_names = model_names or ['gpt2-medium', 'flan-t5-base']
        self.memory_efficient = memory_efficient
        self.verbose = verbose

        self.models: Dict[str, BaseBrandModel] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self.generated_names: Set[str] = set()

        # Initialize model configs
        for name in self.model_names:
            if name in self.DEFAULT_CONFIGS:
                self.model_configs[name] = self.DEFAULT_CONFIGS[name]
            else:
                # Create default config for unknown model
                self.model_configs[name] = ModelConfig(
                    model_name=name,
                    model_type='causal',
                    weight=1.0
                )

    def _get_model_class(self, model_name: str) -> type:
        """Get appropriate model class."""
        if 't5' in model_name.lower() or 'flan' in model_name.lower():
            return T5BrandModel
        elif 'phi' in model_name.lower():
            return PhiBrandModel
        elif 'tinyllama' in model_name.lower() or 'llama' in model_name.lower():
            return TinyLlamaBrandModel
        else:
            return GPT2BrandModel

    def load_models(self) -> None:
        """Load all models in ensemble."""
        for name in self.model_names:
            config = self.model_configs[name]
            model_class = self._get_model_class(config.model_name)
            model = model_class(config, verbose=self.verbose)
            model.load_model()
            self.models[name] = model

            if self.memory_efficient:
                self._clear_gpu_cache()

    def fine_tune_all(
        self,
        brands_df: pd.DataFrame,
        epochs_per_model: Optional[Dict[str, int]] = None,
        output_dir: str = './models/ensemble'
    ) -> None:
        """
        Fine-tune all models sequentially (memory efficient).

        Args:
            brands_df: Training data
            epochs_per_model: Dict of model_name -> epochs (optional)
            output_dir: Base output directory
        """
        default_epochs = {'gpt2-medium': 3, 'flan-t5-base': 3, 'phi-2': 2, 'tinyllama': 3}
        epochs_per_model = epochs_per_model or default_epochs

        for name in self.model_names:
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"FINE-TUNING: {name}")
                print(f"{'='*60}")

            config = self.model_configs[name]
            model_class = self._get_model_class(config.model_name)
            model = model_class(config, verbose=self.verbose)

            epochs = epochs_per_model.get(name, 3)
            model_output = os.path.join(output_dir, name)

            model.fine_tune(
                brands_df,
                epochs=epochs,
                output_dir=model_output
            )

            # Save and unload to free memory
            if self.memory_efficient:
                del model
                self._clear_gpu_cache()

        if self.verbose:
            print(f"\nAll models fine-tuned and saved to {output_dir}")

    def generate_candidates(
        self,
        company_name: str,
        industry_name: str,
        n_per_model: int = 5,
        temperature: float = 0.7
    ) -> List[List[BrandCandidate]]:
        """Generate candidates from each model."""
        all_candidates = []

        for name, model in self.models.items():
            try:
                candidates = model.generate(
                    company_name=company_name,
                    industry_name=industry_name,
                    n_candidates=n_per_model,
                    temperature=temperature
                )
                all_candidates.append(candidates)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: {name} generation failed: {e}")
                all_candidates.append([])

        return all_candidates

    def generate_brand_names(
        self,
        company_name: str,
        industry_name: str,
        n_names: int = 5,
        temperature: float = 0.7,
        require_unique: bool = True
    ) -> List[str]:
        """
        Generate brand names using ensemble voting.

        Process:
        1. Generate candidates from each model
        2. Apply quality-weighted voting
        3. Return top-N unique names
        """
        # Generate from all models
        all_candidates = self.generate_candidates(
            company_name, industry_name,
            n_per_model=max(5, n_names),
            temperature=temperature
        )

        # Flatten and score
        scored_candidates = []
        for model_candidates in all_candidates:
            for candidate in model_candidates:
                quality = calculate_quality_score(
                    candidate.name,
                    candidate.confidence,
                    self.generated_names
                )
                candidate.quality_score = quality
                scored_candidates.append(candidate)

        # Aggregate by name (voting)
        name_scores: Dict[str, List[BrandCandidate]] = {}
        for candidate in scored_candidates:
            key = candidate.name.lower()
            if key not in name_scores:
                name_scores[key] = []
            name_scores[key].append(candidate)

        # Calculate final scores
        final_candidates = []
        model_weights = {name: self.model_configs[name].weight for name in self.model_names}
        total_weight = sum(model_weights.values())

        for name_lower, candidates in name_scores.items():
            # Vote score
            vote_weight = sum(
                model_weights.get(c.source_model.split('/')[-1], 1.0)
                for c in candidates
            )
            vote_score = vote_weight / total_weight

            # Best quality score
            best_candidate = max(candidates, key=lambda c: c.quality_score)

            # Combined score
            final_score = 0.4 * vote_score + 0.6 * best_candidate.quality_score

            final_candidates.append(BrandCandidate(
                name=best_candidate.name,
                source_model=best_candidate.source_model,
                confidence=best_candidate.confidence,
                votes=len(candidates),
                quality_score=final_score
            ))

        # Sort by final score
        final_candidates.sort(key=lambda c: c.quality_score, reverse=True)

        # Get unique names
        result = []
        for candidate in final_candidates:
            if require_unique and candidate.name.lower() in self.generated_names:
                continue
            result.append(candidate.name)
            if require_unique:
                self.generated_names.add(candidate.name.lower())
            if len(result) >= n_names:
                break

        return result

    def _clear_gpu_cache(self):
        """Clear GPU memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def save_ensemble(self, output_dir: str) -> None:
        """Save ensemble configuration."""
        os.makedirs(output_dir, exist_ok=True)
        config = {
            'model_names': self.model_names,
            'model_configs': {
                name: {
                    'model_name': cfg.model_name,
                    'model_type': cfg.model_type,
                    'weight': cfg.weight
                }
                for name, cfg in self.model_configs.items()
            }
        }
        with open(os.path.join(output_dir, 'ensemble_config.json'), 'w') as f:
            json.dump(config, f, indent=2)

    def load_ensemble(self, ensemble_dir: str) -> None:
        """Load ensemble from directory."""
        with open(os.path.join(ensemble_dir, 'ensemble_config.json'), 'r') as f:
            config = json.load(f)

        self.model_names = config['model_names']

        # Load each model
        for name in self.model_names:
            model_dir = os.path.join(ensemble_dir, name)
            if os.path.exists(model_dir):
                model_config = self.model_configs.get(name, self.DEFAULT_CONFIGS.get(name))
                if model_config:
                    model_class = self._get_model_class(model_config.model_name)
                    model = model_class(model_config, verbose=self.verbose)
                    model.load(model_dir)
                    self.models[name] = model


# ============================================================
# MAIN API CLASS (DROP-IN REPLACEMENT)
# ============================================================

class BrandNameGeneratorV2:
    """
    Improved brand name generator using ensemble of local LLMs.

    Drop-in replacement for BrandNameGenerator with enhanced quality.

    Example usage:
        generator = BrandNameGeneratorV2(
            models=['gpt2-medium', 'flan-t5-base'],
            voting_strategy='quality_weighted'
        )
        generator.fine_tune(brands_df)
        names = generator.generate_brand_names("Apple", "Technology", n_names=5)
    """

    def __init__(
        self,
        models: Optional[List[str]] = None,
        memory_efficient: bool = True,
        verbose: bool = True
    ):
        """
        Initialize the V2 generator.

        Args:
            models: List of model names to use (default: ['gpt2-medium', 'flan-t5-base'])
            memory_efficient: Sequential model loading for memory efficiency
            verbose: Print progress
        """
        self.models_list = models or ['gpt2-medium', 'flan-t5-base']
        self.ensemble = BrandNameEnsemble(
            model_names=self.models_list,
            memory_efficient=memory_efficient,
            verbose=verbose
        )
        self.verbose = verbose
        self.generated_names: Set[str] = set()

    def prepare_model(self) -> None:
        """Load the ensemble models."""
        self.ensemble.load_models()

    def fine_tune(
        self,
        brands_df: pd.DataFrame,
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        output_dir: str = './models/brand_name_generator_v2'
    ) -> None:
        """Fine-tune the ensemble on brand names."""
        epochs_per_model = {name: epochs for name in self.models_list}
        self.ensemble.fine_tune_all(
            brands_df,
            epochs_per_model=epochs_per_model,
            output_dir=output_dir
        )

    def generate_brand_names(
        self,
        company_name: str,
        industry_name: str,
        n_names: int = 5,
        temperature: float = 0.7,
        max_length: int = 40,
        require_unique: bool = True
    ) -> List[str]:
        """Generate brand names using ensemble."""
        return self.ensemble.generate_brand_names(
            company_name=company_name,
            industry_name=industry_name,
            n_names=n_names,
            temperature=temperature,
            require_unique=require_unique
        )

    def generate_for_dataframe(
        self,
        synthetic_df: pd.DataFrame,
        n_names_per_brand: int = 1,
        temperature: float = 0.7,
        verbose: bool = True
    ) -> pd.DataFrame:
        """Generate brand names for a dataframe."""
        if verbose:
            print(f"\n=== Generating Brand Names for {len(synthetic_df)} Brands ===")

        generated_names = []
        failed_count = 0

        for idx, row in synthetic_df.iterrows():
            company = str(row['company_name'])
            industry = str(row['industry_name'])

            candidates = self.generate_brand_names(
                company_name=company,
                industry_name=industry,
                n_names=max(3, n_names_per_brand),
                temperature=temperature,
                require_unique=True
            )

            if candidates:
                brand_name = candidates[0]
            else:
                brand_name = f"{company[:10].strip()}-{idx % 1000:03d}"
                failed_count += 1

            generated_names.append(brand_name)

            if verbose and (idx + 1) % 100 == 0:
                print(f"  Generated {idx + 1}/{len(synthetic_df)} names... (Failed: {failed_count})")

        synthetic_df = synthetic_df.copy()
        synthetic_df['brand_name'] = generated_names

        if verbose:
            print(f"\nGenerated {len(generated_names)} brand names")
            print(f"  Unique names: {len(set(generated_names))}")
            print(f"  Failed/fallback: {failed_count}")
            print(f"  Success rate: {(1 - failed_count/len(generated_names))*100:.1f}%")

        return synthetic_df

    def reset_uniqueness_tracker(self) -> None:
        """Reset the generated names tracker."""
        self.ensemble.generated_names.clear()
        self.generated_names.clear()
        if self.verbose:
            print("Uniqueness tracker reset")

    def save_model(self, output_dir: str) -> None:
        """Save the ensemble."""
        self.ensemble.save_ensemble(output_dir)

    def load_model(self, model_dir: str) -> None:
        """Load the ensemble."""
        self.ensemble.load_ensemble(model_dir)
