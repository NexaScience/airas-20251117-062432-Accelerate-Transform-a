import os
import random
from typing import Tuple, Any, Dict, List

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

__all__ = [
    "set_seed",
    "get_tokenizer",
    "load_dataset",
]


# -------------------------------------------------------------
# Reproducibility helpers
# -------------------------------------------------------------

def set_seed(seed: int):
    """Set RNG seeds for python, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------------------------------------------
# Tokeniser
# -------------------------------------------------------------

def get_tokenizer(run_cfg: Dict[str, Any]):
    """Returns a Hugging-Face tokenizer based on the model name in the run config."""
    model_name = run_cfg["model"].get("name", "bert-base-uncased")
    return AutoTokenizer.from_pretrained(model_name, use_auth_token=os.getenv("HF_TOKEN"))


# -------------------------------------------------------------
# Generic helper classes & functions
# -------------------------------------------------------------

class _DictDataset(Dataset):
    """Tiny Dataset wrapper turning a dict[Tensor] into an indexable PyTorch Dataset."""

    def __init__(self, inputs: Dict[str, torch.Tensor], labels: torch.Tensor):
        self.inputs = inputs
        self.labels = labels
        self.size = labels.size(0)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.inputs.items()}
        item["labels"] = self.labels[idx]
        return item


def _build_dataloader(inputs: Dict[str, torch.Tensor], labels: torch.Tensor, batch_size: int) -> DataLoader:
    """Create a DataLoader from already-tensorised inputs and labels."""

    dataset = _DictDataset(inputs, labels)

    def _collate(batch: List[Dict[str, torch.Tensor]]):
        collated: Dict[str, torch.Tensor] = {}
        for key in batch[0].keys():
            collated[key] = torch.stack([sample[key] for sample in batch])
        return collated

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=_collate)


# -------------------------------------------------------------
# Synthetic fall-back (unit tests / offline mode)
# -------------------------------------------------------------

def _load_synthetic(run_cfg: Dict[str, Any], tokenizer, smoke_test: bool):
    """Generate a random synthetic classification problem for quick CI checks."""
    seq_len = run_cfg["dataset"].get("seq_length", 32)
    num_classes = run_cfg["dataset"].get("num_classes", 2)
    num_samples = 200 if smoke_test else run_cfg["dataset"].get("num_samples", 2000)

    input_ids = torch.randint(low=0, high=tokenizer.vocab_size, size=(num_samples, seq_len))
    attention_mask = torch.ones_like(input_ids)
    labels = torch.randint(low=0, high=num_classes, size=(num_samples,))

    split = int(0.8 * num_samples)
    train_inputs = {"input_ids": input_ids[:split], "attention_mask": attention_mask[:split]}
    val_inputs = {"input_ids": input_ids[split:], "attention_mask": attention_mask[split:]}
    train_labels, val_labels = labels[:split], labels[split:]

    train_loader = _build_dataloader(train_inputs, train_labels, run_cfg["dataset"].get("batch_size", 32))
    val_loader = _build_dataloader(val_inputs, val_labels, run_cfg["dataset"].get("batch_size", 32))
    return train_loader, val_loader


# -------------------------------------------------------------
# Real dataset loader (MRPC / SST-2 and generic text classification)
# -------------------------------------------------------------

def _detect_sent_columns(example: Dict[str, Any]):
    """Return a tuple (text_col_a, text_col_b|None) depending on the dataset schema."""
    if "sentence1" in example and "sentence2" in example:
        return "sentence1", "sentence2"
    if "sentence" in example:
        return "sentence", None
    if "text" in example:
        return "text", None
    # Fallback – try first string column
    for k, v in example.items():
        if isinstance(v, str):
            return k, None
    raise ValueError("Unable to infer text columns for the provided dataset sample.")


def _tokenise_dataset(ds, tokenizer, seq_len):
    """Tokenise the dataset in-place, keeping only input_ids/attention_mask/label."""

    sample = ds[0]
    col_a, col_b = _detect_sent_columns(sample)

    def _tok_fn(examples):
        if col_b is None:
            return tokenizer(examples[col_a], truncation=True, padding="max_length", max_length=seq_len)
        return tokenizer(examples[col_a], examples[col_b], truncation=True, padding="max_length", max_length=seq_len)

    return ds.map(_tok_fn, batched=True)


def load_dataset(run_cfg: Dict[str, Any], tokenizer, smoke_test: bool = False) -> Tuple[Any, Any]:
    """Main entry point: returns (train_loader, val_loader) for the experiment run config."""

    ds_name = run_cfg["dataset"].get("name", "SYNTHETIC_CLASSIFICATION")
    subset = run_cfg["dataset"].get("subset", None)
    seq_len = run_cfg["dataset"].get("seq_length", 128)
    batch_size = run_cfg["dataset"].get("batch_size", 32)

    # --------------- Synthetic fall-back ---------------
    if ds_name == "SYNTHETIC_CLASSIFICATION":
        return _load_synthetic(run_cfg, tokenizer, smoke_test)

    # --------------- Hugging-Face dataset ---------------
    if subset:
        raw_ds = load_dataset(ds_name, subset)
    else:
        raw_ds = load_dataset(ds_name)

    # Ensure we have a validation split. If not – create 10 % split from train.
    if "validation" not in raw_ds:
        split_ds = raw_ds["train"].train_test_split(test_size=0.1, seed=run_cfg.get("seed", 42))
        raw_ds["train"], raw_ds["validation"] = split_ds["train"], split_ds["test"]

    # Smoke test → drastically reduce dataset size for rapid checks.
    if smoke_test:
        max_samples = min(64, len(raw_ds["train"]))
        raw_ds["train"] = raw_ds["train"].select(range(max_samples))
        raw_ds["validation"] = raw_ds["validation"].select(range(int(max_samples * 0.2)))

    # Tokenisation & formatting
    tokenised_train = _tokenise_dataset(raw_ds["train"], tokenizer, seq_len)
    tokenised_val = _tokenise_dataset(raw_ds["validation"], tokenizer, seq_len)

    # Identify label column name ("label" or "labels")
    label_col = "label" if "label" in tokenised_train.column_names else "labels"

    keep_cols = ["input_ids", "attention_mask", label_col]
    tokenised_train.set_format(type="torch", columns=keep_cols)
    tokenised_val.set_format(type="torch", columns=keep_cols)

    def _collate(batch):
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        labels = torch.tensor([item[label_col] for item in batch])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    train_loader = DataLoader(tokenised_train, batch_size=batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(tokenised_val, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    return train_loader, val_loader