import os
import random
from typing import Tuple, Any, Dict, List

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import AutoTokenizer

__all__ = [
    "set_seed",
    "get_tokenizer",
    "load_dataset",
]

# -------------------------------------------------------------------------------------
# Reproducibility helpers
# -------------------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------------------------------------------------------------------------
# Tokeniser
# -------------------------------------------------------------------------------------

def get_tokenizer(run_cfg: Dict[str, Any]):
    """Instantiate a Hugging-Face tokenizer matching the model name contained in run_cfg."""
    model_name = run_cfg["model"].get("name", "bert-base-uncased")
    return AutoTokenizer.from_pretrained(model_name, use_auth_token=os.getenv("HF_TOKEN"))

# -------------------------------------------------------------------------------------
# Generic helpers for building PyTorch DataLoaders
# -------------------------------------------------------------------------------------

def _build_dataloader(inputs: Dict[str, torch.Tensor], labels: torch.Tensor, batch_size: int) -> DataLoader:
    """Convert a dict of tensors + label tensor into a DataLoader with an explicit collate_fn."""
    feature_keys: List[str] = list(inputs.keys())  # maintain order for TensorDataset
    tensors: List[torch.Tensor] = [inputs[k] for k in feature_keys] + [labels]
    dataset = TensorDataset(*tensors)

    def _collate(xs):
        cols = list(zip(*xs))
        feature_tensors = {k: torch.stack(cols[i]) for i, k in enumerate(feature_keys)}
        label_tensor = torch.stack(cols[-1])
        feature_tensors["labels"] = label_tensor
        return feature_tensors

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=_collate)

# -------------------------------------------------------------------------------------
# Synthetic fallback (used only when no HF dataset requested)
# -------------------------------------------------------------------------------------

def _load_synthetic(run_cfg: Dict[str, Any], tokenizer, smoke_test: bool):
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

    batch_size = run_cfg["dataset"].get("batch_size", 32)
    train_loader = _build_dataloader(train_inputs, train_labels, batch_size)
    val_loader = _build_dataloader(val_inputs, val_labels, batch_size)
    return train_loader, val_loader

# -------------------------------------------------------------------------------------
# Hugging-Face text classification loader
# -------------------------------------------------------------------------------------

def _detect_text_fields(example: Dict[str, Any]):
    """Return a tuple (primary_text_field, secondary_text_field_or_None)."""
    candidates_single = [
        "text",
        "sentence",
        "content",
        "review",
    ]
    candidates_pair_first = [
        "sentence1",
        "text1",
        "premise",
        "question1",
    ]
    candidates_pair_second = [
        "sentence2",
        "text2",
        "hypothesis",
        "question2",
    ]

    for k in candidates_single:
        if k in example:
            return k, None
    for k1, k2 in zip(candidates_pair_first, candidates_pair_second):
        if k1 in example and k2 in example:
            return k1, k2
    # Fallback to the very first str field found
    str_fields = [k for k, v in example.items() if isinstance(v, str)]
    if len(str_fields) == 0:
        raise ValueError("Could not locate text fields in dataset example: keys=%s" % list(example.keys()))
    return str_fields[0], None


def _load_hf_text_classification(run_cfg: Dict[str, Any], tokenizer, smoke_test: bool):
    ds_name = run_cfg["dataset"]["name"]
    subset = run_cfg["dataset"].get("subset", None)
    dataset = load_dataset(ds_name, subset) if subset else load_dataset(ds_name)

    # Determine text field names from a sample example
    sample_example = dataset["train"][0]
    txt_a, txt_b = _detect_text_fields(sample_example)

    max_len = run_cfg["dataset"].get("seq_length", 128)

    def tok_fn(batch):
        if txt_b is not None:
            enc = tokenizer(batch[txt_a], batch[txt_b], truncation=True, padding="max_length", max_length=max_len)
        else:
            enc = tokenizer(batch[txt_a], truncation=True, padding="max_length", max_length=max_len)
        enc["labels"] = batch["label"] if "label" in batch else batch.get("labels", 0)
        return enc

    dataset = dataset.map(tok_fn, batched=True, remove_columns=dataset["train"].column_names)
    dataset.set_format(type="torch")  # exposes torch tensors directly

    # If no validation split, create an 80/20 random split from train
    if "validation" not in dataset:
        total = len(dataset["train"])
        val_count = int(0.2 * total)
        train_count = total - val_count
        dataset["train"], dataset["validation"] = random_split(dataset["train"], [train_count, val_count])

    # Smoke-test downsizing to only a handful of samples to speed-up CI
    if smoke_test:
        dataset["train"] = dataset["train"].select(range(min(64, len(dataset["train"]))))
        dataset["validation"] = dataset["validation"].select(range(min(64, len(dataset["validation"]))))

    batch_size = run_cfg["dataset"].get("batch_size", 32)

    def collate_hf(batch):
        return {
            "input_ids": torch.stack([x["input_ids"] for x in batch]),
            "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
            "labels": torch.tensor([x["labels"] for x in batch]),
        }

    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True, collate_fn=collate_hf)
    val_loader = DataLoader(dataset["validation"], batch_size=batch_size, shuffle=False, collate_fn=collate_hf)
    return train_loader, val_loader

# -------------------------------------------------------------------------------------
# Public entry-point
# -------------------------------------------------------------------------------------

def load_dataset(run_cfg: Dict[str, Any], tokenizer, smoke_test: bool = False) -> Tuple[Any, Any]:
    """Generate (train_loader, val_loader) from run_cfg.

    Supports:
      • Any Hugging-Face text classification dataset with fields (sentence1/sentence2 or text/label, etc.)
      • A built-in synthetic classification dataset for unit tests / smoke tests.
    """
    ds_name = run_cfg["dataset"].get("name", "SYNTHETIC_CLASSIFICATION")

    # Synthetic fall-back
    if ds_name.upper() == "SYNTHETIC_CLASSIFICATION":
        return _load_synthetic(run_cfg, tokenizer, smoke_test)

    # Natural language datasets (classification)
    return _load_hf_text_classification(run_cfg, tokenizer, smoke_test)