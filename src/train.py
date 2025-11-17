import argparse
import json
import os
import random
import time
from datetime import datetime
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from rich.console import Console
from rich.table import Table
from sklearn.metrics import accuracy_score, f1_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.preprocess import set_seed, get_tokenizer, load_dataset
from src.model import get_model

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(description="Run a single experiment variation.")
    parser.add_argument("--run-config", type=str, required=True, help="Path to a JSON file containing a single run variation configuration.")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory in which to store all run outputs.")
    parser.add_argument("--smoke-test", action="store_true", help="If set, override epochs & dataset size for faster execution.")
    return parser.parse_args()


def build_optimizer(model: nn.Module, lr: float):
    return torch.optim.AdamW(model.parameters(), lr=lr)


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**{k: v for k, v in batch.items() if k != "labels"})
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            preds.append(torch.argmax(logits, dim=-1).cpu())
            labels.append(batch["labels"].cpu())
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return acc, f1


def main():
    args = parse_args()

    with open(args.run_config, "r") as f:
        run_cfg: Dict[str, Any] = json.load(f)

    run_id = run_cfg["run_id"]
    results_dir = os.path.join(args.results_dir, run_id)
    os.makedirs(results_dir, exist_ok=True)

    # Structured run description printed before numerical data (requirement)
    console.rule(f"[bold cyan]Experiment Description — {run_id}")
    console.print_json(data=run_cfg)

    # Save run config for posterity
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(run_cfg, f, indent=2)

    # Reproducibility
    seed = run_cfg.get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = get_tokenizer(run_cfg)
    train_loader, val_loader = load_dataset(run_cfg, tokenizer, smoke_test=args.smoke_test)

    model = get_model(run_cfg, tokenizer)
    model.to(device)

    optimizer = build_optimizer(model, lr=run_cfg["training"].get("learning_rate", 5e-5))
    scaler = GradScaler(enabled=run_cfg["training"].get("fp16", False))

    epochs = 1 if args.smoke_test else run_cfg["training"].get("epochs", 3)

    epoch_metrics = []
    global_step = 0
    start_time = time.time()
    torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss, step_count = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            with autocast(enabled=run_cfg["training"].get("fp16", False)):
                outputs = model(**{k: v for k, v in batch.items() if k != "labels"}, labels=batch["labels"])
                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            step_count += 1
            global_step += 1
            pbar.set_postfix(loss=loss.item())

        avg_train_loss = epoch_loss / max(step_count, 1)
        val_acc, val_f1 = evaluate(model, val_loader, device)

        epoch_entry = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_accuracy": val_acc,
            "val_f1": val_f1,
        }
        epoch_metrics.append(epoch_entry)

        # Structured per-epoch metrics to stdout
        console.print_json(data={"run_id": run_id, "epoch_metrics": epoch_entry})

    total_training_time = time.time() - start_time
    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / 1024 ** 2 if torch.cuda.is_available() else 0
    )
    throughput_seq_per_sec = len(train_loader.dataset) * epochs / total_training_time

    final_metrics = {
        "final_val_accuracy": epoch_metrics[-1]["val_accuracy"],
        "final_val_f1": epoch_metrics[-1]["val_f1"],
        "best_val_accuracy": max(m["val_accuracy"] for m in epoch_metrics),
        "best_val_f1": max(m["val_f1"] for m in epoch_metrics),
        "peak_memory_mb": peak_memory_mb,
        "throughput_seq_per_sec": throughput_seq_per_sec,
        "training_time_sec": total_training_time,
    }

    # Save model checkpoint
    torch.save(model.state_dict(), os.path.join(results_dir, "model.pt"))

    # Persist structured results
    results_obj = {
        "run_id": run_id,
        "config": run_cfg,
        "epoch_metrics": epoch_metrics,
        "final_metrics": final_metrics,
        "timestamp": datetime.utcnow().isoformat(),
    }

    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(results_obj, f, indent=2)

    # Print END-OF-RUN numerical data as JSON to stdout
    console.rule(f"[bold green]Final Results — {run_id}")
    console.print_json(data=results_obj)


if __name__ == "__main__":
    main()