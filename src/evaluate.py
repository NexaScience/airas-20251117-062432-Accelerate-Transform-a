import argparse
import json
import os
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rich.console import Console

sns.set(style="whitegrid")
console = Console()

FIG_DIR_NAME = "images"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate and compare all experiment variations.")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory containing all run sub-directories.")
    parser.add_argument("--config-path", type=str, required=True, help="Path to the YAML configuration file used for the run set.")
    return parser.parse_args()


def load_all_results(results_dir: str) -> List[Dict[str, Any]]:
    records = []
    for run_id in os.listdir(results_dir):
        run_path = os.path.join(results_dir, run_id, "results.json")
        if not os.path.isfile(run_path):
            continue
        with open(run_path, "r") as f:
            records.append(json.load(f))
    return records


def save_fig(fig, results_dir: str, filename: str):
    img_dir = os.path.join(results_dir, FIG_DIR_NAME)
    os.makedirs(img_dir, exist_ok=True)
    path = os.path.join(img_dir, filename)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", format="pdf")
    console.print(f"[bold cyan]Saved figure:[/] {path}")


def plot_training_curves(df_epochs: pd.DataFrame, metric: str, results_dir: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=df_epochs, x="epoch", y=metric, hue="run_id", marker="o", ax=ax)
    for run_id, sub in df_epochs.groupby("run_id"):
        best_point = sub.iloc[-1]
        ax.annotate(f"{best_point[metric]:.3f}", (best_point["epoch"], best_point[metric]))
    ax.set_title(f"{metric} over epochs")
    save_fig(fig, results_dir, f"{metric}.pdf")


def plot_bar(df_final: pd.DataFrame, metric: str, results_dir: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df_final, x="run_id", y=metric, palette="deep", ax=ax)
    for idx, row in df_final.iterrows():
        ax.text(idx, row[metric] * 1.01, f"{row[metric]:.2f}", ha="center")
    ax.set_title(metric)
    save_fig(fig, results_dir, f"{metric}.pdf")


def main():
    args = parse_args()
    records = load_all_results(args.results_dir)

    if not records:
        console.print("[red]No result files found. Evaluation aborted.")
        return

    # Flatten epoch metrics
    epoch_rows = []
    final_rows = []
    for rec in records:
        run_id = rec["run_id"]
        for epoch_entry in rec["epoch_metrics"]:
            epoch_rows.append({"run_id": run_id, **epoch_entry})
        final = rec["final_metrics"]
        final_rows.append({"run_id": run_id, **final})

    df_epochs = pd.DataFrame(epoch_rows)
    df_final = pd.DataFrame(final_rows)

    # Generate required figures
    plot_training_curves(df_epochs, "train_loss", args.results_dir)
    plot_training_curves(df_epochs, "val_accuracy", args.results_dir)
    plot_bar(df_final, "peak_memory_mb", args.results_dir)
    plot_bar(df_final, "throughput_seq_per_sec", args.results_dir)

    # Structured comparison output
    comparison = df_final.set_index("run_id").to_dict(orient="index")
    console.rule("[bold green]Aggregated Comparison Results")
    console.print_json(data=comparison)


if __name__ == "__main__":
    main()