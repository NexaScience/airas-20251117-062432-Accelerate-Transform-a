import argparse
import json
import os
import subprocess
import sys
import tempfile
from typing import Dict, Any, List

import yaml
from rich.console import Console
from rich.progress import Progress

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="Main orchestrator for PM-Drop experiments.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true", help="Run smoke_test.yaml configuration.")
    group.add_argument("--full-experiment", action="store_true", help="Run full_experiment.yaml configuration.")
    parser.add_argument("--results-dir", type=str, required=True, help="Root directory for all outputs.")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def tee_subprocess(cmd: List[str], stdout_path: str, stderr_path: str):
    with open(stdout_path, "w") as out_f, open(stderr_path, "w") as err_f:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        while True:
            out_line = process.stdout.readline()
            err_line = process.stderr.readline()
            if out_line:
                console.print(out_line.rstrip())
                out_f.write(out_line)
            if err_line:
                console.print(err_line.rstrip(), style="red")
                err_f.write(err_line)
            if out_line == "" and err_line == "" and process.poll() is not None:
                break
        return process.returncode


def run_variation(run_cfg: Dict[str, Any], results_dir: str, smoke: bool):
    run_id = run_cfg["run_id"]
    run_dir = os.path.join(results_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Serialize run config to a temporary JSON file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
        json.dump(run_cfg, tmp)
        tmp_path = tmp.name

    cmd = [
        sys.executable,
        "-m",
        "src.train",
        "--run-config",
        tmp_path,
        "--results-dir",
        results_dir,
    ]
    if smoke:
        cmd.append("--smoke-test")

    stdout_path = os.path.join(run_dir, "stdout.log")
    stderr_path = os.path.join(run_dir, "stderr.log")

    console.rule(f"[bold yellow]Launching Run — {run_id}")
    retcode = tee_subprocess(cmd, stdout_path, stderr_path)
    if retcode != 0:
        console.print(f"[red]Run {run_id} failed with return code {retcode}")
    else:
        console.print(f"[green]Run {run_id} completed successfully.")


def main():
    args = parse_args()

    cfg_path = "config/smoke_test.yaml" if args.smoke_test else "config/full_experiment.yaml"
    master_cfg = load_config(cfg_path)
    variations = master_cfg.get("variations", [])

    os.makedirs(args.results_dir, exist_ok=True)

    with Progress() as progress:
        task = progress.add_task("Running variations", total=len(variations))
        for run_cfg in variations:
            run_variation(run_cfg, args.results_dir, smoke=args.smoke_test)
            progress.advance(task)

    # After all runs, trigger evaluation
    eval_cmd = [
        sys.executable,
        "-m",
        "src.evaluate",
        "--results-dir",
        args.results_dir,
        "--config-path",
        cfg_path,
    ]
    console.rule("[bold magenta]Aggregating Results via evaluate.py")
    subprocess.run(eval_cmd, check=True)


if __name__ == "__main__":
    main()