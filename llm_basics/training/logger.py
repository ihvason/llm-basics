"""Experiment tracking: per-run config capture plus local and optional remote logs.

Metrics are always appended to a local JSONL file; feeding Weights & Biases is
opt-in, so the dependency is imported lazily and is never required to train.
"""

import json
import os
import time
from typing import Any


class ExperimentTracker:
    """Record a training run's hyperparameters and metric stream.

    Creates ``<log_dir>/<exp_name>_config.json`` once, then appends one JSON
    object per :meth:`log` call to ``<log_dir>/<exp_name>_metrics.jsonl``.
    """

    def __init__(
        self,
        exp_name: str,
        config: dict[str, Any],
        log_dir: str = "./runs",
        use_wandb: bool = False,
        wandb_project: str = "llm_basics",
    ):
        """Open the run's output directory and persist its config.

        Args:
            exp_name: Run name, used as the filename prefix and as the wandb
                run id.
            config: Hyperparameters to persist alongside the metrics.
            log_dir: Directory that receives the run's files.
            use_wandb: Whether to mirror metrics to Weights & Biases.
            wandb_project: wandb project name, used only when ``use_wandb``.
        """
        self.exp_name = exp_name
        self.config = config
        self.log_dir = log_dir
        self.use_wandb = use_wandb
        self.start_time = time.time()

        os.makedirs(log_dir, exist_ok=True)
        self.local_log_path = os.path.join(log_dir, f"{exp_name}_metrics.jsonl")
        self.config_path = os.path.join(log_dir, f"{exp_name}_config.json")

        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        if self.use_wandb:
            import wandb

            wandb.init(
                project=wandb_project,
                name=exp_name,
                id=exp_name,
                resume="allow",
                config=config,
            )

    def log(self, metrics: dict[str, Any], step: int) -> None:
        """Append one metrics record, stamped with the step and wall-clock time.

        Args:
            metrics: Metric name to value mapping, e.g. ``{"train/loss": 1.23}``.
            step: Optimizer step the metrics belong to.
        """
        wall_time = time.time() - self.start_time
        record = {
            "step": step,
            "wall_clock_time": round(wall_time, 2),
            **metrics,
        }

        with open(self.local_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        if self.use_wandb:
            import wandb

            wandb.log(record, step=step)

    def close(self) -> None:
        """Finish the remote run, if one was started."""
        if self.use_wandb:
            import wandb

            wandb.finish()
