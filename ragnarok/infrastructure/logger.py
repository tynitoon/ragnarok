"""Logging: TensorBoard + CSV for training metrics."""

import csv
import os
import time
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Dual TensorBoard + CSV logger for training metrics."""

    def __init__(self, log_dir: str, run_name: str | None = None):
        if run_name is None:
            run_name = f"run_{int(time.time())}"
        self.log_path = Path(log_dir) / run_name
        self.log_path.mkdir(parents=True, exist_ok=True)

        self.tb_writer = SummaryWriter(log_dir=str(self.log_path / "tensorboard"))

        self.csv_path = self.log_path / "metrics.csv"
        self._csv_file = open(self.csv_path, "w", newline="")
        self._csv_writer = None
        self._csv_fields = None

    def log(self, step: int, metrics: dict[str, float]):
        """Log a dict of metrics at a given step."""
        for key, value in metrics.items():
            self.tb_writer.add_scalar(key, value, step)

        if self._csv_writer is None:
            self._csv_fields = ["step"] + sorted(metrics.keys())
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._csv_fields)
            self._csv_writer.writeheader()

        row = {"step": step}
        row.update(metrics)
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def close(self):
        """Flush and close all writers."""
        self.tb_writer.close()
        self._csv_file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
