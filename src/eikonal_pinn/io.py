"""Run directory writer: results/<run_name>/{config.json, loss.csv, metrics.json, fields.npz, figs}."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np


class RunWriter:
    def __init__(self, root: str, run_name: str, loss_columns):
        self.dir = Path(root) / run_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.loss_path = self.dir / "loss.csv"
        self.loss_columns = list(loss_columns)
        self._f = open(self.loss_path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=self.loss_columns)
        self._w.writeheader()

    def write_config(self, cfg: dict):
        with open(self.dir / "config.json", "w") as f:
            json.dump(cfg, f, indent=2, sort_keys=True)

    def log(self, row: dict):
        self._w.writerow({k: row.get(k, "") for k in self.loss_columns})
        self._f.flush()

    def write_metrics(self, metrics: dict):
        with open(self.dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)

    def write_fields(self, **arrays):
        np.savez(self.dir / "fields.npz", **{k: np.asarray(v) for k, v in arrays.items()})

    def path(self, name: str) -> str:
        return str(self.dir / name)

    def close(self):
        self._f.close()
