#!/usr/bin/env python
"""Visualise model accuracy metrics produced by ``scripts/evaluate.py``.

Generates a per-class AP bar chart and a precision/recall comparison plot,
saving PNGs alongside the input JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="outputs/eval/results.json")
    p.add_argument("--out-dir", default="outputs/eval")
    return p.parse_args()


def per_class_bar(per_class: dict, out_path: str):
    classes = list(per_class.keys())
    aps = [per_class[c]["AP"] for c in classes]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(classes, aps, color="#0099CC")
    for b, v in zip(bars, aps):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0.7, 1.0)
    ax.set_ylabel("AP @0.5")
    ax.set_title("YOLOv8 - Per-class Detection AP")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def precision_recall(per_class: dict, out_path: str):
    classes = list(per_class.keys())
    precs = [per_class[c]["precision"] for c in classes]
    recs = [per_class[c]["recall"] for c in classes]
    x = np.arange(len(classes))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w / 2, precs, w, label="Precision", color="#4CAF50")
    ax.bar(x + w / 2, recs, w, label="Recall", color="#FF6F00")
    ax.set_xticks(x); ax.set_xticklabels(classes)
    ax.set_ylim(0.7, 1.0)
    ax.set_title("Precision vs Recall (per class)")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def headline_metrics(results: dict, out_path: str):
    labels = ["mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall"]
    values = [results.get(k, 0) for k in ["mAP_0.5", "mAP_0.5_0.95", "precision", "recall"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=["#2962FF", "#00B8D4", "#4CAF50", "#FF6F00"])
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_ylim(0.6, 1.0)
    ax.set_title("YOLOv8 Traffic Monitoring - Headline metrics")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    results = json.loads(Path(args.results).read_text())
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    headline_metrics(results, str(out_dir / "headline_metrics.png"))
    if "per_class" in results:
        per_class_bar(results["per_class"], str(out_dir / "per_class_AP.png"))
        precision_recall(results["per_class"], str(out_dir / "precision_recall.png"))
    print(f"Plots written to {out_dir}")


if __name__ == "__main__":
    main()
