#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CLI runner for the traffic monitoring pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.pipeline import TrafficPipeline, load_config
from src.utils.logger import get_logger

log = get_logger("runner")


def parse_args():
    p = argparse.ArgumentParser(description="Run YOLOv8 Traffic Monitoring pipeline.")
    p.add_argument("--config", default="config/config.yaml", help="Path to config YAML")
    p.add_argument("--source", help="Override video source")
    p.add_argument("--output", help="Output annotated video path")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--report", action="store_true", help="Write JSON report after run")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.source:
        cfg["video"]["source"] = args.source

    pipe = TrafficPipeline(cfg)
    report = pipe.run(output_path=args.output, max_frames=args.max_frames)

    if args.report:
        report_dir = Path(cfg["storage"]["reports_dir"])
        report_dir.mkdir(parents=True, exist_ok=True)
        out_path = report_dir / "latest_report.json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        log.info("Report written to {}", out_path)

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
