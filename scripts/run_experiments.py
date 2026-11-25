"""Main CLI entry point for SEED DGCNN + MMD/DANN Emotion Recognition benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import run_seed_benchmark
from src.utils import load_config, set_seed, setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SJTU SEED DGCNN + Domain Adaptation Benchmark.")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs" / "config.yaml"))
    parser.add_argument("--mode", choices=["quick", "full"], default="full", help="full=all 15 subjects, quick=5 subjects")
    parser.add_argument("--subjects", type=int, nargs="*", default=None)
    parser.add_argument("--protocol", choices=["both", "within", "loso"], default="both")
    args = parser.parse_args()

    cfg = load_config(args.config)
    for key in ("data_dir", "results_dir"):
        p = Path(cfg["project"][key])
        if not p.is_absolute():
            cfg["project"][key] = str(PROJECT_ROOT / p)

    set_seed(int(cfg["project"]["seed"]))
    setup_logger("seed_dgcnn", log_file=Path(cfg["project"]["results_dir"]) / "run.log")

    if args.subjects:
        targets = args.subjects
    else:
        targets = cfg["dataset"]["quick_subjects"] if args.mode == "quick" else cfg["dataset"]["subjects"]

    run_seed_benchmark(
        target_subjects=targets,
        cfg=cfg,
        run_within=args.protocol in ("both", "within"),
        run_loso=args.protocol in ("both", "loso"),
    )


if __name__ == "__main__":
    main()
