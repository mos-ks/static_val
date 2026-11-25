"""Command-line entry point: ``static-val <command> --config configs/kitti.yaml``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from static_val.config import load_config
from static_val.constants import CLASS_GROUPS


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="static-val", description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/kitti.yaml"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--autolabeling-dir", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)

    bdd = sub.add_parser("bdd-tfrecord", help="Write the BDD100K val TFRecord in KITTI classes.")
    bdd.add_argument("--bdd-dir", type=Path, help="BDD100K root with labels/ and images/.")

    strata = sub.add_parser("strata", help="Write the image-id lists of all evaluated subsets.")
    strata.add_argument(
        "--inverse-depth",
        action="store_true",
        help="Depth maps hold inverse depth (larger is closer) instead of depth.",
    )

    evaluate = sub.add_parser("evaluate", help="Run all detector evaluations (resumable).")
    evaluate.add_argument("--python", default=sys.executable, help="Detector environment.")

    sub.add_parser("match", help="Build per-detection TP tables from validation predictions.")

    analyze = sub.add_parser("analyze", help="Stage 1 to 3 analyses into results/analysis.")
    analyze.add_argument("--stage2-classes", choices=list(CLASS_GROUPS), default="head")
    analyze.add_argument("--stage3-classes", choices=list(CLASS_GROUPS), default="all")

    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.data_root, args.autolabeling_dir)

    if args.command == "bdd-tfrecord":
        from static_val.bdd_tfrecord import write_bdd_tfrecord

        root = args.bdd_dir or cfg.data_root / "BDD100K" / "bdd100k"
        n = write_bdd_tfrecord(
            root / "labels" / "bdd100k_labels_images_val.json",
            root / "images" / "100k" / "val",
            cfg.targets["BDD"].tfrecord,
        )
        print(f"{n} images -> {cfg.targets['BDD'].tfrecord}")
    elif args.command == "strata":
        from static_val.strata import build_strata

        print(build_strata(cfg, inverse_depth=args.inverse_depth))
    elif args.command == "evaluate":
        from static_val.evaluate import run

        run(cfg, args.python)
    elif args.command == "match":
        from static_val.analysis import build_cache

        print(build_cache(cfg))
    elif args.command == "analyze":
        from static_val.analysis import run_all

        out = run_all(cfg, args.stage2_classes, args.stage3_classes)
        with pd.option_context("display.width", 200, "display.precision", 2):
            for name in (
                "stage2_transferability",
                "stage3_deployment_regret",
                "stage3_protocol_ranking",
            ):
                print(f"\n{name}\n", pd.read_csv(out / f"{name}.csv").to_string(index=False))
        print(f"\nAll tables: {out}")


if __name__ == "__main__":
    main()
