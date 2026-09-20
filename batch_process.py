"""
Batch processing CLI.

Example:
    python batch_process.py \
        --input-dir data/raw \
        --output-dir data/processed \
        --pipeline adaptive \
        --save-intermediate \
        --visualize
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import List

from dip.pipeline import build_pipeline, load_config
from dip.utils.image_io import SUPPORTED_EXTENSIONS, read_image, write_image
from dip.utils.logging_utils import get_logger
from dip.utils.visualization import side_by_side


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Medical DIP — batch processing")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--pipeline",
        default="adaptive",
        choices=["minimal", "traditional", "handwriting", "adaptive"],
    )
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--report-dir", default=None,
                   help="Where to write per-image JSON reports (default: <output-dir>/reports)")
    p.add_argument("--intermediate-dir", default=None,
                   help="Where to write intermediates (default: <output-dir>/intermediate)")
    p.add_argument("--comparison-dir", default=None,
                   help="Where to write original|final comparisons (default: <output-dir>/comparisons)")
    p.add_argument("--save-intermediate", action="store_true")
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="Process only first N images")
    p.add_argument("--recursive", action="store_true", help="Recurse into subdirectories")
    return p.parse_args()


def gather_images(root: Path, recursive: bool) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {root}")
    paths: List[Path] = []
    iterator = root.rglob("*") if recursive else root.iterdir()
    for p in iterator:
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            paths.append(p)
    paths.sort()
    return paths


def main() -> int:
    args = parse_args()
    log = get_logger()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir) if args.report_dir else output_dir / "reports"
    inter_dir = Path(args.intermediate_dir) if args.intermediate_dir else output_dir / "intermediate"
    comp_dir = Path(args.comparison_dir) if args.comparison_dir else output_dir / "comparisons"

    for d in (output_dir, report_dir):
        d.mkdir(parents=True, exist_ok=True)
    if args.save_intermediate:
        inter_dir.mkdir(parents=True, exist_ok=True)
    if args.visualize:
        comp_dir.mkdir(parents=True, exist_ok=True)

    images = gather_images(input_dir, args.recursive)
    if args.limit:
        images = images[: args.limit]
    if not images:
        log.error("No supported images in %s", input_dir)
        return 2

    config = load_config(args.config)
    pipeline = build_pipeline(
        args.pipeline,
        config=config,
        intermediate_dir=inter_dir if args.save_intermediate else None,
        save_intermediate=args.save_intermediate,
    )

    summary_path = output_dir / "batch_summary.csv"
    with open(summary_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "image", "pipeline", "status", "seconds",
                "operations_applied", "operations_skipped",
                "output_image", "report", "error",
            ]
        )

        ok, fail = 0, 0
        for img_path in images:
            log.info("[%d/%d] %s", ok + fail + 1, len(images), img_path.name)
            t0 = time.time()
            try:
                result, report = pipeline.run(img_path)
                out_path = output_dir / f"{img_path.stem}__{args.pipeline}.png"
                write_image(out_path, result)
                report.set_output(str(out_path))
                rpt_path = report_dir / f"{img_path.stem}__{args.pipeline}.json"
                report.save(rpt_path)

                if args.visualize:
                    try:
                        original = read_image(img_path)
                        comp_path = comp_dir / f"{img_path.stem}__{args.pipeline}.png"
                        side_by_side([original, result], titles=["Original", args.pipeline], save_path=comp_path)
                    except Exception as e:
                        log.warning("Could not build comparison for %s: %s", img_path.name, e)

                elapsed = time.time() - t0
                writer.writerow([
                    img_path.name, args.pipeline, "ok", f"{elapsed:.2f}",
                    "|".join(report.operations_applied),
                    "|".join(report.operations_skipped),
                    str(out_path), str(rpt_path), "",
                ])
                ok += 1
            except Exception as e:
                elapsed = time.time() - t0
                log.warning("FAILED %s: %s", img_path.name, e)
                writer.writerow([
                    img_path.name, args.pipeline, "error", f"{elapsed:.2f}",
                    "", "", "", "", str(e),
                ])
                fail += 1

    log.info("Batch complete: %d ok, %d failed. Summary: %s", ok, fail, summary_path)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
