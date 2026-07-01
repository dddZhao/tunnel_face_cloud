from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tunnel_cloud_matplotlib"))

from .config import cycle_distance_positions, load_config, output_root
from .frame_estimation import estimate_frames
from .geometric_features import generate_features
from .projection import generate_projections
from .quality import inspect_dataset
from .relative_placement import place_cycles


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def paths(cfg: dict) -> dict[str, Path]:
    root = output_root(cfg)
    return {
        "quality": root / "01_quality",
        "frames": root / "02_frames",
        "placement": root / "03_placement",
        "features": root / "04_features",
        "projection": root / "05_projection",
    }


def cmd_inspect(cfg: dict) -> None:
    inspect_dataset(cfg, paths(cfg)["quality"])


def cmd_estimate_frames(cfg: dict) -> None:
    estimate_frames(cfg, paths(cfg)["frames"], cycle_distance_positions(cfg))


def cmd_place_cycles(cfg: dict) -> None:
    p = paths(cfg)
    place_cycles(cfg, p["frames"], p["placement"])


def cmd_features(cfg: dict) -> None:
    p = paths(cfg)
    generate_features(cfg, p["placement"], p["features"])


def cmd_project(cfg: dict) -> None:
    p = paths(cfg)
    generate_projections(cfg, p["placement"], p["projection"])


def cmd_refine(cfg: dict) -> None:
    raise SystemExit("Stable-region constrained refinement is intentionally not enabled in phase 1.")


def cmd_run_all(cfg: dict) -> None:
    cmd_inspect(cfg)
    cmd_estimate_frames(cfg)
    cmd_place_cycles(cfg)
    cmd_features(cfg)
    cmd_project(cfg)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tunnel-cloud")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in [
        "inspect",
        "estimate-frames",
        "place-cycles",
        "refine",
        "features",
        "project",
        "run-all",
    ]:
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    logging.info("Loaded config: %s", args.config)
    dispatch = {
        "inspect": cmd_inspect,
        "estimate-frames": cmd_estimate_frames,
        "place-cycles": cmd_place_cycles,
        "refine": cmd_refine,
        "features": cmd_features,
        "project": cmd_project,
        "run-all": cmd_run_all,
    }
    dispatch[args.command](cfg)


if __name__ == "__main__":
    main()
