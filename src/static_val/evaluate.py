"""Run every detector evaluation the analysis needs.

Each evaluation is a separate process of ``static_val.detector_eval`` so that
TensorFlow state never carries over between checkpoints. Finished results are
skipped, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from static_val.config import Config
from static_val.strata import UNC_TIERS


@dataclass(frozen=True)
class EvalJob:
    dataset: str
    stratum: str
    method: str
    tfrecord: Path
    num_images: int
    image_ids: Path | None

    def output(self, cfg: Config) -> Path:
        return cfg.detector_dir / self.dataset / self.stratum / f"{self.method}.json"


def plan(cfg: Config) -> list[EvalJob]:
    strata = cfg.output_dir / "strata"
    jobs: list[EvalJob] = []
    kitti = (cfg.kitti_tfrecord, cfg.kitti_num_images)
    kitti_lists = strata / "KITTI"
    for method in cfg.methods:
        jobs.append(EvalJob("KITTI", "full", method, *kitti, None))
        kitti_strata = [f"subset_{i}" for i in range(cfg.subset_count)]
        kitti_strata += [f"depth_{tier}" for tier in ("near", "medium", "far")]
        for name in kitti_strata:
            jobs.append(EvalJob("KITTI", name, method, *kitti, kitti_lists / f"{name}.txt"))
        for tier in UNC_TIERS:
            ids = kitti_lists / f"entropy_{tier}" / f"{method}.txt"
            jobs.append(EvalJob("KITTI", f"entropy_{tier}", method, *kitti, ids))
        for name, target in cfg.targets.items():
            for i in range(cfg.subset_count):
                jobs.append(
                    EvalJob(
                        name,
                        f"subset_{i}",
                        method,
                        target.tfrecord,
                        target.num_images,
                        strata / name / f"subset_{i}.txt",
                    )
                )
    return jobs


def run(cfg: Config, python: str = sys.executable) -> None:
    jobs = plan(cfg)
    missing = [j.image_ids for j in jobs if j.image_ids is not None and not j.image_ids.is_file()]
    if missing:
        raise FileNotFoundError(f"Run 'static-val strata' first. Missing: {missing[0]}")

    # The detector environment only needs this package's source on its path.
    env = os.environ.copy()
    src = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(p for p in (src, env.get("PYTHONPATH")) if p)

    for n, job in enumerate(jobs, start=1):
        out = job.output(cfg)
        if out.is_file():
            continue
        print(f"[{n}/{len(jobs)}] {job.dataset}/{job.stratum} {job.method}", flush=True)
        cmd = [
            python, "-m", "static_val.detector_eval",
            "--autolabeling-dir", str(cfg.autolabeling_dir),
            "--hparams", str(cfg.hparams),
            "--checkpoint", str(cfg.checkpoint(job.method)),
            "--tfrecord", str(job.tfrecord),
            "--num-images", str(job.num_images),
            "--output", str(out),
        ]  # fmt: skip
        if job.image_ids is not None:
            cmd += ["--image-ids", str(job.image_ids)]
        subprocess.run(cmd, check=True, env=env)


def load_metrics(cfg: Config, dataset: str, stratum: str) -> dict[str, dict[str, float]]:
    """Method -> metrics of one evaluated stratum, in config method order."""
    out = {}
    for method in cfg.methods:
        path = cfg.detector_dir / dataset / stratum / f"{method}.json"
        with open(path) as f:
            out[method] = json.load(f)
    return out
