"""Experiment configuration loaded from a YAML file.

Relative paths in the YAML resolve against the directory that holds the YAML
file. Paths may use ``{data_root}`` and ``{autolabeling_dir}``; per-method
paths also use ``{run}``, the run name a method's detector was trained under.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TargetSet:
    name: str
    tfrecord: Path
    num_images: int
    first_source_id: int

    def source_ids(self) -> list[int]:
        return list(range(self.first_source_id, self.first_source_id + self.num_images))


@dataclass(frozen=True)
class Config:
    data_root: Path
    output_dir: Path
    autolabeling_dir: Path
    methods: dict[str, str]
    hparams: Path
    checkpoint_template: str
    predictions_template: str
    kitti_label_dir: Path
    kitti_image_dir: Path
    kitti_val_index: Path
    kitti_tfrecord: Path
    kitti_num_images: int
    kitti_depth_dir: Path
    targets: dict[str, TargetSet]
    subset_count: int
    subset_fraction: float
    subset_seed: int
    bootstrap_samples: int
    bootstrap_seed: int

    def checkpoint(self, method: str) -> Path:
        return Path(self.checkpoint_template.format(run=self.methods[method]))

    def predictions(self, method: str) -> Path:
        return Path(self.predictions_template.format(run=self.methods[method]))

    def kitti_val_ids(self) -> list[int]:
        with open(self.kitti_val_index) as f:
            return [int(line) for line in f if line.strip()]

    @property
    def cache_dir(self) -> Path:
        return self.output_dir / "cache"

    @property
    def detector_dir(self) -> Path:
        return self.output_dir / "detector_eval"


class _Resolver:
    def __init__(self, base: Path, variables: dict[str, Path]):
        self.base = base
        self.variables = variables

    def path(self, value: str) -> Path:
        for key, val in self.variables.items():
            value = value.replace("{" + key + "}", str(val))
        path = Path(value)
        return path if path.is_absolute() else (self.base / path).resolve()

    def template(self, value: str) -> str:
        return str(self.path(value.replace("{run}", "\0"))).replace("\0", "{run}")


def load_config(
    path: str | Path,
    data_root: str | Path | None = None,
    autolabeling_dir: str | Path | None = None,
) -> Config:
    path = Path(path).resolve()
    with open(path) as f:
        raw = yaml.safe_load(f)

    plain = _Resolver(path.parent, {})
    root = Path(data_root).resolve() if data_root else plain.path(raw["data_root"])
    autolab = (
        Path(autolabeling_dir).resolve()
        if autolabeling_dir
        else plain.path(raw["autolabeling_dir"])
    )
    res = _Resolver(path.parent, {"data_root": root, "autolabeling_dir": autolab})

    kitti = raw["kitti"]
    detector = raw["detector"]
    targets = {
        name: TargetSet(
            name=name,
            tfrecord=res.path(spec["tfrecord"]),
            num_images=int(spec["num_images"]),
            first_source_id=int(spec["first_source_id"]),
        )
        for name, spec in raw["targets"].items()
    }

    return Config(
        data_root=root,
        output_dir=res.path(raw["output_dir"]),
        autolabeling_dir=autolab,
        methods=dict(raw["methods"]),
        hparams=res.path(detector["hparams"]),
        checkpoint_template=res.template(detector["checkpoint"]),
        predictions_template=res.template(detector["predictions"]),
        kitti_label_dir=res.path(kitti["label_dir"]),
        kitti_image_dir=res.path(kitti["image_dir"]),
        kitti_val_index=res.path(kitti["val_index"]),
        kitti_tfrecord=res.path(kitti["tfrecord"]),
        kitti_num_images=int(kitti["num_images"]),
        kitti_depth_dir=res.path(kitti["depth_dir"]),
        targets=targets,
        subset_count=int(raw["subsets"]["count"]),
        subset_fraction=float(raw["subsets"]["fraction"]),
        subset_seed=int(raw["subsets"]["seed"]),
        bootstrap_samples=int(raw["bootstrap"]["samples"]),
        bootstrap_seed=int(raw["bootstrap"]["seed"]),
    )
