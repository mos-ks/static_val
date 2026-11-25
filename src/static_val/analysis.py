"""Stage 1 to 3 analyses from the detector evaluations and the match-table cache.

Stage 1: ranking inconsistency within a validation set (random subsets) and
across datasets. Stage 2: transferability of the KITTI ranking per stratum
(class, depth, entropy tertile) to the cross-domain targets. Stage 3:
deployment regret of selecting the AL method on each uncertainty tertile, and
the protocol ranking against the cross-dataset score.

The cross-domain ranking of each target is the mean mAP over its random
subsets. All mAP values are reported in percentage points.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from static_val.ap import MethodTable, mean_ap
from static_val.config import Config
from static_val.constants import CLASS_GROUPS, KITTI_CLASSES, SIGNALS, TERTILES
from static_val.evaluate import load_metrics
from static_val.kitti import load_ground_truth
from static_val.matching import build_tables, gt_count_matrix, read_predictions
from static_val.strata import UNC_TIERS
from static_val.tertiles import assign_tertiles

AP_KEYS = tuple(f"AP_/{name.lower()}" for name in KITTI_CLASSES)


# ─── Match-table cache ─────────────────────────────────────────────
def build_cache(cfg: Config) -> Path:
    """Match every method's validation predictions against the KITTI ground truth."""
    gt = load_ground_truth(cfg.kitti_label_dir, cfg.kitti_val_ids())
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    stems: set[str] = set(gt)
    for method in cfg.methods:
        detections, features = build_tables(read_predictions(cfg.predictions(method)), gt)
        method_dir = cfg.cache_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        detections.to_pickle(method_dir / "detections.pkl")
        features.to_pickle(method_dir / "image_features.pkl")
        stems |= set(features["image_stem"])
        print(f"{method}: {len(detections)} detections")
    ordered = sorted(stems)
    np.save(cfg.cache_dir / "gt_counts.npy", gt_count_matrix(gt, ordered))
    with open(cfg.cache_dir / "image_stems.json", "w") as f:
        json.dump(ordered, f)
    return cfg.cache_dir


def load_cache(cfg: Config) -> tuple[list[str], dict[str, MethodTable], dict[str, pd.DataFrame]]:
    with open(cfg.cache_dir / "image_stems.json") as f:
        stems = json.load(f)
    gt_counts = np.load(cfg.cache_dir / "gt_counts.npy")
    tables, features = {}, {}
    for method in cfg.methods:
        method_dir = cfg.cache_dir / method
        tables[method] = MethodTable.from_frames(
            pd.read_pickle(method_dir / "detections.pkl"), gt_counts, stems
        )
        features[method] = pd.read_pickle(method_dir / "image_features.pkl")
    return stems, tables, features


# ─── Shared helpers ────────────────────────────────────────────────
def _rho(a, b) -> float:
    return float(spearmanr(np.asarray(a), np.asarray(b))[0])


def _ranks(values: np.ndarray) -> np.ndarray:
    """Rank 1 = highest value; ties share their average rank."""
    return pd.Series(values).rank(ascending=False, method="average").to_numpy()


def _ap(metrics: dict[str, float], key: str = "AP") -> float:
    return 100.0 * metrics[key]


def _group_ap(metrics: dict[str, float], classes: tuple[int, ...]) -> float:
    return 100.0 * float(np.mean([metrics[AP_KEYS[c]] for c in classes]))


def subset_maps(cfg: Config, dataset: str) -> np.ndarray:
    """(n_subsets, n_methods) mAP of each method on each random subset."""
    return np.asarray(
        [
            [_ap(m) for m in load_metrics(cfg, dataset, f"subset_{i}").values()]
            for i in range(cfg.subset_count)
        ]
    )


def target_reference(cfg: Config) -> dict[str, np.ndarray]:
    """Per target, the mean mAP over its random subsets for each method."""
    return {name: subset_maps(cfg, name).mean(axis=0) for name in cfg.targets}


# ─── Stage 1 ───────────────────────────────────────────────────────
def stage1(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    methods = list(cfg.methods)
    rank_rows = []
    for dataset in ["KITTI", *cfg.targets]:
        maps = subset_maps(cfg, dataset)
        ranks = np.stack([_ranks(row) for row in maps])
        for k, method in enumerate(methods):
            rank_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "mean_map": maps[:, k].mean(),
                    "std_map": maps[:, k].std(),
                    "rank_min": ranks[:, k].min(),
                    "rank_median": float(np.median(ranks[:, k])),
                    "rank_max": ranks[:, k].max(),
                    "rank_range": ranks[:, k].max() - ranks[:, k].min(),
                }
            )

    kitti_full = np.asarray([_ap(m) for m in load_metrics(cfg, "KITTI", "full").values()])
    rankings = {"KITTI": kitti_full, **target_reference(cfg)}
    names = list(rankings)
    cross_rows = [
        {"source": a, "target": b, "rho": _rho(rankings[a], rankings[b])}
        for i, a in enumerate(names)
        for b in names[i + 1 :]
    ]
    return pd.DataFrame(rank_rows), pd.DataFrame(cross_rows)


# ─── Stage 2 ───────────────────────────────────────────────────────
def stage2(cfg: Config, classes: str = "head") -> pd.DataFrame:
    """Spearman rho between each KITTI stratum ranking and each target ranking."""
    group = CLASS_GROUPS[classes]
    reference = target_reference(cfg)
    full = load_metrics(cfg, "KITTI", "full")

    def group_maps(metrics: dict[str, dict[str, float]], classes: tuple[int, ...]) -> np.ndarray:
        return np.asarray([_group_ap(m, classes) for m in metrics.values()])

    strata: list[tuple[str, str, np.ndarray]] = []
    for c, name in enumerate(KITTI_CLASSES):
        strata.append(("class", name, group_maps(full, (c,))))
    for tier in ("near", "medium", "far"):
        metrics = load_metrics(cfg, "KITTI", f"depth_{tier}")
        strata.append(("depth", tier, group_maps(metrics, group)))
    strata.append(("entropy", "full", group_maps(full, group)))
    for tier in UNC_TIERS:
        metrics = load_metrics(cfg, "KITTI", f"entropy_{tier}")
        strata.append(("entropy", tier, group_maps(metrics, group)))

    rows = []
    for axis, stratum, values in strata:
        row = {"axis": axis, "stratum": stratum}
        for target, ref in reference.items():
            row[f"rho_{target}"] = _rho(values, ref)
        row["transfers"] = all(row[f"rho_{t}"] > 0 for t in reference)
        rows.append(row)
    return pd.DataFrame(rows)


# ─── Stage 3 ───────────────────────────────────────────────────────
def stratified_maps(cfg: Config, classes: str = "all") -> pd.DataFrame:
    """In-domain mAP of every method on the full set and on each uncertainty tertile."""
    stems, tables, features = load_cache(cfg)
    tiers = assign_tertiles(features, stems)
    group = CLASS_GROUPS[classes]
    ones = np.ones(len(stems), dtype=np.int32)
    rows = []
    for k, method in enumerate(cfg.methods):
        rows.append(
            {
                "mode": "-",
                "signal": "-",
                "slice": "FULL",
                "method": method,
                "map": 100.0 * mean_ap(tables[method], ones, group),
            }
        )
        for mode, by_signal in tiers.items():
            for signal in SIGNALS:
                for t, tier in enumerate(TERTILES):
                    weights = (by_signal[signal][k] == t).astype(np.int32)
                    rows.append(
                        {
                            "mode": mode,
                            "signal": signal,
                            "slice": tier,
                            "method": method,
                            "map": 100.0 * mean_ap(tables[method], weights, group),
                        }
                    )
    return pd.DataFrame(rows)


def deployment_regret(source: np.ndarray, targets: dict[str, np.ndarray]) -> dict[str, float]:
    """Regret per target of deploying argmax(source): deployed minus oracle mAP (<= 0)."""
    pick = int(np.argmax(source))
    return {name: float(vals[pick] - vals.max()) for name, vals in targets.items()}


def stage3(
    methods: list[str],
    maps: pd.DataFrame,
    reference: dict[str, np.ndarray],
    mode: str = "global",
) -> pd.DataFrame:
    """Regret of picking the AL method by in-domain mAP on FULL and on each tertile.

    Recovery is the share of the FULL rule's summed regret that a rule removes.
    """

    def source(slice_: str, signal: str) -> np.ndarray:
        if slice_ == "FULL":
            sel = maps[maps["slice"] == "FULL"]
        else:
            sel = maps[
                (maps["mode"] == mode) & (maps["signal"] == signal) & (maps["slice"] == slice_)
            ]
        return sel.set_index("method").loc[methods, "map"].to_numpy()

    rows = []
    for signal in SIGNALS:
        baseline = None
        for slice_ in ("FULL", *TERTILES):
            src = source(slice_, signal)
            regret = deployment_regret(src, reference)
            total = sum(regret.values())
            baseline = total if slice_ == "FULL" else baseline
            row = {"signal": signal, "rule": slice_, "pick": methods[int(np.argmax(src))]}
            row.update({f"regret_{t}": r for t, r in regret.items()})
            row["regret_sum"] = total
            row["regret_mean"] = total / len(regret)
            row["recovery_pct"] = (
                np.nan
                if slice_ == "FULL" or baseline == 0
                else 100.0 * (total - baseline) / abs(baseline)
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _min_max(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    span = values.max() - values.min()
    return (values - values.min()) / span if span > 0 else np.zeros_like(values)


def protocol_ranking(
    methods: list[str],
    kitti_full: np.ndarray,
    protocol: np.ndarray,
    targets: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Rank by aggregate KITTI mAP and by the protocol, against the cross-dataset score.

    The cross-dataset score of a method is the mean of its min-max normalised mAP
    on KITTI and on every target. ``rho_*`` is the Spearman correlation of each
    source-only ranking with that score and is the same in every row.
    """
    score = np.mean([_min_max(v) for v in (kitti_full, *targets.values())], axis=0)
    frame = pd.DataFrame(
        {
            "method": methods,
            "kitti_map": kitti_full,
            "protocol_map": protocol,
            "cross_dataset_score": score,
            "rank_aggregate": _ranks(np.asarray(kitti_full)),
            "rank_protocol": _ranks(np.asarray(protocol)),
        }
    )
    frame["rho_aggregate"] = _rho(kitti_full, score)
    frame["rho_protocol"] = _rho(protocol, score)
    return frame


def stage3_protocol(cfg: Config, classes: str = "shared") -> pd.DataFrame:
    """Protocol ranking on KITTI: ``classes`` mAP on each method's own MID entropy tertile."""
    group = CLASS_GROUPS[classes]
    kitti_full = np.asarray([_ap(m) for m in load_metrics(cfg, "KITTI", "full").values()])
    mid = load_metrics(cfg, "KITTI", "entropy_mid")
    protocol = np.asarray([_group_ap(m, group) for m in mid.values()])
    return protocol_ranking(list(cfg.methods), kitti_full, protocol, target_reference(cfg))


def run_all(cfg: Config, stage2_classes: str = "head", stage3_classes: str = "all") -> Path:
    out = cfg.output_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    ranks, cross = stage1(cfg)
    ranks.to_csv(out / "stage1_subset_ranks.csv", index=False)
    cross.to_csv(out / "stage1_cross_dataset_rho.csv", index=False)
    stage2(cfg, stage2_classes).to_csv(out / "stage2_transferability.csv", index=False)
    maps = stratified_maps(cfg, stage3_classes)
    maps.to_csv(out / "stage3_stratified_map.csv", index=False)
    regret = stage3(list(cfg.methods), maps, target_reference(cfg))
    regret.to_csv(out / "stage3_deployment_regret.csv", index=False)
    stage3_protocol(cfg).to_csv(out / "stage3_protocol_ranking.csv", index=False)
    return out
