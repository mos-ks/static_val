from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from static_val.analysis import deployment_regret, protocol_ranking, stage3
from static_val.ap import MethodTable, mean_ap
from static_val.config import load_config
from static_val.kitti import depth_tertiles, load_ground_truth, random_subsets
from static_val.matching import TP_COLUMNS, build_tables, gt_count_matrix, match_detections
from static_val.strata import entropy_tertiles
from static_val.tertiles import assign_tertiles

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "kitti.yaml"


def test_greedy_matching_assigns_each_gt_once_by_score():
    gt = np.array([[0, 0, 10, 10]], dtype=np.float32)
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 8]], dtype=np.float32)
    tp = match_detections(boxes, np.array([0.4, 0.9], dtype=np.float32), gt)
    # IoU of the higher-scored box is 0.8: it takes the GT at thresholds 0.50 to 0.80.
    assert tp[1, :7].all() and not tp[0, :7].any()
    # Above 0.8 it misses, so the exact but lower-scored box gets the GT.
    assert tp[0, 7:].all() and not tp[1, 7:].any()


def _table(predictions, gt):
    detections, _ = build_tables(predictions, gt)
    stems = sorted(gt)
    return MethodTable.from_frames(detections, gt_count_matrix(gt, stems), stems), stems


def test_perfect_detections_give_map_one_and_misses_lower_it():
    gt = {
        "000000": {0: np.array([[0, 0, 50, 50]], dtype=np.float32)},
        "000001": {0: np.array([[10, 10, 60, 60]], dtype=np.float32)},
    }
    preds = [
        {"image_name": "000000.png", "class": 1, "bbox": [0, 0, 50, 50],
         "det_score": 0.9, "entropy": 0.1},
        {"image_name": "000001.png", "class": 1, "bbox": [10, 10, 60, 60],
         "det_score": 0.8, "entropy": 0.2},
    ]  # fmt: skip
    table, _ = _table(preds, gt)
    assert mean_ap(table, np.ones(2, dtype=np.int32), (0,)) == pytest.approx(1.0)
    assert mean_ap(table, np.array([1, 0]), (0,)) == pytest.approx(1.0)
    assert np.isnan(mean_ap(table, np.ones(2, dtype=np.int32), (1,)))

    table, _ = _table(preds[:1], gt)
    assert mean_ap(table, np.ones(2, dtype=np.int32), (0,)) < 0.6


def test_image_weight_counts_an_image_repeatedly():
    gt = {
        "000000": {0: np.array([[0, 0, 50, 50]], dtype=np.float32)},
        "000001": {0: np.array([[0, 0, 50, 50]], dtype=np.float32)},
    }
    preds = [
        {"image_name": "000000.png", "class": 1, "bbox": [0, 0, 50, 50],
         "det_score": 0.9, "entropy": 0.1},
        {"image_name": "000001.png", "class": 1, "bbox": [100, 100, 150, 150],
         "det_score": 0.95, "entropy": 0.1},
    ]  # fmt: skip
    table, _ = _table(preds, gt)
    once = mean_ap(table, np.array([1, 1]), (0,))
    twice = mean_ap(table, np.array([2, 1]), (0,))
    assert twice > once


def test_global_tertiles_are_shared_and_balanced():
    stems = [f"{i:06d}" for i in range(9)]
    features = {
        m: pd.DataFrame({"image_stem": stems, "mean_entropy": np.arange(9.0) * s,
                         "mean_one_minus_score": np.arange(9.0)[::-1], "num_dets": 1})
        for m, s in (("a", 1.0), ("b", 2.0))
    }  # fmt: skip
    tiers = assign_tertiles(features, stems)
    shared = tiers["global"]["entropy"]
    assert (shared[0] == shared[1]).all()
    assert np.bincount(shared[0]).tolist() == [3, 3, 3]
    assert tiers["global"]["score"][0][0] == 2


def test_entropy_tertiles_split_by_image_mean():
    preds = [{"image_name": f"{i:06d}.png", "entropy": float(i)} for i in (5, 1, 3, 2, 4, 0)]
    assert entropy_tertiles(preds) == {"low": [0, 1], "mid": [2, 3], "high": [4, 5]}


def test_inverse_depth_puts_large_values_in_near():
    sig = {1: 1.0, 2: 2.0, 3: 3.0}
    assert depth_tertiles(sig, inverse_depth=True) == {"near": [3], "medium": [2], "far": [1]}
    assert depth_tertiles(sig, inverse_depth=False) == {"near": [1], "medium": [2], "far": [3]}


def test_random_subsets_are_reproducible():
    a = random_subsets(list(range(100)), 3, 0.2, 42)
    assert a == random_subsets(list(range(100)), 3, 0.2, 42)
    assert all(len(s) == 20 and len(set(s)) == 20 for s in a)


def test_ground_truth_keeps_only_evaluated_classes(tmp_path):
    (tmp_path / "000007.txt").write_text(
        "Car 0 0 0 1 2 3 4 0 0 0 0 0 0 0\n"
        "DontCare -1 -1 -10 5 5 9 9 -1 -1 -1 -1000 -1000 -1000 -10\n"
        "Misc 0 0 0 1 1 2 2 0 0 0 0 0 0 0\n"
        "Tram 0 0 0 10 20 30 40 0 0 0 0 0 0 0\n"
    )
    gt = load_ground_truth(tmp_path, [7])["000007"]
    assert sorted(gt) == [0, 6]
    assert gt[6].tolist() == [[10, 20, 30, 40]]


def test_deployment_regret_and_recovery():
    targets = {"A": np.array([1.0, 3.0, 2.0]), "B": np.array([5.0, 4.0, 6.0])}
    assert deployment_regret(np.array([0.9, 0.1, 0.2]), targets) == {"A": -2.0, "B": -1.0}

    methods = ["m0", "m1", "m2"]
    rows = []
    for signal in ("entropy", "score"):
        for slice_, src in (("LOW", [3, 2, 1]), ("MID", [1, 2, 3]), ("HIGH", [3, 1, 2])):
            for m, v in zip(methods, src):
                rows.append({"mode": "global", "signal": signal, "slice": slice_,
                             "method": m, "map": v})  # fmt: skip
    for m, v in zip(methods, [3, 2, 1]):
        rows.append({"mode": "-", "signal": "-", "slice": "FULL", "method": m, "map": v})
    out = stage3(methods, pd.DataFrame(rows), targets)
    mid = out[(out.signal == "entropy") & (out.rule == "MID")].iloc[0]
    assert mid.pick == "m2"
    assert mid.regret_sum == pytest.approx(-1.0)
    assert mid.recovery_pct == pytest.approx(100.0 * (-1.0 + 3.0) / 3.0)


def test_protocol_ranking_correlates_with_cross_dataset_score():
    methods = ["m0", "m1", "m2"]
    kitti = np.array([50.0, 40.0, 30.0])
    targets = {"A": np.array([1.0, 3.0, 2.0]), "B": np.array([4.0, 6.0, 5.0])}
    out = protocol_ranking(methods, kitti, np.array([1.0, 3.0, 2.0]), targets)
    # Normalised scores: KITTI (1, .5, 0), A and B (0, 1, .5) -> mean (1/3, 5/6, 1/3).
    assert out.cross_dataset_score.tolist() == pytest.approx([1 / 3, 5 / 6, 1 / 3])
    assert out.rank_aggregate.tolist() == [1, 2, 3]
    assert out.rank_protocol.tolist() == [3, 1, 2]
    assert out.rho_aggregate.iloc[0] == pytest.approx(0.0)
    assert out.rho_protocol.iloc[0] > 0.8


def test_config_resolves_templates():
    cfg = load_config(CONFIG, data_root="/data", autolabeling_dir="/code")
    assert cfg.kitti_tfrecord == Path("/data/KITTI/tf/_val.tfrecord").resolve()
    assert cfg.checkpoint("Ent").name == cfg.methods["Ent"]
    assert cfg.predictions("Random").parts[-2] == f"EXP_{cfg.methods['Random']}"
    assert cfg.targets["CODA"].source_ids()[0] == 1
    assert len(TP_COLUMNS) == 10
