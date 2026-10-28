"""Unit tests of src/eval on small examples computed by hand. Run: python -m pytest -q tests"""

import csv

import numpy as np
import pytest

from src.eval.bootstrap import bootstrap_miou, paired_delta
from src.eval.metrics import (CROPS, confusion_matrix, crops_only, miou, per_class_f1, per_class_iou,
                              summarize)
from src.eval.regions import boundary_core
from src.eval.runs import COLUMNS, RunEvaluator, log_run, run_name


def test_confusion_matrix_ignores_void_and_invalid():
    y_true = np.array([[0, 1], [19, 1]])
    y_pred = np.array([[0, 0], [1, 1]])
    cm = confusion_matrix(y_true, y_pred)
    assert cm.shape == (19, 19) and cm.sum() == 3
    assert cm[0, 0] == 1 and cm[1, 0] == 1 and cm[1, 1] == 1
    cm = confusion_matrix(y_true, y_pred, valid=np.array([[True, False], [True, True]]))
    assert cm.sum() == 2 and cm[1, 0] == 0


def test_confusion_matrix_rejects_out_of_range_prediction():
    with pytest.raises(ValueError):
        confusion_matrix(np.array([0, 1]), np.array([0, 19]))


def test_iou_and_f1_by_hand():
    cm = np.array([[2, 1], [0, 3]])
    assert per_class_iou(cm) == pytest.approx([2 / 3, 3 / 4])
    assert per_class_f1(cm) == pytest.approx([4 / 5, 6 / 7])
    assert miou(cm) == pytest.approx((2 / 3 + 3 / 4) / 2)


def test_absent_class_is_left_out():
    cm = np.array([[2, 1, 0], [0, 3, 0], [0, 0, 0]])
    iou = per_class_iou(cm)
    assert np.isnan(iou[2])
    assert miou(cm) == pytest.approx((2 / 3 + 3 / 4) / 2)


def test_predicted_but_absent_class_counts_as_zero():
    cm = np.array([[2, 1, 1], [0, 3, 0], [0, 0, 0]])  # class 2 predicted once, never true
    assert per_class_iou(cm)[2] == 0
    assert miou(cm) == pytest.approx((2 / 4 + 3 / 4 + 0) / 3)


def test_summarize_and_crops_only():
    cm = np.zeros((19, 19), dtype=int)
    cm[0, 0], cm[0, 1], cm[1, 1], cm[1, 0], cm[2, 2] = 10, 2, 5, 1, 4
    s = summarize(cm)
    assert s["oa"] == pytest.approx(19 / 22) and s["pixels"] == 22
    c = crops_only(cm)
    assert c[0].sum() == 0 and c[1, 0] == 1  # crop pixels predicted as background stay errors
    s = summarize(c, CROPS)
    # class 1: tp 5, fp 0 (the background row is removed), fn 1 -> 5/6; class 2: 4/4
    assert s["miou"] == pytest.approx((5 / 6 + 1) / 2)


def test_boundary_and_core():
    inst = np.array([[1, 1, 1, 2, 2, 2]] * 4)
    boundary, core = boundary_core(inst, width=1)
    assert boundary[:, [2, 3]].all() and not boundary[:, [0, 1, 4, 5]].any()
    assert core[:, [0, 1, 4, 5]].all() and not core[:, [2, 3]].any()
    boundary, core = boundary_core(inst, width=2)
    assert boundary[:, 1:5].all() and core[:, [0, 5]].all()
    inst = np.array([[0, 0, 3, 3]] * 2)
    boundary, core = boundary_core(inst)
    assert not boundary[:, :2].any() and not core[:, :2].any()  # background is in neither region
    assert boundary[:, 2].all() and core[:, 3].all()


def test_bootstrap_of_identical_patches_is_constant():
    cm = np.array([[5, 1], [2, 7]])
    cms = np.stack([cm] * 10)
    out = bootstrap_miou(cms, n_boot=200)
    assert out["ci_low"] == pytest.approx(out["miou"]) and out["ci_high"] == pytest.approx(out["miou"])


def test_paired_delta():
    rng = np.random.default_rng(0)
    cms = rng.integers(0, 20, size=(30, 3, 3))
    same = paired_delta(cms, cms, n_boot=200)
    assert same["delta"] == 0 and same["ci_low"] == 0 and same["ci_high"] == 0
    better = cms.copy()
    better[:, np.arange(3), np.arange(3)] += 50  # more correct pixels in every class
    out = paired_delta(cms, better, n_boot=200)
    assert out["delta"] > 0 and out["ci_low"] > 0 and out["share_le_0"] == 0
    with pytest.raises(ValueError):
        paired_delta(cms, cms[:-1])


def test_run_evaluator_and_log_run(tmp_path):
    labels = np.zeros((2, 4, 4), dtype=np.uint8)
    labels[:, :, 2:] = 1
    labels[1, 0, 0] = 19
    instances = np.zeros((2, 4, 4), dtype=np.int32)
    instances[:, :, 2:] = 5
    valid = labels != 19
    ev = RunEvaluator(labels, instances, valid)
    for i in range(2):
        ev.add(i, labels[i].astype(int) % 19)  # perfect prediction
    idx, cms = ev.result()
    assert cms.shape == (2, 3, 19, 19)
    run = {"run_id": run_name("base", "perfect", 1.0, "foldI", 0), "method": "baseline", "variant": "perfect",
           "label_frac": 1.0, "split": "foldI", "seed": 0, "n_params": 0, "train_seconds": 0}
    assert run["run_id"] == "base_perfect_100_foldI_s0"
    row = log_run(tmp_path / "runs.csv", tmp_path / run["run_id"], run, idx, cms)
    assert row["miou"] == pytest.approx(1.0) and row["oa"] == pytest.approx(1.0)
    with open(tmp_path / "runs.csv") as f:
        lines = list(csv.DictReader(f))
    assert list(lines[0]) == COLUMNS and float(lines[0]["miou"]) == pytest.approx(1.0)
    assert (tmp_path / run["run_id"] / "cm_per_patch.npz").exists()
    log_run(tmp_path / "runs.csv", tmp_path / run["run_id"], run, idx, cms)  # rerun: same line replaced
    with open(tmp_path / "runs.csv") as f:
        assert len(list(csv.DictReader(f))) == 1
