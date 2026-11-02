"""A few training steps on a tiny fake dataset (CPU): outputs of a run, skipping, low-label
subset, and the same run from both data paths and after an interruption."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.train.batches import BatchOrder
from src.train.loop import TrainConfig, train_run

N, SIZE = 12, 16


@pytest.fixture
def processed(tmp_path):
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, size=(N, SIZE, SIZE)).astype(np.uint8)
    labels[:, :2] = 19  # void rows
    emb = rng.standard_normal((N, 4, SIZE, SIZE)) + labels[:, None]  # inputs that carry the label
    instances = np.zeros((N, SIZE, SIZE), dtype=np.int32)
    instances[:, :, :8], instances[:, :, 8:] = 1, 2
    np.save(tmp_path / "labels.npy", labels)
    np.save(tmp_path / "valid_mask.npy", labels != 19)
    np.save(tmp_path / "instances.npy", instances)
    np.save(tmp_path / "ae_2019.npy", emb.astype(np.float16))
    splits = {"official": {"I": {"train": list(range(8)), "val": [8, 9], "test": [10, 11]}},
              "leave_one_tile_out": {}}
    (tmp_path / "splits.json").write_text(json.dumps(splits))
    return tmp_path


def config(**kw):
    base = dict(sources=["ae_2019"], variant="unet-test", widths=(4, 8, 16), steps=6, eval_every=2,
                checkpoint_every=2, batch_size=4, eval_batch_size=4, amp=False, stats_patches=8, workers=0)
    return TrainConfig(**{**base, **kw})


def run(processed, name, cfg, **kw):
    return train_run(cfg, processed, processed / name, processed / f"{name}.csv", device="cpu",
                     log=lambda *a: None, **kw)


def run_info(processed, name, row):
    return json.loads((processed / name / row["run_id"] / "run.json").read_text())


def test_batch_order():
    order = BatchOrder(10, 4, seed=0)
    assert order.per_epoch == 2
    epoch0 = torch.cat([order.positions(0), order.positions(1)]).tolist()
    assert len(set(epoch0)) == 8  # no patch twice in an epoch
    assert torch.equal(BatchOrder(10, 4, seed=0).positions(3), order.positions(3))
    assert not torch.equal(BatchOrder(10, 4, seed=1).positions(0), order.positions(0))
    assert BatchOrder(3, 4, seed=0).batch_size == 3  # fewer patches than the batch size


def test_run_outputs_and_skip(processed):
    row = run(processed, "runs", config())
    assert row["run_id"] == "base_unet-test-w4_100_foldI_s0" and 0 <= row["miou"] <= 1
    d = processed / "runs" / row["run_id"]
    assert (d / "best.pt").exists() and (d / "cm_per_patch.npz").exists() and not (d / "last.pt").exists()
    info = run_info(processed, "runs", row)
    assert [h["step"] for h in info["history"]] == [2, 4, 6] and info["best_step"] in (2, 4, 6)
    assert info["n_test_patches"] == 2 and len(info["train_patches"]) == 8
    again = run(processed, "runs", config())
    assert again["finished_utc"] == info["finished_utc"]  # skipped, not trained again


def test_low_label_subset(processed):
    row = run(processed, "runs", config(label_frac=0.25, head="mlp", variant="mlp-test", hidden=8))
    assert row["run_id"] == "base_mlp-test_025_foldI_s0"
    info = run_info(processed, "runs", row)
    assert info["subset"]["n_patches"] == 2 and len(info["train_patches"]) == 2


def test_same_run_from_both_data_paths_and_after_interruption(processed):
    a = run(processed, "a", config())
    b = run(processed, "b", config(data_on_device=False))
    assert run(processed, "c", config(), stop_at=3) is None
    assert (processed / "c" / a["run_id"] / "last.pt").exists()
    c = run(processed, "c", config())
    best_a = torch.load(processed / "a" / a["run_id"] / "best.pt")
    for name, row in (("b", b), ("c", c)):
        assert run_info(processed, name, row)["history"] == run_info(processed, "a", a)["history"]
        best = torch.load(processed / name / row["run_id"] / "best.pt")
        assert all(torch.equal(best[k], best_a[k]) for k in best_a)
