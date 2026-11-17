"""A few training steps on a tiny fake dataset (CPU): outputs of a run, skipping, low-label
subset, the same run from both data paths and after an interruption, and the validation
matrices added to a run finished without them."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.train.batches import BatchOrder
from src.eval.metrics import miou
from src.eval.runs import load_cms
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
    assert (d / "cm_val_per_patch.npz").exists()
    info = run_info(processed, "runs", row)
    assert [h["step"] for h in info["history"]] == [2, 4, 6] and info["best_step"] in (2, 4, 6)
    assert info["n_test_patches"] == 2 and len(info["train_patches"]) == 8
    again = run(processed, "runs", config())
    assert again["finished_utc"] == info["finished_utc"]  # skipped, not trained again


def test_low_label_subset(processed):
    row = run(processed, "runs", config(label_frac=0.25))
    assert row["run_id"] == "base_unet-test-w4_025_foldI_s0"
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


def test_val_matrices_added_to_finished_run(processed):
    row = run(processed, "runs", config())
    d = processed / "runs" / row["run_id"]
    idx, cms = load_cms(d, part="val")
    assert idx.tolist() == [8, 9]
    assert miou(cms.sum(axis=0)) == pytest.approx(run_info(processed, "runs", row)["best_val_miou"], abs=1e-9)
    (d / "cm_val_per_patch.npz").unlink()  # as for a run finished before the matrices were saved
    again = run(processed, "runs", config())
    assert again["finished_utc"] == row["finished_utc"]  # not trained again
    idx2, cms2 = load_cms(d, part="val")
    assert np.array_equal(idx2, idx) and np.array_equal(cms2, cms)


def test_schedule_and_class_weighting(processed):
    from dataclasses import asdict

    from src.train.loop import comparable

    base = run_info(processed, "runs", run(processed, "runs", config(lr=1e-3)))
    lrs = [h["lr"] for h in base["history"]]
    assert lrs[0] < 1e-3 and lrs[-1] == pytest.approx(0, abs=1e-12)  # cosine to 0
    assert len(set(base["class_weights"])) > 1  # 1/sqrt(frequency)
    row = run(processed, "runs", config(variant="unet-test-const", lr="1e-3", schedule="constant",
                                        class_weighting="none"))
    info = run_info(processed, "runs", row)
    assert [h["lr"] for h in info["history"]] == [1e-3] * 3  # YAML reads 1e-3 as text: converted to float
    assert info["class_weights"] == [1.0] * len(info["class_weights"])
    with pytest.raises(ValueError):
        config(schedule="linear")
    with pytest.raises(ValueError):
        config(class_weighting="inverse")
    # a checkpoint written before these options resumes with the default recipe
    old = {k: v for k, v in asdict(config()).items() if k not in ("schedule", "class_weighting")}
    assert comparable(old) == comparable(asdict(config()))


def test_gated_run_and_late_fusion(processed):
    from scripts.m1_late_fusion import late_cms
    from src.train.loop import load_trained

    row = run(processed, "runs", config(sources=["ae_2019", "noise"], head="gated", variant="gated-test"))
    assert row["run_id"] == "base_gated-test-w4_100_foldI_s0"
    d = processed / "runs" / row["run_id"]
    # late fusion of a run with itself gives the predictions of that run
    member = load_trained(d, processed, torch.device("cpu"))
    labels = np.load(processed / "labels.npy")
    valid = np.load(processed / "valid_mask.npy")
    instances = np.load(processed / "instances.npy")
    idx, cms = late_cms([member, member], labels, instances, valid, [10, 11], torch.device("cpu"), 4, 0)
    idx0, cms0 = load_cms(d)
    assert np.array_equal(idx, idx0) and np.array_equal(cms, np.stack([cms0, load_cms(d, "boundary")[1],
                                                                        load_cms(d, "core")[1]], axis=1))
