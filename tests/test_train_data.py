"""Input stack, targets and class weights on a small fake `processed` directory."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.train.data import IGNORE, InputStack, PatchDataset, class_weights, noise_patch

SIZE = 8


def fake_processed(path, n=4):
    rng = np.random.default_rng(0)
    q = rng.integers(-127, 127, size=(n, 128, SIZE, SIZE)).astype(np.int8)
    scale = rng.uniform(0.01, 0.1, size=(n, SIZE, SIZE)).astype(np.float32)
    scale[1, 0, 0] = np.nan  # one no-data pixel
    ae = rng.standard_normal((n, 64, SIZE, SIZE)).astype(np.float16)
    labels = rng.integers(0, 4, size=(n, SIZE, SIZE)).astype(np.uint8)
    labels[:, 0, 1] = 19
    valid = labels != 19
    valid[1, 0, 0] = False
    np.save(path / "tessera_2019_q.npy", q)
    np.save(path / "tessera_2019_scale.npy", scale)
    np.save(path / "ae_2019.npy", ae)
    return q, scale, ae, labels, valid


def batch(stack, i):
    return {k: torch.from_numpy(v)[None] for k, v in stack.read(i).items()}


def test_decode_matches_the_stored_values(tmp_path):
    q, scale, ae, _, _ = fake_processed(tmp_path)
    stack = InputStack(tmp_path, ["tessera_2019", "ae_2019"], size=SIZE)
    assert stack.channels == [128, 64] and stack.in_channels == 192
    x = stack.decode(batch(stack, 2))  # mean 0 and std 1 before fit_stats: plain decoding
    expected = np.concatenate([q[2] * scale[2][None], ae[2].astype(np.float32)])
    np.testing.assert_allclose(x[0].numpy(), expected, rtol=1e-6)
    with pytest.raises(FileNotFoundError):
        InputStack(tmp_path, ["tessera_2018"], size=SIZE)
    with pytest.raises(ValueError):
        InputStack(tmp_path, ["ae_2019"], size=128)


def test_standardisation_and_no_data(tmp_path):
    _, _, _, _, valid = fake_processed(tmp_path)
    stack = InputStack(tmp_path, ["tessera_2019", "ae_2019"], size=SIZE).fit_stats(range(4), valid)
    assert stack.fit_patches.tolist() == [0, 1, 2, 3]
    assert len(InputStack(tmp_path, ["ae_2019"], size=SIZE).fit_stats(range(4), valid, max_patches=2).fit_patches) == 2
    xs = torch.cat([stack.decode(batch(stack, i))[0][:, torch.from_numpy(valid[i])] for i in range(4)], 1)
    mean = xs.mean(1)
    std = ((xs - mean[:, None]) ** 2).mean(1).sqrt()
    assert mean.abs().max() < 1e-4 and (std - 1).abs().max() < 1e-3
    x1 = stack.decode(batch(stack, 1))
    assert torch.isfinite(x1).all() and x1[0, :128, 0, 0].abs().max() == 0  # no-data pixel -> training mean
    other = InputStack(tmp_path, ["tessera_2019", "ae_2019"], size=SIZE).load_state(stack.state())
    assert torch.equal(other.mean, stack.mean) and torch.equal(other.std, stack.std)
    with pytest.raises(ValueError):
        InputStack(tmp_path, ["ae_2019"], size=SIZE).load_state(stack.state())


def test_noise_is_fixed_per_seed_and_patch(tmp_path):
    stack = InputStack(tmp_path, ["noise"], noise_seed=3, size=SIZE)
    assert stack.in_channels == 64
    same = np.random.default_rng([3, 2]).standard_normal((64, SIZE, SIZE), dtype=np.float32)
    np.testing.assert_array_equal(stack.read(2)["noise"], same)
    np.testing.assert_array_equal(noise_patch(3, 2, SIZE), same)
    assert not np.array_equal(stack.read(1)["noise"], same)


def test_dataset_targets_and_batches(tmp_path):
    _, _, _, labels, valid = fake_processed(tmp_path)
    stack = InputStack(tmp_path, ["tessera_2019", "noise"], size=SIZE)
    ds = PatchDataset(stack, labels, valid, [0, 1, 3])
    _, target, i = ds[1]
    assert i == 1 and target.dtype == np.int64
    assert target[0, 0] == IGNORE and target[0, 1] == IGNORE  # no-data pixel and void pixel
    assert (target[valid[1]] == labels[1][valid[1]]).all()
    raw, target, idx = next(iter(torch.utils.data.DataLoader(ds, batch_size=2)))
    assert raw["tessera_2019"].dtype == torch.int8 and raw["tessera_2019/scale"].shape == (2, SIZE, SIZE)
    assert stack.decode(raw).shape == (2, 192, SIZE, SIZE) and target.shape == (2, SIZE, SIZE)
    assert idx.tolist() == [0, 1]


def test_class_weights():
    labels = np.array([[[0, 0, 0, 0], [1, 1, 2, 19]]], dtype=np.uint8)
    w, counts = class_weights(labels, labels != 19, [0], n_classes=4)
    assert counts.tolist() == [4, 2, 1, 0]
    freq = np.array([4, 2, 1]) / 7
    assert w[:3] == pytest.approx(1 / np.sqrt(freq) / (freq / np.sqrt(freq)).sum())
    assert w[3] == 0 and (counts * w).sum() / counts.sum() == pytest.approx(1)
