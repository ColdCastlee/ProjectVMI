"""Low-label subsets on a small synthetic training set."""

import numpy as np

from src.data.subsets import class_pixels, label_subset

N, SIZE = 40, 32


def fake_train():
    labels = np.zeros((N, SIZE, SIZE), dtype=np.uint8)  # background
    for k in range(N):
        labels[k, :16, :16] = 1 + k % 3  # classes 1, 2, 3: about 13 patches each
    labels[[3, 11], 16:, :16] = 4  # class 4 in two patches
    labels[7, 16:, 16:] = 5  # class 5 in one patch
    valid = np.ones_like(labels, dtype=bool)
    valid[:, 0, 0] = False
    return labels, valid, np.arange(N)


def test_class_pixels_use_valid_pixels_only():
    labels, valid, train = fake_train()
    pixels = class_pixels(labels, valid, train)
    assert pixels.shape == (N, 19) and pixels[0, 1] == 255 and pixels[7, 5] == 256


def test_covering_set_comes_first():
    labels, valid, train = fake_train()
    idx, info = label_subset(labels, valid, train, 3 / N, seed=0)
    assert info["n_cover"] == 3 and len(idx) == 3
    assert 7 in idx and (3 in idx or 11 in idx)
    assert info["missing_classes"] == list(range(6, 19))  # only classes absent from the whole training set
    idx, info = label_subset(labels, valid, train, 1 / N, seed=0)  # budget smaller than the covering set
    assert idx.tolist() == [7] and 4 in info["missing_classes"]  # the rarest class is taken first


def test_subsets_are_nested_and_reproducible():
    labels, valid, train = fake_train()
    for seed in range(3):
        sets = [set(label_subset(labels, valid, train, f, seed)[0].tolist()) for f in (0.1, 0.3, 0.5, 1.0)]
        assert all(a <= b for a, b in zip(sets, sets[1:])) and sets[-1] == set(range(N))
    a = label_subset(labels, valid, train, 0.3, seed=1)[0]
    assert np.array_equal(a, label_subset(labels, valid, train, 0.3, seed=1)[0])
    assert not np.array_equal(a, label_subset(labels, valid, train, 0.3, seed=2)[0])
