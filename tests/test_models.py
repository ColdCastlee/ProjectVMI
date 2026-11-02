"""Shapes, parameter counts and receptive field of the heads."""

import pytest

torch = pytest.importorskip("torch")

from src.models.unet import LightUNet, PixelMLP, build_head, count_parameters


def test_unet_keeps_the_patch_size():
    model = LightUNet(8, n_classes=5, widths=(8, 16, 32)).eval()
    with torch.no_grad():
        assert model(torch.randn(2, 8, 32, 32)).shape == (2, 5, 32, 32)


def test_parameter_counts():
    # counted by hand: 1x1 input conv + 5 blocks of two 3x3 convs with batch norm + 1x1 output conv
    assert count_parameters(LightUNet(128)) == 485_267
    assert count_parameters(LightUNet(128, widths=(64, 128, 256))) == 1_928_979
    assert count_parameters(PixelMLP(128)) == 128 * 128 + 128 + 128 * 19 + 19


def outputs_changed(model, channels, pixel):
    x = torch.randn(1, channels, 16, 16)
    y = x.clone()
    y[0, :, pixel[0], pixel[1]] += 3.0
    with torch.no_grad():
        return (model(x) - model(y)).abs().sum(1)[0]  # (16, 16)


def test_pixel_mlp_has_no_spatial_context():
    diff = outputs_changed(PixelMLP(6, n_classes=4, hidden=8).eval(), 6, (5, 7))
    assert diff[5, 7] > 0
    diff[5, 7] = 0
    assert diff.max() < 1e-6


def test_unet_uses_the_neighbours():
    diff = outputs_changed(LightUNet(6, n_classes=4, widths=(8, 16, 32)).eval(), 6, (8, 8))
    assert diff[8, 11] > 0 and diff[4, 8] > 0


def test_build_head():
    assert isinstance(build_head("unet", 10), LightUNet)
    assert isinstance(build_head("mlp", 10), PixelMLP)
    with pytest.raises(ValueError):
        build_head("transformer", 10)
