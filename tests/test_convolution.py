"""Convolution tests: every expected value computed by hand."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from edge_detection.convolution import convolve2d, correlate2d  # noqa: E402

IMG = np.array([[1., 2., 3., 4.],
                [5., 6., 7., 8.],
                [9., 10., 11., 12.],
                [13., 14., 15., 16.]])


def test_ones_kernel_hand_computed():
    out = correlate2d(IMG, np.ones((3, 3)), padding="zero")
    assert out[0, 0] == 14    # 1+2+5+6 (rest is padding)
    assert out[0, 3] == 22    # 3+4+7+8
    assert out[1, 1] == 54    # full 3x3 block: 1..3, 5..7, 9..11
    assert out[2, 2] == 99    # 6+7+8+10+11+12+14+15+16
    assert out[3, 3] == 54    # 11+12+15+16


def test_identity_and_constant_kernels():
    identity = np.array([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]])
    assert np.array_equal(correlate2d(IMG, identity), IMG)
    assert np.array_equal(convolve2d(IMG, identity), IMG)
    # edge padding on a constant image: every 3x3 window is full
    out = correlate2d(np.full((4, 4), 2.0), np.ones((3, 3)),
                      padding="edge")
    assert np.all(out == 18.0)
    # zero padding: corners see only 4 valid samples
    out = correlate2d(np.full((4, 4), 2.0), np.ones((3, 3)),
                      padding="zero")
    assert out[0, 0] == 8.0 and out[1, 1] == 18.0


def test_correlate_vs_convolve_shift_kernel():
    shift = np.array([[0., 0., 0.], [0., 0., 1.], [0., 0., 0.]])
    # correlation with the tap at [1, 2]: out[i, j] = img[i, j + 1]
    out = correlate2d(IMG, shift, padding="zero")
    expected = np.zeros_like(IMG)
    expected[:, :-1] = IMG[:, 1:]
    assert np.array_equal(out, expected)
    # convolution flips the kernel: out[i, j] = img[i, j - 1]
    out = convolve2d(IMG, shift, padding="zero")
    expected = np.zeros_like(IMG)
    expected[:, 1:] = IMG[:, :-1]
    assert np.array_equal(out, expected)


def test_convolve_is_flipped_correlate_in_general():
    rng = np.random.default_rng(0)
    img = rng.normal(size=(6, 7))
    ker = rng.normal(size=(3, 3))
    assert np.allclose(convolve2d(img, ker),
                       correlate2d(img, ker[::-1, ::-1]))


def test_even_kernel_rejected():
    for fn in (correlate2d, convolve2d):
        try:
            fn(IMG, np.ones((2, 2)))
        except ValueError:
            pass
        else:
            raise AssertionError("even kernel not rejected")


if __name__ == "__main__":
    test_ones_kernel_hand_computed()
    test_identity_and_constant_kernels()
    test_correlate_vs_convolve_shift_kernel()
    test_convolve_is_flipped_correlate_in_general()
    test_even_kernel_rejected()
    print("test_convolution: all passed")
