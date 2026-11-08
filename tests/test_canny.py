"""Canny stage tests: NMS ridge behaviour and hysteresis connectivity.

The hysteresis fixture is the textbook case (and the one the OpenCV
tutorial draws): a weak pixel *between* strong pixels is an edge; a
weak pixel in a chain that reaches a strong pixel is an edge; a
weak pixel with no path to any strong pixel is not — even though
its own value is identical to the kept ones.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from edge_detection import canny as C  # noqa: E402


def test_gaussian_kernel_properties():
    k = C.gaussian_kernel_1d(5, 1.0)
    assert abs(k.sum() - 1.0) < 1e-12
    assert np.allclose(k, k[::-1])
    assert k[2] == k.max()
    blurred = C.gaussian_blur(np.full((6, 6), 77.0))
    assert np.allclose(blurred, 77.0)   # a constant survives smoothing


def test_nms_keeps_ridge_suppresses_slopes():
    mag = np.zeros((7, 7))
    mag[1:6, 3] = [10, 30, 50, 30, 10]      # the ridge (vertical edge)
    mag[1:6, 2] = [5, 15, 25, 15, 5]        # its slopes
    mag[1:6, 4] = [5, 15, 25, 15, 5]
    angle = np.zeros((7, 7))                # gradient points horizontally
    out = C.non_max_suppression(mag, angle)
    assert out[3, 3] == 50.0                # ridge peak kept
    assert out[1, 3] == 10.0                # ridge end kept (>= both nb)
    assert out[3, 2] == 0.0 and out[3, 4] == 0.0   # slopes suppressed
    # same arrangement transposed: gradient vertical (90 deg)
    out = C.non_max_suppression(mag.T.copy(), np.full((7, 7), 90.0))
    assert out[3, 3] == 50.0 and out[2, 3] == 0.0


def test_double_threshold_bands():
    mag = np.array([[0.0, 5.0, 15.0, 25.0]])
    strong, weak = C.double_threshold(mag, low=10.0, high=20.0)
    assert strong.tolist() == [[False, False, False, True]]
    assert weak.tolist() == [[False, False, True, False]]


def test_hysteresis_connectivity_rules():
    strong = np.zeros((5, 5), dtype=bool)
    strong[2, 1] = strong[2, 3] = True
    weak = np.zeros((5, 5), dtype=bool)
    weak[2, 2] = True     # between two strong pixels -> edge
    weak[1, 1] = True     # diagonally adjacent to a strong one -> edge
    weak[0, 1] = True     # reaches strong only via another weak -> edge
    weak[0, 4] = True     # isolated -> NOT an edge
    edges = C.hysteresis(strong, weak)
    expected = strong.copy()
    expected[2, 2] = expected[1, 1] = expected[0, 1] = True
    assert np.array_equal(edges, expected)
    assert not edges[0, 4]


def test_canny_on_clean_step():
    img = np.zeros((10, 12))
    img[:, 6:] = 200.0
    result = C.canny(img)
    edges = result["edges"]
    assert edges.any()
    cols = set(np.flatnonzero(edges.any(axis=0)).tolist())
    assert cols <= {5, 6}, f"edge columns {cols}"


if __name__ == "__main__":
    test_gaussian_kernel_properties()
    test_nms_keeps_ridge_suppresses_slopes()
    test_double_threshold_bands()
    test_hysteresis_connectivity_rules()
    test_canny_on_clean_step()
    print("test_canny: all passed")
