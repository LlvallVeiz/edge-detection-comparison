"""Operator tests on a synthetic step edge with hand-computed responses.

The fixture is a 10x12 image, columns 0..5 = 0 and 6..11 = 200:
a vertical step whose boundary sits between columns 5 and 6.
For a step of height h the unnormalised kernels answer, at the two
boundary columns:

    Sobel Gx   = 4h = 800      Prewitt Gx = 3h = 600
    Laplacian-4: +h on the dark side, -h on the bright side
    Laplacian-8: +3h / -3h = +600 / -600

and exactly 0 in the flat interior — all asserted literally below.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from edge_detection import operators  # noqa: E402


def _step():
    img = np.zeros((10, 12))
    img[:, 6:] = 200.0
    return img


def test_sobel_step_response():
    g = operators.gradient(_step(), "sobel")
    assert g["gx"][5, 5] == 800.0 and g["gx"][5, 6] == 800.0
    assert g["gx"][5, 4] == 0.0 and g["gx"][5, 7] == 0.0
    assert np.all(g["gy"] == 0.0)
    peak_cols = np.flatnonzero(g["magnitude"][5] == g["magnitude"][5].max())
    assert set(peak_cols.tolist()) == {5, 6}


def test_prewitt_step_response():
    g = operators.gradient(_step(), "prewitt")
    assert g["gx"][5, 5] == 600.0 and g["gx"][5, 6] == 600.0
    assert g["gx"][5, 2] == 0.0


def test_laplacian_step_response_and_zero_crossings():
    img = _step()
    r4 = operators.laplacian(img, neighbours=4)
    assert r4[5, 5] == 200.0 and r4[5, 6] == -200.0
    assert r4[5, 3] == 0.0
    r8 = operators.laplacian(img, neighbours=8)
    assert r8[5, 5] == 600.0 and r8[5, 6] == -600.0
    zc = operators.laplacian_zero_crossings(r4, neighbours=4)
    assert set(np.flatnonzero(zc.any(axis=0)).tolist()) == {5, 6}
    assert zc.sum() == 20  # both columns, all 10 rows


def test_thresholded_sobel_map_hugs_the_boundary():
    g = operators.gradient(_step(), "sobel")
    edges = operators.threshold_magnitude(g["magnitude"], 0.25)
    assert set(np.flatnonzero(edges.any(axis=0)).tolist()) == {5, 6}


def test_flat_image_has_no_edges():
    flat = np.full((8, 8), 128.0)
    g = operators.gradient(flat, "sobel")
    assert np.all(g["magnitude"] == 0.0)
    assert not operators.threshold_magnitude(g["magnitude"]).any()
    resp = operators.laplacian(flat, neighbours=4)
    assert np.all(resp == 0.0)
    assert not operators.laplacian_zero_crossings(resp).any()


if __name__ == "__main__":
    test_sobel_step_response()
    test_prewitt_step_response()
    test_laplacian_step_response_and_zero_crossings()
    test_thresholded_sobel_map_hugs_the_boundary()
    test_flat_image_has_no_edges()
    print("test_operators: all passed")
