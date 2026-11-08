"""A synthetic test image whose edges are known exactly.

Because every shape is drawn from stated geometry, method outputs
can be *checked*, not just eyeballed: the CLI measures where the
Sobel response to the rectangle's left edge actually peaks, against
the true boundary position recorded here.

Layout (128 x 128, background 0):

* **filled rectangle**, rows/cols 16..55, value 200 — its left side
  is a clean vertical step edge at column boundary 15.5 (between
  the 0 of column 15 and the 200 of column 16);
* **filled disk**, centre (row 92, col 94), radius 20, value 150 —
  curved edges in every direction;
* **anti-diagonal band**, |row + col - 140| <= 2 for rows 80..124,
  value 120 — a 45-degree edge pair;
* **thin vertical line**, column 96, rows 10..52, value 255 — a
  1-pixel-wide feature (two edges, two pixels apart).

The shapes do not overlap. ``add_noise`` produces the degraded
variant (additive Gaussian noise, clipped to [0, 255]) that the
comparison is also run on.
"""

from __future__ import annotations

import numpy as np

SIZE = 128
RECT = {"rows": slice(16, 56), "cols": slice(16, 56), "value": 200}
STEP_BOUNDARY_COL = 15.5     # between col 15 (0) and col 16 (200)
SCAN_ROW = 36                # middle row of the rectangle
DISK = {"centre": (92, 94), "radius": 20, "value": 150}
BAND = {"offset": 140, "half_width": 2, "rows": slice(80, 125), "value": 120}
LINE = {"col": 96, "rows": slice(10, 53), "value": 255}


def make_test_image() -> np.ndarray:
    """The clean synthetic image, float64 in [0, 255]."""
    img = np.zeros((SIZE, SIZE), dtype=float)
    img[RECT["rows"], RECT["cols"]] = RECT["value"]
    rr, cc = np.mgrid[0:SIZE, 0:SIZE]
    cr, cl = DISK["centre"]
    img[(rr - cr) ** 2 + (cc - cl) ** 2 <= DISK["radius"] ** 2] = DISK["value"]
    band = (np.abs(rr + cc - BAND["offset"]) <= BAND["half_width"])
    band[:BAND["rows"].start, :] = False
    band[BAND["rows"].stop:, :] = False
    img[band] = BAND["value"]
    img[LINE["rows"], LINE["col"]] = LINE["value"]
    return img


def ground_truth() -> dict:
    """The geometry the measurements are checked against."""
    return {
        "step_boundary_col": STEP_BOUNDARY_COL,
        "scan_row": SCAN_ROW,
        "rect": (16, 56, 16, 56),
        "disk_centre": DISK["centre"],
        "disk_radius": DISK["radius"],
        "line_col": LINE["col"],
    }


def add_noise(image: np.ndarray, sigma: float = 10.0,
              seed: int = 3) -> np.ndarray:
    """Additive Gaussian noise, seeded, clipped back to [0, 255]."""
    rng = np.random.default_rng(seed)
    noisy = image + rng.normal(0.0, sigma, size=image.shape)
    return np.clip(noisy, 0.0, 255.0)
