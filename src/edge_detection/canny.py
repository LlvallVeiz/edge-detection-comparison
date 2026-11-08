"""A simplified Canny pipeline, stage by stage (Canny 1986).

Stages, each a separate function so the tests can pin them
individually:

1. ``gaussian_blur``     — separable Gaussian smoothing (the noise
                           reduction Canny's optimality analysis
                           assumes; also what makes the later
                           fixed thresholds survivable in noise).
2. Sobel gradients       — from ``operators.gradient``.
3. ``non_max_suppression`` — thin the magnitude ridge: a pixel
                           survives only if it is a local maximum
                           along its own gradient direction
                           (quantised to 4 sectors). This is the
                           "single response" criterion made concrete.
4. ``double_threshold``  — split survivors into *strong* (>= high)
                           and *weak* (low <= . < high) candidates.
5. ``hysteresis``        — a weak pixel becomes an edge only if it
                           is 8-connected, possibly through other
                           weak pixels, to a strong one. Real edges
                           are contours; isolated weak responses
                           are noise.

Thresholds are fractions of the post-NMS maximum magnitude
(``low_frac`` / ``high_frac``), stated in the CLI output and the
README; production implementations expose them as absolute values
tuned per image domain instead.

The outer 1-pixel frame of the NMS output is zeroed: a pixel whose
gradient-direction neighbours fall outside the image cannot be
checked for maximality, and keeping it would paint a spurious
frame around every result.
"""

from __future__ import annotations

import numpy as np

from .convolution import correlate2d
from .operators import gradient


def gaussian_kernel_1d(size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Normalised 1-D Gaussian kernel of odd ``size``."""
    if size % 2 == 0:
        raise ValueError("kernel size must be odd")
    x = np.arange(size, dtype=float) - size // 2
    k = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def gaussian_blur(image: np.ndarray, size: int = 5,
                  sigma: float = 1.0) -> np.ndarray:
    """Separable Gaussian blur (rows then columns), edge padding."""
    k = gaussian_kernel_1d(size, sigma)
    tmp = correlate2d(image, k.reshape(1, -1), padding="edge")
    return correlate2d(tmp, k.reshape(-1, 1), padding="edge")


def non_max_suppression(magnitude: np.ndarray,
                        angle: np.ndarray) -> np.ndarray:
    """Keep only local maxima along the gradient direction.

    ``angle`` is in degrees, folded to [0, 180). Sectors and the
    neighbour pairs compared (offsets in (row, col)):

        [0, 22.5) + [157.5, 180) : (0, -1), (0, +1)     gradient ~horizontal
        [22.5, 67.5)             : (-1, -1), (+1, +1)  gradient ~45 deg
        [67.5, 112.5)            : (-1, 0), (+1, 0)    gradient ~vertical
        [112.5, 157.5)           : (-1, +1), (+1, -1)  gradient ~135 deg

    Ties are kept (>= both neighbours): on a perfect two-pixel-wide
    ridge both pixels are legitimate maxima, and the threshold /
    hysteresis stages decide from there.
    """
    h, w = magnitude.shape
    padded = np.pad(magnitude, 1, mode="constant")
    out = np.zeros_like(magnitude)
    sectors = [
        ((angle < 22.5) | (angle >= 157.5), (0, -1), (0, 1)),
        ((angle >= 22.5) & (angle < 67.5), (-1, -1), (1, 1)),
        ((angle >= 67.5) & (angle < 112.5), (-1, 0), (1, 0)),
        ((angle >= 112.5) & (angle < 157.5), (-1, 1), (1, -1)),
    ]
    for mask, (di1, dj1), (di2, dj2) in sectors:
        n1 = padded[1 + di1:1 + di1 + h, 1 + dj1:1 + dj1 + w]
        n2 = padded[1 + di2:1 + di2 + h, 1 + dj2:1 + dj2 + w]
        keep = mask & (magnitude >= n1) & (magnitude >= n2)
        out[keep] = magnitude[keep]
    out[0, :] = out[-1, :] = 0.0
    out[:, 0] = out[:, -1] = 0.0
    return out


def double_threshold(magnitude: np.ndarray, low: float, high: float):
    """Split into (strong, weak) boolean masks by two thresholds."""
    strong = magnitude >= high
    weak = (magnitude >= low) & ~strong
    return strong, weak


def hysteresis(strong: np.ndarray, weak: np.ndarray) -> np.ndarray:
    """Edge map: strong pixels plus weak pixels connected to them.

    Flood fill (8-connectivity) seeded at every strong pixel,
    spreading only through weak pixels. A weak pixel with no path
    to a strong one — however long the weak chain it sits in, if
    the chain never reaches a strong pixel — is dropped.
    """
    edges = strong.copy()
    h, w = strong.shape
    stack = [tuple(p) for p in np.argwhere(strong)]
    while stack:
        i, j = stack.pop()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ni, nj = i + di, j + dj
                if (0 <= ni < h and 0 <= nj < w
                        and weak[ni, nj] and not edges[ni, nj]):
                    edges[ni, nj] = True
                    stack.append((ni, nj))
    return edges


def canny(image: np.ndarray, sigma: float = 1.0,
          low_frac: float = 0.05, high_frac: float = 0.15,
          kernel_size: int = 5) -> dict:
    """Run the full pipeline; returns the edge map + intermediates."""
    blurred = gaussian_blur(image, size=kernel_size, sigma=sigma)
    grad = gradient(blurred, operator="sobel")
    nms = non_max_suppression(grad["magnitude"], grad["angle"])
    peak = nms.max()
    strong, weak = double_threshold(nms, low_frac * peak, high_frac * peak)
    edges = hysteresis(strong, weak)
    return {
        "edges": edges,
        "blurred": blurred,
        "magnitude": grad["magnitude"],
        "nms": nms,
        "strong": strong,
        "weak": weak,
        "low": low_frac * peak,
        "high": high_frac * peak,
    }
