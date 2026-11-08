"""First- and second-derivative edge operators, from scratch.

Sign conventions (with ``correlate2d`` semantics): a left-dark /
right-bright vertical step gives a *positive* Sobel/Prewitt Gx, and
the 4-neighbour Laplacian answers +step on the dark side and -step
on the bright side — a sign change straddling the boundary, which
is exactly what the zero-crossing detector looks for.

The Sobel kernel is the [1 2 1] smoothing direction crossed with a
central difference; Prewitt is the same idea with uniform [1 1 1]
smoothing. Both kernels are used unnormalised (the classical
integer form), so a step of height h answers 4h (Sobel) and 3h
(Prewitt) — the tests pin those exact values.
"""

from __future__ import annotations

import numpy as np

from .convolution import correlate2d

SOBEL_GX = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]], dtype=float)
SOBEL_GY = np.array([[-1, -2, -1],
                     [0, 0, 0],
                     [1, 2, 1]], dtype=float)
PREWITT_GX = np.array([[-1, 0, 1],
                       [-1, 0, 1],
                       [-1, 0, 1]], dtype=float)
PREWITT_GY = np.array([[-1, -1, -1],
                       [0, 0, 0],
                       [1, 1, 1]], dtype=float)
LAPLACIAN_4 = np.array([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]], dtype=float)
LAPLACIAN_8 = np.array([[1, 1, 1],
                        [1, -8, 1],
                        [1, 1, 1]], dtype=float)

_KERNELS = {"sobel": (SOBEL_GX, SOBEL_GY), "prewitt": (PREWITT_GX, PREWITT_GY)}


def gradient(image: np.ndarray, operator: str = "sobel") -> dict:
    """Gradient of a greyscale image via the named 3x3 operator.

    Returns gx, gy, magnitude = sqrt(gx^2 + gy^2) and the gradient
    direction in degrees, folded into [0, 180) (an edge has no
    arrow: direction theta and theta + 180 are the same edge).
    """
    if operator not in _KERNELS:
        raise ValueError(f"operator must be one of {sorted(_KERNELS)}")
    kx, ky = _KERNELS[operator]
    gx = correlate2d(image, kx, padding="edge")
    gy = correlate2d(image, ky, padding="edge")
    magnitude = np.hypot(gx, gy)
    angle = np.degrees(np.arctan2(gy, gx)) % 180.0
    return {"gx": gx, "gy": gy, "magnitude": magnitude, "angle": angle}


def threshold_magnitude(magnitude: np.ndarray,
                        rel_threshold: float = 0.25) -> np.ndarray:
    """Binary edge map (bool): magnitude >= rel_threshold * max."""
    if magnitude.max() == 0:
        return np.zeros(magnitude.shape, dtype=bool)
    return magnitude >= rel_threshold * magnitude.max()


def laplacian(image: np.ndarray, neighbours: int = 4) -> np.ndarray:
    """Second-derivative response with the 4- or 8-neighbour kernel."""
    if neighbours == 4:
        ker = LAPLACIAN_4
    elif neighbours == 8:
        ker = LAPLACIAN_8
    else:
        raise ValueError("neighbours must be 4 or 8")
    return correlate2d(image, ker, padding="edge")


def laplacian_zero_crossings(response: np.ndarray, neighbours: int = 4,
                             rel_threshold: float = 0.25) -> np.ndarray:
    """Edge map (bool) from sign changes of the Laplacian response.

    A pixel is an edge pixel if it carries a nonzero response and
    has a neighbour (4- or 8-connected, matching the kernel) whose
    response has the opposite sign, with |difference| across the
    pair at least ``rel_threshold * max|response|`` — the strength
    condition that keeps faint noise ripples from counting as
    crossings (the classical Marr-Hildreth caveat: bare zero
    crossings are everywhere in noise).
    """
    if neighbours == 4:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                   (0, 1), (1, -1), (1, 0), (1, 1)]
    strength = rel_threshold * np.abs(response).max()
    padded = np.pad(response, 1, mode="constant")
    edges = np.zeros(response.shape, dtype=bool)
    h, w = response.shape
    for di, dj in offsets:
        nb = padded[1 + di:1 + di + h, 1 + dj:1 + dj + w]
        crossing = (response * nb < 0) & (np.abs(response - nb) >= strength)
        edges |= crossing
    return edges
