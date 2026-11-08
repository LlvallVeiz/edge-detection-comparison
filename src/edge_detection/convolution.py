"""2-D correlation / convolution with numpy — the only filtering
primitive this project uses.

Every operator in ``operators.py`` and every stage of the Canny
pipeline in ``canny.py`` is expressed through ``correlate2d`` here;
there are no OpenCV / scikit-image calls anywhere in the project.

Definitions (stated once, used consistently):

* ``correlate2d`` slides the kernel over the image *as written*:
  out[i, j] = sum_{u,v} kernel[u, v] * img[i + u - kh//2, j + v - kw//2].
  This is what most image-processing libraries call filtering
  (e.g. cv2.filter2D), and it is the natural form for gradient
  kernels, whose sign conventions are defined in this orientation.
* ``convolve2d`` is the mathematician's convolution: the kernel is
  flipped on both axes first. For the symmetric Laplacian and the
  antisymmetric gradient kernels the two differ at most in sign;
  for the symmetric Gaussian used in Canny they coincide.

Kernels must be odd-sized in both dimensions so they have a centre.
Padding is either ``"zero"`` (out-of-image samples are 0) or
``"edge"`` (replicate the nearest border pixel).
"""

from __future__ import annotations

import numpy as np

_PAD_MODES = {"zero": "constant", "edge": "edge"}


def _prepare(image: np.ndarray, kernel: np.ndarray, padding: str):
    img = np.asarray(image, dtype=float)
    ker = np.asarray(kernel, dtype=float)
    if img.ndim != 2:
        raise ValueError("image must be a 2-D array")
    if ker.ndim != 2 or ker.shape[0] % 2 == 0 or ker.shape[1] % 2 == 0:
        raise ValueError("kernel must be 2-D with odd dimensions")
    if padding not in _PAD_MODES:
        raise ValueError(f"padding must be one of {sorted(_PAD_MODES)}")
    ph, pw = ker.shape[0] // 2, ker.shape[1] // 2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode=_PAD_MODES[padding])
    return img, ker, padded


def correlate2d(image: np.ndarray, kernel: np.ndarray,
                padding: str = "zero") -> np.ndarray:
    """Slide ``kernel`` over ``image`` as written (no flipping)."""
    img, ker, padded = _prepare(image, kernel, padding)
    h, w = img.shape
    out = np.zeros((h, w), dtype=float)
    for u in range(ker.shape[0]):
        for v in range(ker.shape[1]):
            if ker[u, v] != 0.0:
                out += ker[u, v] * padded[u:u + h, v:v + w]
    return out


def convolve2d(image: np.ndarray, kernel: np.ndarray,
               padding: str = "zero") -> np.ndarray:
    """True convolution: correlate with the kernel flipped both ways."""
    ker = np.asarray(kernel, dtype=float)
    return correlate2d(image, ker[::-1, ::-1], padding=padding)
