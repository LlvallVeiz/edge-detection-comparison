"""CLI: run every method on the synthetic image and compare.

Measurements (all against the ground truth in ``testimage.py``):

* **edge-pixel counts** per method, on the clean image and on the
  Gaussian-noise variant (sigma = 10) — the cheapest summary of how
  much structure (and how much noise) each method reports;
* **step-edge localisation** — on the scan row through the
  rectangle, the column where |Gx| peaks, against the true boundary
  at column 15.5; for Canny, the edge columns found on that row.

Fixed decision rules, stated so the counts are reproducible:
Sobel / Prewitt threshold at 0.25 x max magnitude; Laplacian
zero crossings need a sign change with |difference| >= 0.25 x
max|response|; Canny thresholds at 0.05 / 0.15 x post-NMS maximum
after a sigma = 1.0, 5-tap Gaussian.

Outputs (into --outdir, default: current directory): input.png,
input_noisy.png, edges_{sobel,prewitt,laplacian,canny}.png (clean
image results), comparison.png, and the table on stdout.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image

from . import canny as canny_mod
from . import operators, testimage

REL_THRESHOLD = 0.25
NOISE_SIGMA = 10.0


def _save(arr: np.ndarray, path: str) -> None:
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8),
                    mode="L").save(path)


def _save_edges(edges: np.ndarray, path: str) -> None:
    _save(edges.astype(float) * 255.0, path)


def _run_methods(image: np.ndarray) -> dict:
    out = {}
    for name in ("sobel", "prewitt"):
        grad = operators.gradient(image, operator=name)
        out[name] = {
            "edges": operators.threshold_magnitude(grad["magnitude"],
                                                   REL_THRESHOLD),
            "magnitude": grad["magnitude"],
            "gx": grad["gx"],
        }
    for nb in (4, 8):
        resp = operators.laplacian(image, neighbours=nb)
        out[f"laplacian{nb}"] = {
            "edges": operators.laplacian_zero_crossings(
                resp, neighbours=nb, rel_threshold=REL_THRESHOLD),
            "response": resp,
        }
    c = canny_mod.canny(image)
    out["canny"] = {"edges": c["edges"], "detail": c}
    return out


def _localisation(clean: dict, gt: dict) -> dict:
    row = gt["scan_row"]
    result = {}
    for name in ("sobel", "prewitt"):
        profile = np.abs(clean[name]["gx"][row, :40])
        result[name] = int(np.argmax(profile))
    canny_cols = np.flatnonzero(clean["canny"]["edges"][row, :40])
    result["canny_cols"] = [int(c) for c in canny_cols]
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Sobel vs Prewitt vs Laplacian vs Canny, from scratch.")
    ap.add_argument("--outdir", default=".",
                    help="where the PNG outputs go")
    args = ap.parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)

    gt = testimage.ground_truth()
    image = testimage.make_test_image()
    noisy = testimage.add_noise(image, sigma=NOISE_SIGMA)
    clean = _run_methods(image)
    noisy_res = _run_methods(noisy)
    loc = _localisation(clean, gt)

    print("=== edge-pixel counts (128x128 synthetic image) ===")
    print(f"{'method':<14}{'clean':>8}{'noisy (sigma=10)':>18}")
    for name in ("sobel", "prewitt", "laplacian4", "laplacian8", "canny"):
        print(f"{name:<14}{int(clean[name]['edges'].sum()):>8}"
              f"{int(noisy_res[name]['edges'].sum()):>18}")
    print()
    print("=== step-edge localisation (scan row "
          f"{gt['scan_row']}, true boundary at column "
          f"{gt['step_boundary_col']}) ===")
    for name in ("sobel", "prewitt"):
        col = loc[name]
        print(f"{name:<14}peak |Gx| at column {col}"
              f"  (error {abs(col - gt['step_boundary_col']):.1f} px)")
    print(f"canny         edge columns on scan row: {loc['canny_cols']}"
          f"  (error "
          f"{min(abs(c - gt['step_boundary_col']) for c in loc['canny_cols']):.1f}"
          " px)")
    c = clean["canny"]["detail"]
    print()
    print(f"canny thresholds used: low {c['low']:.1f}, high {c['high']:.1f}"
          " (0.05 / 0.15 x post-NMS max)")

    _save(image, os.path.join(args.outdir, "input.png"))
    _save(noisy, os.path.join(args.outdir, "input_noisy.png"))
    _save_edges(clean["sobel"]["edges"],
                os.path.join(args.outdir, "edges_sobel.png"))
    _save_edges(clean["prewitt"]["edges"],
                os.path.join(args.outdir, "edges_prewitt.png"))
    _save_edges(clean["laplacian4"]["edges"],
                os.path.join(args.outdir, "edges_laplacian.png"))
    _save_edges(clean["canny"]["edges"],
                os.path.join(args.outdir, "edges_canny.png"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [
        (image, "input (clean)"),
        (clean["sobel"]["edges"], "Sobel"),
        (clean["prewitt"]["edges"], "Prewitt"),
        (clean["laplacian4"]["edges"], "Laplacian-4 zero crossings"),
        (clean["canny"]["edges"], "Canny (clean)"),
        (noisy_res["canny"]["edges"], "Canny (noisy)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.8))
    for ax, (arr, title) in zip(axes.flat, panels):
        ax.imshow(arr, cmap="gray", vmin=0,
                  vmax=255 if arr.dtype != bool else 1)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "comparison.png"), dpi=150)
    plt.close(fig)
    print()
    print("wrote input.png, input_noisy.png, edges_sobel.png, "
          "edges_prewitt.png, edges_laplacian.png, edges_canny.png, "
          "comparison.png")
    return 0
