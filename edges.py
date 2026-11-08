#!/usr/bin/env python3
"""Entry point: edge-detection comparison on a synthetic test image.

Usage:
    python3 edges.py [--outdir examples]

Runs Sobel, Prewitt, Laplacian (4/8) and a from-scratch Canny
pipeline on a synthetic image with known geometry (plus a noisy
variant), prints edge-pixel counts and step-edge localisation
against the ground truth, and writes the input/output PNGs and a
comparison figure into the output directory.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from edge_detection import cli  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(cli.main(sys.argv[1:]))
