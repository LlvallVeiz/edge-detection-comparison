# Background & Research — Edge Detection Comparison

This document records the primary sources each method is built
on, the exact conventions implemented, and where this project
simplifies relative to the originals. Every source below was
consulted (and its URL or DOI checked) while writing the code.

## 1. Sobel–Feldman (1968): an operator with an unusual paper trail

The 3×3 gradient operator universally called "Sobel" was presented
by **Irwin Sobel and Gary Feldman** at the Stanford Artificial
Intelligence Project in 1968, in a talk titled *A 3×3 Isotropic
Gradient Operator for Image Processing* — and never formally
published by them. Sobel's own history note [2] records that its
first appearance in print was as a footnote credit ("suggested by
I. Sobel") in Duda & Hart's *Pattern Classification and Scene
Analysis* (1973, pp. 271–272), and that Raj Reddy coined the name
"Sobel operator". (At least one modern web article invents a 1968
journal paper by a "George Sobel"; no such paper exists, and the
history note is the authoritative account.)

The design goal was *isotropy*: a gradient estimate less dependent
on edge orientation than the 2×2 Roberts cross then in use. The
kernel factorises as a central difference crossed with a smoothing
strip,

```text
Gx = [1 2 1]^T x [-1 0 1]  =  [[-1 0 1], [-2 0 2], [-1 0 1]]
```

so it is separable — two 1-D passes instead of a 3×3 convolution,
an implementation property this project does *not* exploit (the
shared 3×3 correlator is kept deliberately uniform), but the
factorisation is why the centre row/column carries weight 2.
For an ideal step of height h the response at the two boundary
columns is 4h; `tests/test_operators.py` asserts 800 for h = 200.

## 2. Prewitt (1970)

Judith M. S. Prewitt's operator comes from her chapter *Object
Enhancement and Extraction* in *Picture Processing and
Psychopictorics* (Lipkin & Rosenfeld, eds., Academic Press, 1970,
pp. 75–149) [4]. It is the same construction with uniform
smoothing `[1 1 1]` in place of `[1 2 1]` — historically derived
from fitting a quadratic surface over the 3×3 neighbourhood and
taking its gradient. Step response 3h (asserted: 600 for h = 200).

The comparison's one systematic Sobel/Prewitt difference — Prewitt
marking ~10% more edge pixels under an identical *relative*
threshold — follows from these constants, not from either operator
being "better": a threshold at 0.25 × own maximum is 0.75h for
Prewitt versus 1.0h for Sobel in absolute terms.

## 3. The Laplacian and zero crossings: Marr & Hildreth (1980)

The Laplacian is the only rotation-invariant second-order
differential operator; across an ideal step its response is a
positive lobe on one side and a negative lobe on the other, so the
edge sits at the **zero crossing** between them. Marr & Hildreth's
*Theory of Edge Detection* [5] built the canonical detector on
this: detect the zero crossings of `∇²G * I`, the Laplacian of a
Gaussian-smoothed image, at several scales.

Two documented simplifications separate this project from [5]:

* **No Gaussian.** The discrete 4-/8-neighbour Laplacian is
  applied to the raw image, so the experiment doubles as a
  demonstration of *why* the smoothing is there: injected
  σ = 10 noise raises 4-neighbour crossing counts by 41%, against
  +1% for the first-derivative methods. (Second differences
  amplify exactly the high frequencies noise lives in.)
* **A strength gate instead of a scale.** A crossing counts only
  if the response difference across it reaches 0.25 × max
  |response| — the practical stand-in, also described in
  textbook treatments [7], for Marr & Hildreth's observation that
  not every zero crossing corresponds to a physical edge.

On the clean step the detector is exact in the test fixture:
responses +200 / −200 straddle the boundary and the marked
columns are precisely the two boundary columns.

## 4. Canny (1986): criteria first, pipeline second

Canny's paper [1] is unusual in starting from *criteria* — good
detection (low error rate), good localisation (marked points near
the true edge), and a single response to a single edge — and
deriving an operator (approximately the first derivative of a
Gaussian) plus the machinery this project implements as its five
stages: Gaussian smoothing, gradient estimation, **non-maximum
suppression** along the gradient direction, **double threshold**,
and **hysteresis** (thresholding by connectivity), which the paper
introduces to stop edges streaking into broken fragments. The
stage list and the hysteresis rule (a between-thresholds pixel
survives iff connected to an above-threshold pixel) match the
summary in the OpenCV tutorial [6], which this project used as a
cross-check while using no OpenCV code.

Documented simplifications relative to [1]:

* gradients from the Sobel operator rather than a sampled
  derivative-of-Gaussian (the standard practical substitution,
  also what OpenCV does);
* thresholds as fractions of the post-NMS maximum rather than
  estimated from a noise model — the CLI prints the absolute
  values derived (25.9 / 77.6 on the test image) so nothing is
  hidden;
* NMS by 4-sector quantisation with ties kept, and the outer
  1-pixel frame discarded rather than extrapolated.

The hysteresis unit test is the paper's argument in miniature:
three weak pixels with the *same* value are kept or dropped
purely by whether a weak chain connects them to a strong pixel.

## 5. Conventions

*Filtering* throughout is correlation (kernel as written), the
convention of the image-processing literature and of
`cv2.filter2D`; `convolution.py` also provides true convolution
and the tests demonstrate the difference on a shift kernel.
Gradient kernels use edge-replicate padding inside the operators
(so border-adjacent edges are not halved by phantom zeros), and
zero padding in the raw convolution tests where hand computation
is the point. Sign conventions: Gx > 0 for a dark→bright step
left-to-right; Laplacian positive on the dark side of a step.
Textbook statements of all four methods follow Gonzalez & Woods
[7], ch. 10.

## References

1. Canny J (1986). A Computational Approach to Edge Detection.
   *IEEE Transactions on Pattern Analysis and Machine
   Intelligence* 8(6):679–698. DOI 10.1109/TPAMI.1986.4767851 —
   https://ieeexplore.ieee.org/document/4767851
2. Sobel I (2014). History and Definition of the so-called
   "Sobel Operator", more appropriately named the Sobel–Feldman
   Operator —
   https://skynet.ecn.purdue.edu/~ace/vip/History_and_Definition_of_Sobel_Operator.pdf
3. Sobel I, Feldman G (1968). A 3×3 Isotropic Gradient Operator
   for Image Processing. Talk, Stanford Artificial Intelligence
   Project; first print description in Duda RO, Hart PE (1973),
   *Pattern Classification and Scene Analysis*, Wiley,
   pp. 271–272 (documented in [2]).
4. Prewitt JMS (1970). Object Enhancement and Extraction. In
   Lipkin BS, Rosenfeld A (eds.), *Picture Processing and
   Psychopictorics*, Academic Press, pp. 75–149 —
   https://shop.elsevier.com/books/picture-processing-and-psychopictorics/lipkin/978-0-12-451550-5
5. Marr D, Hildreth E (1980). Theory of Edge Detection.
   *Proceedings of the Royal Society of London B*
   207(1167):187–217. DOI 10.1098/rspb.1980.0020 —
   https://pubmed.ncbi.nlm.nih.gov/6102765/
6. OpenCV documentation. *Canny Edge Detection* (tutorial: the
   five stages and the hysteresis connectivity rule) —
   https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
7. Gonzalez RC, Woods RE (2018). *Digital Image Processing*,
   4th ed., Pearson, ch. 10 (Image Segmentation: gradient and
   Laplacian operators, zero crossings, Canny).
