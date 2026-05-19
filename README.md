# pseudospec

Pseudospectrum computation for normal and nonnormal matrices, based on algorithms from Trefethen & Embree, *Spectra and Pseudospectra* (Princeton University Press, 2005).

## Requirements

```
python    >= 3.9
numpy     >= 1.24
scipy     >= 1.10
matplotlib >= 3.8
shapely   >= 2.0
```

Install dependencies:

```bash
pip install numpy scipy matplotlib "shapely>=2.0"
```

---

## Core concept

The **ε-pseudospectrum** of a matrix A is the set of complex numbers z for which the resolvent norm exceeds 1/ε:

```
Λ_ε(A) = { z ∈ ℂ : ‖(A − zI)⁻¹‖ ≥ 1/ε }
```

For **normal** matrices this reduces to the union of discs of radius ε around the eigenvalues — cheap to compute via geometry. For **nonnormal** matrices the pseudospectrum can bulge far beyond the eigenvalues, and must be computed by evaluating the smallest singular value of (A − zI) across a grid of points.

---

## Normal matrices

Use `Normal` (from a matrix) or `NormalEvals` (from eigenvalues directly). Computation is exact and fast — no resolvent evaluations.

```python
import numpy as np
from matplotlib import pyplot
import pseudospec

# Build a normal matrix (e.g. skew-symmetric → purely imaginary spectrum)
A = np.array([[0, -2, 0],
              [2,  0, -1],
              [0,  1,  0]], dtype=float)

ps = pseudospec.Normal(A)

epsilons = [0.1, 0.3, 0.5, 0.8, 1.0]
ps.plot(epsilons, spectrum=ps.evals)

pyplot.title("Normal matrix pseudospectrum")
pyplot.xlabel("Re(z)")
pyplot.ylabel("Im(z)")
pyplot.show()
```

If you already have eigenvalues:

```python
evals = np.linalg.eigvals(A)
ps = pseudospec.NormalEvals(evals)
ps.plot([0.2, 0.5, 1.0])
pyplot.show()
```

---

## Nonnormal matrices

### Uniform grid (recommended starting point)

`NonnormalMeshgrid` evaluates the resolvent on a regular Cartesian grid. SVDs are parallelised automatically across available CPU cores.

```python
import numpy as np
from matplotlib import pyplot
import pseudospec

# Grcar matrix — a classic nonnormal example
def grcar(n, k=3):
    A = np.diag(-np.ones(n - 1), -1)
    for i in range(k + 1):
        A += np.diag(np.ones(n - i), i)
    return A

A = grcar(16)

ps = pseudospec.NonnormalMeshgrid(
    A,
    real_min=-1.5, real_max=3.5,
    imag_min=-2.5, imag_max=2.5,
    real_n=100, imag_n=100,     # grid resolution
    method="svd",               # "svd" | "lanczos" | "lanczosinv"
)

epsilons = [10**k for k in range(-4, 0)]   # 1e-4, 1e-3, 1e-2, 1e-1
ps.plot(epsilons, contour_labels=True)

pyplot.title("Grcar matrix pseudospectrum")
pyplot.xlabel("Re(z)")
pyplot.ylabel("Im(z)")
pyplot.show()
```

Control parallelism with `n_workers` (defaults to CPU count; pass `1` to disable):

```python
ps = pseudospec.NonnormalMeshgrid(A, ..., n_workers=4)
```

### Auto bounding-box

`NonnormalMeshgridAuto` derives a bounding box from the eigenvector condition number — useful when you don't know where to look.

```python
ps = pseudospec.NonnormalMeshgridAuto(
    A,
    eps_max=0.1,        # largest ε of interest
    real_n=80,
    imag_n=80,
)
ps.plot([1e-3, 1e-2, 1e-1])
pyplot.show()
```

> **Note:** the bounding box is guaranteed to contain the pseudospectrum but may be a significant overestimate for ill-conditioned eigenvector bases.

### Fully automatic (unknown territory)

`NonnormalAuto` determines both the bounding region and the evaluation points automatically. The best option when you have no prior knowledge of where the pseudospectrum lives.

```python
ps = pseudospec.NonnormalAuto(
    A,
    eps_min=1e-4,       # smallest ε to resolve
    eps_max=1e-1,       # largest ε of interest
    n_circles=20,       # logarithmically-spaced radii per eigenvalue
    n_points=20,        # evaluation points per circle
    randomize=True,     # rotate circles to reduce aliasing
)
ps.plot([1e-4, 1e-3, 1e-2, 1e-1])
pyplot.show()
```

### Unstructured points

Evaluate at an arbitrary cloud of complex points (auto-triangulated):

```python
# Dense sampling near the origin
rng = np.random.default_rng(0)
points = rng.uniform(-2, 2, 500) + 1j * rng.uniform(-2, 2, 500)

ps = pseudospec.NonnormalPoints(A, points)
ps.plot([1e-3, 1e-2, 1e-1])
pyplot.show()
```

---

## Extracting contour paths

All classes expose `contour_paths(epsilon)` for downstream geometry work (e.g. area calculations, intersection tests):

```python
paths = ps.contour_paths(epsilon=0.01)

print(f"{len(paths)} connected component(s)")
for path in paths:
    print(f"  arc length = {path.length():.4f},  {len(path)} vertices")

# All vertices as a flat list of complex numbers
all_verts = paths.vertices()
```

---

## Choosing a method

| Method | When to use |
|---|---|
| `"svd"` | Dense matrices (default). Exact minimal singular value via full SVD. |
| `"lanczos"` | Large sparse matrices. Iterative estimate; faster but approximate. |
| `"lanczosinv"` | Dense matrices where `A` is already Schur-decomposed internally. Uses inverse iteration; most accurate near the spectrum. |

---

## Package layout

```
pseudospec/
├── __init__.py      Public API and Shapely version check
├── _compat.py       Matplotlib / Shapely compatibility shims (internal)
├── normal.py        NormalEvals, Normal
├── nonnormal.py     NonnormalMeshgrid, NonnormalMeshgridAuto,
│                    NonnormalTriang, NonnormalPoints, NonnormalAuto
└── utils.py         Path, Paths, get_paths, plot_finish
```

---

## References

- Trefethen, L. N. & Embree, M. (2005). *Spectra and Pseudospectra: The Behavior of Nonnormal Matrices and Operators*. Princeton University Press.
- Grcar, J. F. (1989). Operator coefficient methods for linear equations. Sandia National Laboratories Report SAND89-8691.
