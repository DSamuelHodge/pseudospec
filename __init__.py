"""pseudospec — Pseudospectrum computation for normal and nonnormal matrices.

Public API
----------
Normal matrices (exact, geometry-only):

    NormalEvals          eigenvalues → pseudospectrum
    Normal               matrix → eigenvalues → pseudospectrum

Nonnormal matrices (resolvent-based):

    NonnormalMeshgrid    uniform Cartesian grid
    NonnormalMeshgridAuto  auto bounding-box via eigenvector condition number
    NonnormalTriang      arbitrary triangulation
    NonnormalPoints      unstructured point cloud (auto-triangulated)
    NonnormalAuto        fully automatic inclusion-set discovery

Utility types:

    Path                 sequence of complex-valued vertices
    Paths                list of Path objects

Internal:

    _compat              Matplotlib / Shapely compatibility shims (not public)
"""

from ._compat import check_shapely

# Fail fast with a readable message if Shapely < 2.0 is installed.
check_shapely()

from .normal import Normal, NormalEvals
from .nonnormal import (
    NonnormalAuto,
    NonnormalMeshgrid,
    NonnormalMeshgridAuto,
    NonnormalPoints,
    NonnormalTriang,
)
from .utils import Path, Paths

__all__ = [
    # Normal
    "NormalEvals",
    "Normal",
    # Nonnormal
    "NonnormalMeshgrid",
    "NonnormalMeshgridAuto",
    "NonnormalTriang",
    "NonnormalPoints",
    "NonnormalAuto",
    # Utilities
    "Path",
    "Paths",
]
