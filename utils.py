"""Utility classes and helpers for pseudospectrum computation and plotting.

Changelog (modernization pass):
- Replaced removed ``contours.collections`` API with ``_extract_contour_paths``
  shim compatible with Matplotlib >= 3.8 *and* older installs.
- Updated Shapely imports from legacy submodule paths
  (``shapely.geometry.polygon.Polygon``) to Shapely 2.0 top-level names
  (``shapely.Polygon``, ``shapely.MultiPolygon``).
- ``import numpy`` → ``import numpy as np``.
- Dropped ``(object)`` base-class on ``Path``; ``Paths`` keeps ``list``.
- Added type hints throughout.
- ``Paths.vertices()`` rewritten to avoid repeated list copies.
- Deferred ``import shapely`` inside ``get_paths`` moved to module level.
- Added docstrings.
"""

from __future__ import annotations

import itertools
from typing import Iterable

import numpy as np
import shapely
from matplotlib import pyplot
from matplotlib.ticker import LogFormatterMathtext

from ._compat import extract_contour_paths


# ---------------------------------------------------------------------------
# Path / Paths
# ---------------------------------------------------------------------------

class Path:
    """A sequence of complex-valued vertices representing a contour path."""

    def __init__(self, vertices: Iterable[complex]) -> None:
        self.vertices: np.ndarray = np.asarray(vertices)

    def __iter__(self):
        return iter(self.vertices)

    def __len__(self) -> int:
        return len(self.vertices)

    def __repr__(self) -> str:
        return f"Path({len(self.vertices)} vertices)"

    def length(self) -> float:
        """Arc length of the path (sum of edge lengths)."""
        return float(np.sum(np.abs(np.diff(self.vertices))))


class Paths(list):
    """A list of :class:`Path` objects."""

    def length(self) -> float:
        """Total arc length across all contained paths."""
        return float(np.sum([path.length() for path in self]))

    def vertices(self) -> list[complex]:
        """Flat list of all vertices across all paths."""
        return list(itertools.chain.from_iterable(path.vertices for path in self))


# ---------------------------------------------------------------------------
# Shapely helper
# ---------------------------------------------------------------------------

def get_paths(obj: shapely.Polygon | shapely.MultiPolygon) -> Paths:
    """Convert a Shapely geometry to a :class:`Paths` collection.

    Parameters
    ----------
    obj:
        A :class:`shapely.Polygon` or :class:`shapely.MultiPolygon`.

    Returns
    -------
    Paths
        Exterior and interior (hole) rings as :class:`Path` objects.
    """
    def _ring_to_path(ring) -> Path:
        verts = np.asarray(ring.coords)
        return Path(verts[:, 0] + 1j * verts[:, 1])

    def _polygon_paths(polygon: shapely.Polygon) -> list[Path]:
        return [_ring_to_path(r) for r in [polygon.exterior, *polygon.interiors]]

    paths = Paths()
    if isinstance(obj, shapely.Polygon):
        paths += _polygon_paths(obj)
    elif isinstance(obj, shapely.MultiPolygon):
        for polygon in obj.geoms:
            paths += _polygon_paths(polygon)
    else:
        raise TypeError(
            f"Expected shapely.Polygon or shapely.MultiPolygon, got {type(obj).__name__!r}"
        )
    return paths


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_finish(
    contours,
    spectrum: np.ndarray | None = None,
    contour_labels: bool = True,
    autofit: bool = True,
) -> None:
    """Finalise a pseudospectrum contour plot.

    Parameters
    ----------
    contours:
        A Matplotlib ``ContourSet`` returned by ``contour`` or ``tricontour``.
    spectrum:
        Optional array of complex eigenvalues to overlay as markers.
    contour_labels:
        If ``True``, add log-formatted labels to contour lines.
    autofit:
        If ``True``, set axis limits to the bounding box of the contours.
    """
    if spectrum is not None:
        pyplot.plot(np.real(spectrum), np.imag(spectrum), "o")

    if autofit:
        all_paths = extract_contour_paths(contours)
        if all_paths:
            vertices = np.concatenate(
                [verts[:, 0] + 1j * verts[:, 1] for verts in all_paths]
            )
            pyplot.xlim(float(np.min(vertices.real)), float(np.max(vertices.real)))
            pyplot.ylim(float(np.min(vertices.imag)), float(np.max(vertices.imag)))

    if contour_labels:
        pyplot.clabel(contours, inline=True, fmt=LogFormatterMathtext())
