"""Internal compatibility shims for third-party API changes.

This module centralises workarounds for breaking changes in dependencies so
that the rest of the codebase stays clean.  Nothing here is part of the
public API — import from the top-level package instead.

Current shims
-------------
- :func:`extract_contour_paths` — Matplotlib < 3.8 vs >= 3.8 contour API.
- :func:`shapely_version` — convenience accessor for runtime Shapely version.
"""

from __future__ import annotations

import importlib.metadata

import numpy as np


# ---------------------------------------------------------------------------
# Matplotlib
# ---------------------------------------------------------------------------

def extract_contour_paths(contours) -> list[np.ndarray]:
    """Return vertex arrays from a Matplotlib ``ContourSet``.

    Matplotlib 3.8 removed ``ContourSet.collections`` and replaced it with
    a direct ``ContourSet.get_paths()`` method.  This shim handles both.

    Parameters
    ----------
    contours:
        A ``ContourSet`` returned by ``pyplot.contour``,
        ``pyplot.tricontour``, or their Axes equivalents.

    Returns
    -------
    list of numpy.ndarray
        Each array has shape ``(N, 2)`` with columns ``[x, y]``.
        Returns an empty list if no paths are present.
    """
    # Current API: Matplotlib >= 3.8
    if hasattr(contours, "get_paths"):
        return [p.vertices for p in contours.get_paths()]

    # Legacy API: Matplotlib < 3.8
    if hasattr(contours, "collections") and contours.collections:
        return [p.vertices for p in contours.collections[0].get_paths()]

    return []


# ---------------------------------------------------------------------------
# Shapely
# ---------------------------------------------------------------------------

def shapely_version() -> tuple[int, ...]:
    """Return the installed Shapely version as a tuple of ints, e.g. ``(2, 0, 1)``."""
    raw = importlib.metadata.version("shapely")
    return tuple(int(x) for x in raw.split(".")[:3])


def check_shapely() -> None:
    """Raise ``ImportError`` with a clear message if Shapely < 2.0 is installed.

    Shapely 2.0 removed ``shapely.geometry.polygon.Polygon``,
    ``shapely.geometry.multipolygon.MultiPolygon``, and
    ``shapely.ops.cascaded_union``.  This library requires 2.0+.
    """
    ver = shapely_version()
    if ver < (2, 0):
        raise ImportError(
            f"Shapely >= 2.0 is required, but {'.'.join(str(v) for v in ver)} "
            "is installed.  Upgrade with:  pip install 'shapely>=2.0'"
        )
