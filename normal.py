"""Pseudospectrum computation for normal matrices.

For a normal matrix the pseudospectrum is exactly the union of discs of
radius :math:`\\varepsilon` centred at the eigenvalues, so no resolvent
evaluations are needed — only geometry.

Changelog (modernization pass):
- Added missing ``import numpy as np`` (bare ``numpy.*`` calls were
  ``NameError`` at runtime).
- Replaced removed ``shapely.ops.cascaded_union`` with ``shapely.unary_union``
  (Shapely 2.0).
- Replaced legacy ``shapely.geometry.Point`` import with top-level
  ``shapely.Point`` (Shapely 2.0).
- ``super(Child, self).__init__()`` → ``super().__init__()``.
- Dropped ``(object)`` base-class.
- Renamed ``lamda`` → ``lam`` (``lamda`` was a workaround for the keyword;
  ``lam`` is the conventional NumPy/SciPy abbreviation).
- Added type hints and docstrings.
- Replaced Python-loop list ``+=`` accumulation with vectorised NumPy ops.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import shapely
from matplotlib import pyplot
from shapely import unary_union

from .utils import Path, Paths, get_paths, plot_finish


class NormalEvals:
    """Pseudospectrum of a normal matrix given its eigenvalues.

    For a normal matrix the :math:`\\varepsilon`-pseudospectrum is exactly

    .. math::

        \\Lambda_\\varepsilon(A) = \\bigcup_{\\lambda \\in \\Lambda(A)}
            D(\\lambda,\\, \\varepsilon)

    so the boundary is computed purely via Shapely geometry — no resolvent
    evaluations required.

    Parameters
    ----------
    evals:
        1-D array of complex eigenvalues.
    """

    def __init__(self, evals: np.ndarray) -> None:
        self.evals = np.asarray(evals)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def contour_paths(self, epsilon: float) -> Paths:
        """Boundary of the union of discs of radius *epsilon* around eigenvalues.

        Parameters
        ----------
        epsilon:
            Disc radius (:math:`\\varepsilon > 0`).

        Returns
        -------
        Paths
            Exterior and interior rings of the union polygon.
        """
        circles = [
            shapely.Point(lam.real, lam.imag).buffer(epsilon)
            for lam in self.evals
        ]
        pseudospec = unary_union(circles)
        return get_paths(pseudospec)

    def plot(self, epsilons: Sequence[float], **kwargs) -> object:
        """Draw pseudospectrum contours at the given epsilon levels.

        The contours are reconstructed by evaluating :meth:`contour_paths` at
        slightly padded epsilon values so that ``tricontour`` can interpolate
        smooth level curves.

        Parameters
        ----------
        epsilons:
            Sequence of :math:`\\varepsilon` levels to draw.
        **kwargs:
            Forwarded to :func:`~.utils.plot_finish`.

        Returns
        -------
        matplotlib.contour.TriContourSet
        """
        epsilons_sorted = list(np.sort(epsilons))
        # Pad by ±10 % so tricontour can bracket the outermost/innermost levels.
        pad_epsilons = (
            [epsilons_sorted[0] * 0.9]
            + epsilons_sorted
            + [epsilons_sorted[-1] * 1.1]
        )

        x_chunks: list[np.ndarray] = []
        y_chunks: list[np.ndarray] = []
        z_chunks: list[np.ndarray] = []

        for epsilon in pad_epsilons:
            for path in self.contour_paths(epsilon):
                # Drop repeated closing vertex (Shapely rings close on themselves).
                verts = path.vertices[:-1]
                if len(verts) == 0:
                    continue
                x_chunks.append(np.real(verts))
                y_chunks.append(np.imag(verts))
                z_chunks.append(np.full(len(verts), epsilon))

        if not x_chunks:
            raise RuntimeError(
                "No contour vertices found — check that epsilons are in a "
                "sensible range relative to the eigenvalue spread."
            )

        X = np.concatenate(x_chunks)
        Y = np.concatenate(y_chunks)
        Z = np.concatenate(z_chunks)

        contours = pyplot.tricontour(
            X,
            Y,
            Z,
            levels=epsilons_sorted,
            colors=pyplot.rcParams["axes.prop_cycle"].by_key()["color"],
        )
        plot_finish(contours, **kwargs)
        return contours


class Normal(NormalEvals):
    """Pseudospectrum of a normal matrix given the matrix itself.

    Eigenvalues are computed via :func:`scipy.linalg.eigvals`.

    Parameters
    ----------
    A:
        Square matrix as a ``numpy.ndarray``.
    """

    def __init__(self, A: np.ndarray) -> None:
        from scipy.linalg import eigvals

        super().__init__(eigvals(A))
