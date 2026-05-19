"""Pseudospectrum computation for nonnormal matrices.

Based on algorithms from Trefethen & Embree, "Spectra and Pseudospectra"
(Princeton University Press, 2005).

Changelog (modernization pass):
- Replaced removed ``numpy.complex`` / ``numpy.complex_`` aliases with
  ``numpy.complex128`` (NumPy >= 1.20 deprecation, removed in 1.24).
- Replaced ``contours.collections[0].get_paths()`` with the current
  Matplotlib >= 3.8 API (``QuadContourSet.get_paths()`` /
  ``TriContourSet.get_paths()``).
- Dropped ``super(Child, self).__init__()`` in favour of bare ``super()``.
- Dropped explicit ``(object)`` base-class.
- Replaced serial list-comprehension in ``_Nonnormal.__init__`` with a
  ``ProcessPoolExecutor`` parallel map for independent SVD calls.
- Added type hints throughout the public API.
- Switched ``import numpy`` to canonical ``import numpy as np``.
- Minor: f-strings, PEP-8 whitespace.
"""

from __future__ import annotations

import functools
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Literal

import numpy as np
from matplotlib import pyplot
from matplotlib.tri import Triangulation
from scipy.linalg import schur, solve_triangular, svdvals
from scipy.sparse.linalg import LinearOperator, eigsh

from ._compat import extract_contour_paths
from .utils import Path, Paths, plot_finish

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

MethodLiteral = Literal["svd", "lanczos", "lanczosinv"]


def inv_resolvent_norm(
    A: np.ndarray,
    z: complex,
    method: Literal["svd", "lanczos"] = "svd",
) -> float:
    r"""Compute the reciprocal norm of the resolvent :math:`(A - zI)^{-1}`.

    Parameters
    ----------
    A:
        Input matrix with ``A.shape == (m, n)`` where :math:`m \geq n`.
        May be a dense ``numpy.ndarray``, a sparse matrix, or a
        ``LinearOperator``.
    z:
        A complex scalar.
    method:
        * ``'svd'`` *(default)* – computes the minimal singular value of
          :math:`A - zI`.  Recommended for dense matrices.
        * ``'lanczos'`` – estimates the minimal singular value via the
          Lanczos iteration applied to
          :math:`\begin{bmatrix}0 & A \\ A^* & 0\end{bmatrix}`.
          Useful for large sparse problems.

    Returns
    -------
    float
        :math:`\sigma_{\min}(A - zI)`.
    """
    if method == "svd":
        return float(np.min(svdvals(A - z * np.eye(*A.shape))))

    if method == "lanczos":
        m, n = A.shape
        if m > n:
            raise ValueError("m > n is not allowed for the lanczos method")
        AH = A.T.conj()

        def matvec(x: np.ndarray) -> np.ndarray:
            r"""Multiply by :math:`\begin{bmatrix}0&A\\A^*&0\end{bmatrix}`."""
            x1 = x[:m]
            x2 = x[m:]
            ret1 = AH.dot(x2) - np.conj(z) * x2
            ret2 = np.array(A.dot(x1), dtype=np.complex128)
            ret2[:n] -= z * x1
            return np.concatenate([ret1, ret2])

        AH_A = LinearOperator(
            matvec=matvec,
            dtype=np.complex128,
            shape=(m + n, m + n),
        )
        evals = eigsh(
            AH_A,
            k=2,
            tol=1e-6,
            which="SM",
            maxiter=m + n + 1,
            ncv=2 * (m + n),
            return_eigenvectors=False,
        )
        return float(np.min(np.abs(evals)))

    raise ValueError(f"Unknown method {method!r}. Choose 'svd' or 'lanczos'.")


# ---------------------------------------------------------------------------
# Internal helper for parallel dispatch
# ---------------------------------------------------------------------------

def _eval_point(args: tuple) -> float:
    """Top-level callable required by ProcessPoolExecutor (must be picklable)."""
    A, point, method = args
    return inv_resolvent_norm(A, point, method=method)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class _Nonnormal:
    """Base class for nonnormal pseudospectra."""

    def __init__(
        self,
        A: np.ndarray,
        points: np.ndarray,
        method: MethodLiteral = "svd",
        n_workers: int | None = None,
    ) -> None:
        """Evaluate the inverse resolvent norm on *points*.

        Results are stored in ``self.vals`` (list of floats) and the
        evaluation points in ``self.points``.

        Parameters
        ----------
        A:
            The matrix to analyse.
        points:
            A 1-D array of complex evaluation points.
        method:
            One of ``'svd'``, ``'lanczos'``, or ``'lanczosinv'``.
        n_workers:
            Number of worker processes for the parallel SVD/Lanczos loop.
            ``None`` lets :class:`ProcessPoolExecutor` choose (typically the
            CPU count).  Set to ``1`` to disable parallelism (useful when
            ``A`` itself already uses multi-threaded BLAS).
        """
        self.points = points

        if method == "lanczosinv":
            # Algorithm from p. 375 of Trefethen/Embree 2005.
            # This branch is inherently serial because each step reuses the
            # Schur factor; parallelism would require duplicating T.
            self.vals = []
            T, _ = schur(A, output="complex")
            m, n = A.shape
            if m != n:
                raise ValueError("m != n is required in dense (lanczosinv) mode")

            for point in points:
                M = T - point * np.eye(*T.shape)

                def matvec(x: np.ndarray, _M: np.ndarray = M) -> np.ndarray:
                    r"""Multiply by :math:`(A-\lambda I)^{-*}(A-\lambda I)^{-1}`."""
                    return solve_triangular(
                        _M,
                        solve_triangular(_M, x, check_finite=False),
                        trans=2,
                        check_finite=False,
                    )

                MH_M = LinearOperator(
                    matvec=matvec,
                    dtype=np.complex128,
                    shape=(n, n),
                )
                evals = eigsh(
                    MH_M,
                    k=1,
                    tol=1e-3,
                    which="LM",
                    maxiter=n,
                    ncv=n,
                    return_eigenvectors=False,
                )
                self.vals.append(1.0 / np.sqrt(np.max(np.abs(evals))))

        else:
            # Independent evaluations — parallelise across points.
            if n_workers == 1:
                # Fast path: avoid multiprocessing overhead for small grids
                # or when the caller already manages parallelism.
                self.vals = [
                    inv_resolvent_norm(A, point, method=method)
                    for point in points
                ]
            else:
                args_iter = ((A, point, method) for point in points)
                # Preserve ordering by submitting with indices.
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    futures = {
                        executor.submit(_eval_point, args): i
                        for i, args in enumerate(
                            (A, p, method) for p in points
                        )
                    }
                    results: dict[int, float] = {}
                    for future in as_completed(futures):
                        idx = futures[future]
                        results[idx] = future.result()
                self.vals = [results[i] for i in range(len(points))]


# ---------------------------------------------------------------------------
# Concrete subclasses
# ---------------------------------------------------------------------------

class NonnormalMeshgrid(_Nonnormal):
    """Evaluate pseudospectrum on a regular Cartesian grid."""

    def __init__(
        self,
        A: np.ndarray,
        real_min: float = -1,
        real_max: float = 1,
        real_n: int = 50,
        imag_min: float = -1,
        imag_max: float = 1,
        imag_n: int = 50,
        method: MethodLiteral = "svd",
        **kwargs,
    ) -> None:
        real = np.linspace(real_min, real_max, real_n)
        imag = np.linspace(imag_min, imag_max, imag_n)
        self.Real, self.Imag = np.meshgrid(real, imag)

        super().__init__(
            A,
            self.Real.flatten() + 1j * self.Imag.flatten(),
            method=method,
            **kwargs,
        )
        self.Vals = np.array(self.vals).reshape((imag_n, real_n))

    def plot(self, epsilons, **kwargs):
        """Draw contour lines at the given epsilon levels."""
        contours = pyplot.contour(
            self.Real,
            self.Imag,
            self.Vals,
            levels=epsilons,
            colors=pyplot.rcParams["axes.prop_cycle"].by_key()["color"],
        )
        plot_finish(contours, **kwargs)
        return contours

    def contour_paths(self, epsilon: float) -> Paths:
        """Extract polygon paths for *epsilon* as a :class:`Paths` object."""
        figure = pyplot.figure()
        ax = figure.gca()
        contours = ax.contour(
            self.Real, self.Imag, self.Vals, levels=[epsilon]
        )
        paths = Paths()
        # Matplotlib >= 3.8: iterate allsegs / use get_paths() on the set.
        # We unify old and new APIs here for broad compatibility.
        all_paths = extract_contour_paths(contours)
        for vertices in all_paths:
            paths.append(Path(vertices[:, 0] + 1j * vertices[:, 1]))
        pyplot.close(figure)
        return paths


class NonnormalTriang(_Nonnormal):
    """Evaluate pseudospectrum on an arbitrary triangulation."""

    def __init__(self, A: np.ndarray, triang: Triangulation, **kwargs) -> None:
        self.triang = triang
        super().__init__(A, triang.x + 1j * triang.y, **kwargs)

    def plot(self, epsilons, **kwargs):
        """Draw contour lines at the given epsilon levels."""
        contours = pyplot.tricontour(self.triang, self.vals, levels=epsilons)
        plot_finish(contours, **kwargs)
        return contours

    def contour_paths(self, epsilon: float) -> Paths:
        """Extract polygon paths for *epsilon* as a :class:`Paths` object."""
        figure = pyplot.figure()
        contours = pyplot.tricontour(self.triang, self.vals, levels=[epsilon])
        paths = Paths()
        for vertices in extract_contour_paths(contours):
            paths.append(Path(vertices[:, 0] + 1j * vertices[:, 1]))
        pyplot.close(figure)
        return paths


class NonnormalPoints(NonnormalTriang):
    """Evaluate pseudospectrum on an unstructured cloud of points."""

    def __init__(self, A: np.ndarray, points: np.ndarray, **kwargs) -> None:
        triang = Triangulation(np.real(points), np.imag(points))
        super().__init__(A, triang, **kwargs)


class NonnormalMeshgridAuto(NonnormalMeshgrid):
    """Auto-determine bounding box via eigenvector condition number.

    Derives a bounding box for the pseudospectrum of a diagonalisable matrix
    from the condition number of its eigenvector basis (Theorem 2.3 in
    Trefethen & Embree).  The box is guaranteed to contain the
    :math:`\\varepsilon_{\\max}`-pseudospectrum but may be an overestimate.

    Parameters
    ----------
    A:
        Square matrix as a ``numpy.ndarray`` with ``A.shape == (N, N)``.
    eps_max:
        Maximum :math:`\\varepsilon` of interest.
    """

    def __init__(self, A: np.ndarray, eps_max: float, **kwargs) -> None:
        from scipy.linalg import eig

        evals, evecs = eig(A)
        kappa = np.linalg.cond(evecs, 2)

        bbox = {
            "real_min": float(np.min(evals.real)) - eps_max * kappa,
            "real_max": float(np.max(evals.real)) + eps_max * kappa,
            "imag_min": float(np.min(evals.imag)) - eps_max * kappa,
            "imag_max": float(np.max(evals.imag)) + eps_max * kappa,
        }
        bbox.update(kwargs)
        super().__init__(A, **bbox)


class NonnormalAuto(NonnormalPoints):
    """Automatically determine an inclusion set for the pseudospectrum.

    Useful when you have no a-priori idea where the pseudospectrum lives.

    Computation is dominated by :math:`N(N+1)/2` Schur decompositions and
    :math:`N \\cdot n_{\\text{circles}} \\cdot n_{\\text{points}}` resolvent
    evaluations.

    Parameters
    ----------
    A:
        Square matrix.
    eps_min:
        Smallest :math:`\\varepsilon` to resolve (must be ``> 0``).
    eps_max:
        Largest :math:`\\varepsilon` of interest.
    n_circles:
        Number of logarithmically-spaced radii per eigenvalue.
    n_points:
        Number of evaluation points per circle.
    randomize:
        Rotate each circle by a random angle to reduce aliasing artefacts.
    """

    def __init__(
        self,
        A: np.ndarray,
        eps_min: float,
        eps_max: float,
        n_circles: int = 20,
        n_points: int = 20,
        randomize: bool = True,
        **kwargs,
    ) -> None:
        from scipy.linalg import eig, schur as _schur

        if eps_min <= 0:
            raise ValueError("eps_min must be > 0")
        if eps_min >= eps_max:
            raise ValueError("eps_min must be < eps_max")

        M = A.copy()
        midpoints: list[complex] = []
        radii: list[float] = [eps_max]

        for _ in range(A.shape[0]):
            evals, evecs = eig(M)
            evec_cond = np.linalg.cond(evecs, 2)

            if len(evals) == 1:
                midpoints.append(complex(evals[0]))
                radii.append(radii[-1])
                continue

            candidates_midpoints: list[complex] = []
            candidates_radii: list[float] = []
            candidates_Ms: list[np.ndarray] = []

            for lam in evals:
                dists = np.sort(np.abs(lam - evals))

                def _sort_key(mu: complex, _lam: complex = lam, _d: float = dists[1]) -> bool:
                    return bool(np.abs(mu - _lam) <= _d)

                T, _Z, _sdim = _schur(M, output="complex", sort=_sort_key)

                c = T[0, 1:]
                M_tmp = T[1:, 1:]
                candidates_midpoints.append(complex(T[0, 0]))

                shift = T[0, 0] * np.eye(*M_tmp.shape)
                r = solve_triangular(M_tmp - shift, c, trans="T")

                sep_min = float(np.min(svdvals(M_tmp - shift)))
                r_norm = float(np.linalg.norm(r, 2))
                p = np.sqrt(1.0 + r_norm ** 2)
                kappa = p + r_norm  # Demmel 1

                # --- Grammont-Largillier bound ---
                g_gram_larg = np.sqrt(1.0 + np.linalg.norm(c, 2) / radii[-1])

                # --- Demmel bound 1 ---
                g_demmel1 = kappa

                # --- Demmel bound 2 ---
                g_demmel2 = np.inf
                if radii[-1] <= sep_min / (2 * kappa):
                    denom = 0.5 * sep_min - p * radii[-1]
                    if denom > 0:
                        g_demmel2 = p + r_norm ** 2 * radii[-1] / denom

                # --- Karow bound (personal communication) ---
                g_mika = np.inf
                if radii[-1] <= sep_min / (2 * kappa):
                    eps_sep = radii[-1] / sep_min
                    discriminant = 0.25 - eps_sep * (p - eps_sep)
                    if discriminant >= 0:
                        g_mika = (p - eps_sep) / (
                            0.5 + np.sqrt(discriminant)
                        )

                candidates_radii.append(
                    radii[-1]
                    * float(np.min([evec_cond, g_gram_larg, g_demmel1, g_demmel2, g_mika]))
                )
                candidates_Ms.append(M_tmp)

            best = int(np.argmin(candidates_radii))
            midpoints.append(candidates_midpoints[best])
            radii.append(candidates_radii[best])
            M = candidates_Ms[best]

        # Drop the sentinel first radius.
        radii = radii[1:]

        # Build evaluation points: concentric circles around each midpoint.
        arg = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        point_chunks: list[np.ndarray] = []
        for midpoint, radius_max in zip(midpoints, radii):
            radius_log = np.logspace(
                np.log10(eps_min), np.log10(radius_max), n_circles
            )
            for radius in radius_log:
                phase = 2 * np.pi * np.random.rand() if randomize else 0.0
                if np.abs(radius) / (np.abs(midpoint) + 1e-300) > 1e-15:
                    point_chunks.append(midpoint + radius * np.exp(1j * (phase + arg)))

        points = np.concatenate(point_chunks)
        super().__init__(A, points, **kwargs)
