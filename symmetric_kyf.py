"""
symmetric_kyf.py
================

Symmetric overcomplete third-order tensor decomposition via Koszul-Young
flattenings: a Python implementation of Algorithm 1 of

    Yu Luo and Alexander S. Wein, "Symmetric Extension of Overcomplete Tensor
    Decomposition via Koszul-Young Flattenings" (draft).

Step 3 (finding the rank-1 matrices in a subspace) uses the implementation of
the Johnston-Lovitz-Vijayaraghavan (JLV) algorithm in

    https://github.com/lemniscate8/linear-intersect-varieties

through its entry point ``linear_conic_intersect.demix_subspace``.

Problem
-------
Given a real symmetric tensor

        T = sum_{l=1}^r  a^(l) (x) a^(l) (x) a^(l)      in R^{n x n x n},

with generic components a^(l) in R^n (possibly r > n), recover the a^(l).

Algorithm 1 (numbering as in the draft)
---------------------------------------
Fix p >= 1, q = 2p + 1 <= n - 2, m = n - q, and split every component as
a = (a_hat, a_tilde), a_hat = (a_1..a_q) in R^q, a_tilde = (a_{q+1}..a_n) in R^m.

 1. Build the flattening M = M(T; p) of eq. (1):
        M_{Sj,Uk} = sum_i 1[U = S u {i}] sigma(U, i) T_{i, q+j, q+k},
    rows (S, j) with |S| = p, columns (U, k) with |U| = p + 1.  By Lemma 2.1,
    M = sum_l A(a_hat^(l); p) (x) a_tilde^(l) a_tilde^(l)^T, and generically
    rank(M) = r * binom(q-1, p) (Theorem 3.1), which also lets us learn r.
 2. Compute B = (phi_{p,q} (x) I)( colspan(M) cap (Z_{p,q} (x) R^m) ), eq. (7);
    generically B = span{ d^(l) (x) a_tilde^(l) } with d^(l) = (a_1..a_{p+1}).
 3. Find the rank-1 matrices delta^(l) (x) alpha^(l) in B with [JLV23, Cor. 3];
    output "fail" if the number found is not r.
 4. Split them with an SVD and solve the linear system
        T_1 = T[q:, q:, q:] = sum_l x_l (alpha^(l))^{(x)3},  x_l = beta_l^{-3},
    for x, then a_tilde^(l) = cbrt(x_l) * alpha^(l) (real cube root), eq. (9).
 5. Solve the linear system
        T_2 = T[:q, q:, q:] = sum_l a_hat^(l) (x) a_tilde^(l) (x) a_tilde^(l)
    for the heads a_hat^(l), and output a^(l) = (a_hat^(l), a_tilde^(l)).

The code also reports, as diagnostics, every quantity the draft uses to argue
that a step succeeds (rank of M, dimension of the intersection in Step 2,
dimension of the JLV kernel, residuals of the linear systems, and the final
reconstruction error), and ``check_uniqueness_conditions`` evaluates
hypotheses (i)-(vii) of Theorem 4.3 for given components.

Conventions
-----------
* All indices are 0-based. Paper coordinate i in [n] is array index i - 1, and
  a_tilde_j = a_{q+j} (paper) is a[q + j] here (j = 0..m-1).
* Subsets S, U are sorted tuples of 0-based indices in lexicographic order, so
  M = M(T; p) has rows (S, j) at position s*m + j and columns (U, k) at
  position u*m + k, exactly the Kronecker ordering of A (x) (a_tilde a_tilde^T).
* With 0-based i, sigma(V, i) = (-1)^i for V = {0..p}, so phi_{p,q} maps
  e_{V minus {i}} to (-1)^i e_i.
* (p+1) x m matrices are vectorized in row-major order (index i*m + j). This
  is the convention of the JLV code for dims=[p+1, m], mults=[1, 1].
* Components are returned as the columns of an n x r array.
"""

from __future__ import annotations

import functools
import itertools
import math
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
from scipy import linalg as sla
from scipy.optimize import linear_sum_assignment

__all__ = [
    "set_jlv_path",
    "sigma",
    "koszul_matrix",
    "koszul_young_flattening",
    "detect_rank",
    "subspace_B",
    "find_rank_one_matrices",
    "solve_tail",
    "solve_head",
    "decompose",
    "SymKYFResult",
    "symmetric_cp_tensor",
    "random_instance",
    "match_components",
    "check_uniqueness_conditions",
    "parameter_bounds",
]

# ---------------------------------------------------------------------------
# 0. Access to the JLV implementation (lemniscate8/linear-intersect-varieties)
# ---------------------------------------------------------------------------

JLV_REPO_URL = "https://github.com/lemniscate8/linear-intersect-varieties"
_JLV_MODULES = None
_JLV_PATH = None


def set_jlv_path(path):
    """Point this module at a local clone of linear-intersect-varieties."""
    global _JLV_PATH, _JLV_MODULES
    _JLV_PATH = os.path.abspath(os.path.expanduser(str(path)))
    _JLV_MODULES = None


def _jlv():
    """Import (once) the modules of the JLV repository used in Step 3.

    The repository is looked up, in order, at: the path given to
    ``set_jlv_path``, the environment variable ``JLV_REPO_PATH``, a folder
    ``linear-intersect-varieties`` next to this file, and the same folder in
    the current working directory. If the modules are already importable,
    nothing needs to be configured.
    """
    global _JLV_MODULES
    if _JLV_MODULES is not None:
        return _JLV_MODULES
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        _JLV_PATH,
        os.environ.get("JLV_REPO_PATH"),
        os.path.join(here, "linear-intersect-varieties"),
        os.path.join(os.getcwd(), "linear-intersect-varieties"),
    ]
    for cand in candidates:
        if cand and os.path.isfile(os.path.join(cand, "linear_conic_intersect.py")):
            if cand not in sys.path:
                sys.path.append(cand)
            break
    try:
        import linear_conic_intersect as lci
        import segre_veronese_ideal_generation as ig
        import subspace_manipulation as suma
    except ImportError as exc:  # pragma: no cover - depends on the user's setup
        raise ImportError(
            "Cannot import the JLV implementation. Clone "
            f"{JLV_REPO_URL} and put it next to this file, set the environment "
            "variable JLV_REPO_PATH, or call set_jlv_path(<path to the clone>)."
        ) from exc
    _JLV_MODULES = (lci, suma, ig)
    return _JLV_MODULES


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _as_cubic_tensor(T):
    T = np.asarray(T, dtype=float)
    if T.ndim != 3 or not (T.shape[0] == T.shape[1] == T.shape[2]):
        raise ValueError(f"T must be an n x n x n array, got shape {T.shape}.")
    return T


def _check_params(n, p):
    if int(p) != p or p < 1:
        raise ValueError(f"p must be an integer >= 1, got {p}.")
    q = 2 * p + 1
    if q > n - 2:
        raise ValueError(f"Need q = 2p + 1 <= n - 2 (so that the rank-1 matrices in "
                         f"Step 3 have at least 2 columns), got q = {q}, n = {n}.")
    return q, n - q


def _svd(X, full_matrices=False, compute_uv=True):
    """SVD with the fast LAPACK driver and a fallback to the robust one."""
    try:
        return sla.svd(X, full_matrices=full_matrices, compute_uv=compute_uv,
                       lapack_driver="gesdd", check_finite=False)
    except np.linalg.LinAlgError:
        return sla.svd(X, full_matrices=full_matrices, compute_uv=compute_uv,
                       lapack_driver="gesvd", check_finite=False)


def _relative_singular_values(X):
    s = _svd(X, compute_uv=False)
    return s / s[0] if s.size and s[0] > 0 else s


def _sines_between(X, Y):
    """Column-wise sine of the angle between X[:, l] and Y[:, l].

    Computed from the orthogonal residual (accurate near 0, unlike
    sqrt(1 - cos^2))."""
    Xn = X / np.linalg.norm(X, axis=0)
    Yn = Y / np.linalg.norm(Y, axis=0)
    return np.linalg.norm(Yn - Xn * np.sum(Xn * Yn, axis=0), axis=0)


def sigma(U, i):
    """sigma(U, i) = (-1)^{|{j in U : j < i}|}, eq. (2) (0-based indices)."""
    return -1 if sum(1 for j in U if j < i) % 2 else 1


@functools.lru_cache(maxsize=None)
def _koszul_pattern(q, p):
    """Nonzero pattern of A(.; p): pairs (S, U = S u {i}) with sign sigma(U, i).

    Returns (rows, cols, s_idx, u_idx, i_idx, signs) where rows / cols are the
    p- / (p+1)-subsets of range(q) and, for every admissible pair, s_idx and
    u_idx are the row and column positions, i_idx the added index i, and signs
    the value sigma(U, i). Treat the returned arrays as read-only (cached).
    """
    rows = tuple(itertools.combinations(range(q), p))
    cols = tuple(itertools.combinations(range(q), p + 1))
    col_pos = {U: k for k, U in enumerate(cols)}
    s_idx, u_idx, i_idx, signs = [], [], [], []
    for s, S in enumerate(rows):
        for i in range(q):
            if i in S:
                continue
            U = tuple(sorted(S + (i,)))
            s_idx.append(s)
            u_idx.append(col_pos[U])
            i_idx.append(i)
            signs.append(sigma(U, i))
    to_int = lambda x: np.asarray(x, dtype=np.intp)  # noqa: E731
    return (rows, cols, to_int(s_idx), to_int(u_idx), to_int(i_idx),
            np.asarray(signs, dtype=float))


def koszul_matrix(a_hat, p):
    """The binom(q,p) x binom(q,p+1) matrix A(a_hat; p) of eq. (3), q = len(a_hat).

    A_{SU} = sum_i 1[U = S u {i}] sigma(U, i) a_i. Column U is the vector
    z^(U)(a_hat; p) used in Section 4.
    """
    a_hat = np.asarray(a_hat, dtype=float).ravel()
    rows, cols, s_idx, u_idx, i_idx, signs = _koszul_pattern(a_hat.size, p)
    A = np.zeros((len(rows), len(cols)))
    A[s_idx, u_idx] = signs * a_hat[i_idx]
    return A


# ---------------------------------------------------------------------------
# Step 1: the flattening M(T; p) and rank detection (Theorem 3.1)
# ---------------------------------------------------------------------------


def koszul_young_flattening(T, p):
    """Step 1: the binom(q,p)(n-q) x binom(q,p+1)(n-q) matrix M(T; p) of eq. (1).

    Block (S, U) with U = S u {i} equals sigma(U, i) * T[i, q:, q:]; all other
    blocks are zero. Since q = 2p + 1, M is square.
    """
    T = _as_cubic_tensor(T)
    n = T.shape[0]
    q, m = _check_params(n, p)
    rows, cols, s_idx, u_idx, i_idx, signs = _koszul_pattern(q, p)
    M = np.zeros((len(rows) * m, len(cols) * m))
    blocks = M.reshape(len(rows), m, len(cols), m)  # a view of M
    blocks[s_idx, :, u_idx, :] = signs[:, None, None] * T[i_idx][:, q:, q:]
    return M


def detect_rank(T, p, tol=1e-8, M=None):
    """Learn r from rank(M) = r * binom(q-1, p) (Theorem 3.1).

    Returns (r_estimate, numerical_rank_of_M, relative_singular_values).
    ``tol`` is the cutoff on singular values relative to the largest one.
    """
    T = _as_cubic_tensor(T)
    q, _ = _check_params(T.shape[0], p)
    if M is None:
        M = koszul_young_flattening(T, p)
    s_rel = _relative_singular_values(M)
    rank = int(np.sum(s_rel > tol))
    return rank // math.comb(q - 1, p), rank, s_rel


# ---------------------------------------------------------------------------
# Step 2: B = (phi (x) I)(colspan(M) cap (Z_{p,q} (x) R^m))
# ---------------------------------------------------------------------------


def _Z_blocks(p):
    """Row positions of S_i = {0..p} minus {i} (i = 0..p) among the p-subsets."""
    q = 2 * p + 1
    rows = _koszul_pattern(q, p)[0]
    row_pos = {S: s for s, S in enumerate(rows)}
    V = tuple(range(p + 1))
    return rows, [row_pos[tuple(x for x in V if x != i)] for i in range(p + 1)]


def subspace_B(Q, p, m, r):
    """Step 2 of Algorithm 1, eq. (7).

    Parameters
    ----------
    Q : (binom(q,p) m) x R array with orthonormal columns spanning colspan(M).
    p, m : flattening parameter and m = n - q.
    r : dimension of B to extract (the number of components).

    Returns
    -------
    B_basis : ((p+1) m) x r array, an orthonormal basis of B; column t is a
        row-major vectorized (p+1) x m matrix.
    sines : length-R array (ascending) of sines of the principal angles
        between colspan(M) and Z_{p,q} (x) R^m. The number of (numerically)
        zero sines is dim(colspan(M) cap (Z_{p,q} (x) R^m)); Lemma 4.4 says it
        is r.
    """
    rows, S_of_i = _Z_blocks(p)
    R = Q.shape[1]
    in_Z = np.zeros(len(rows), dtype=bool)
    in_Z[S_of_i] = True
    row_in_Z = np.repeat(in_Z, m)          # rows (S, j) with S inside {0..p}
    Q_in, Q_out = Q[row_in_Z], Q[~row_in_Z]

    # x = Q y lies in Z (x) R^m  <=>  Q_out y = 0. Such y lie in the row space
    # of the small (p+1)m x R block Q_in (||Q_in y|| = ||y||), so we first take
    # an orthonormal basis Y_in of that row space, then find the directions in
    # it that Q_out annihilates. This equals the SVD of Q_out (same subspace,
    # same sines) at a fraction of the cost, and the sines are computed
    # directly as ||Q_out y|| (no 1 - cos^2 cancellation).
    _, _, yh = _svd(Q_in)
    Y_in = yh.T                            # R x k, k = min((p+1) m, R)
    _, s_k, zh = _svd(Q_out @ Y_in, full_matrices=Q_out.shape[0] < Y_in.shape[1])
    sines_k = np.zeros(Y_in.shape[1])
    sines_k[: s_k.size] = s_k              # descending, padded with exact zeros
    Y = Y_in @ zh[::-1][:r].T              # the r directions with smallest sines
    # remaining directions of colspan(M) are orthogonal to Z (x) R^m (sine 1)
    sines = np.concatenate([sines_k[::-1], np.ones(R - Y_in.shape[1])])

    X = (Q @ Y).reshape(len(rows), m, r)   # vectors in colspan(M), ~supported on Z
    # phi_{p,q}: block S_i = {0..p} minus {i}  ->  (-1)^i e_i   (0-based i)
    B = np.stack([(-1) ** i * X[S_of_i[i]] for i in range(p + 1)])  # (p+1, m, r)
    B_basis, _ = np.linalg.qr(B.reshape((p + 1) * m, r))
    return B_basis, sines


# ---------------------------------------------------------------------------
# Step 3: rank-1 matrices in B via the JLV implementation
# ---------------------------------------------------------------------------


def _realify(Y):
    """Remove the arbitrary complex phase of each column (eigenvectors from
    Jennrich's algorithm may be returned as complex arrays)."""
    Y = np.asarray(Y)
    if not np.iscomplexobj(Y):
        return Y.astype(float), 0.0
    k = np.argmax(np.abs(Y), axis=0)
    ph = Y[k, np.arange(Y.shape[1])]
    ph = ph / np.abs(ph)
    Z = Y * np.conj(ph)[None, :]
    imag = np.linalg.norm(Z.imag, axis=0) / np.linalg.norm(Z, axis=0)
    return Z.real.copy(), float(imag.max(initial=0.0))


def _split_rank_one(Y, shape):
    """SVD split of each (vectorized) rank-1 matrix: Y_t ~ delta_t alpha_t^T."""
    n1, n2 = shape
    r = Y.shape[1]
    deltas, alphas, ratios = np.empty((n1, r)), np.empty((n2, r)), np.zeros(r)
    for t in range(r):
        u, s, vt = np.linalg.svd(Y[:, t].reshape(n1, n2))
        deltas[:, t] = s[0] * u[:, 0]
        alphas[:, t] = vt[0]
        if s.size > 1 and s[0] > 0:
            ratios[t] = s[1] / s[0]
    return deltas, alphas, ratios


def find_rank_one_matrices(B_basis, shape, r, *, kernel="svd", tol=1e-8,
                           check_tol=1e-6, tries=5, seed=None):
    """Step 3 of Algorithm 1: the rank-1 matrices in span(B_basis), via JLV.

    This calls ``linear_conic_intersect.demix_subspace`` from the JLV repo
    (Segre variety, dims=[p+1, m], mults=[1, 1]). That routine forms the
    symmetric lift of the basis, evaluates the 2x2-minor quadrics
    (``segre_veronese_ideal_generation.sv_poly_matrix``), computes the kernel
    (the space spanned by the (y_l (x) y_l) in Lemma 4.5) and runs Jennrich's
    simultaneous diagonalization on it. ``kernel`` selects how the kernel is
    computed:

    "repo"  The repo's own ``extract_kernel_naive``, unchanged: it calls
            ``scipy.linalg.null_space``, whose default cutoff is machine
            precision, sigma_max * eps * max(matrix shape). The number of
            kernel vectors is whatever falls below that cutoff. In our tests
            this gives exactly r vectors for noiseless T (the r-th singular
            value is ~1e-15 to 1e-13), but the margin can be small (within a
            factor ~2 of the cutoff in some runs). With relative noise ~1e-13
            in T it keeps only part of the kernel, and from ~1e-12 on it
            finds no kernel at all, so Step 3 fails.
    "svd"   (default) An SVD of the same matrix that keeps the r smallest
            right singular vectors (r is known at this point). The number of
            singular values below ``tol`` (relative) is recorded as the
            number of rank-1 matrices found, for the "fail if not r" test.
            This keeps working when T is noisy.

    Returns (deltas, alphas, Y, info) where Y[:, t] is the t-th rank-1 matrix
    (row-major vectorized), deltas[:, t] (x) alphas[:, t] = Y[:, t] reshaped,
    and info holds diagnostics; info["jlv_kernel_dim"] is the number of
    kernel vectors detected (by the repo's cutoff, or by ``tol``). If the
    kernel is empty (possible with kernel="repo"), deltas, alphas and Y are
    None.
    """
    if kernel not in ("svd", "repo"):
        raise ValueError(f"kernel must be 'svd' or 'repo', got {kernel!r}.")
    lci, suma, _ = _jlv()
    n1, n2 = int(shape[0]), int(shape[1])
    if B_basis.shape[0] != n1 * n2:
        raise ValueError(f"B_basis must have {n1 * n2} rows, got {B_basis.shape[0]}.")
    # weights turning the symmetric lift's coordinates into entries of c c^T
    weights = suma.symmetric_weights(B_basis.shape[1], 2)
    info = {"jlv_kernel_mode": kernel}
    cache = {}

    def kernel_method(sparse_projector, dense_basis, expected_solutions):
        # The kernel does not depend on the random seed, so compute it once
        # and hand demix_subspace a fresh copy (it rescales it in place).
        if "K" not in cache:
            A = np.asarray(sparse_projector @ dense_basis, dtype=float)
            ncol = A.shape[1]
            s_all, vh = np.zeros(ncol), np.eye(ncol)
            if A.shape[0] and np.any(A):
                if kernel == "svd":
                    _, s, vh = _svd(A, full_matrices=A.shape[0] < ncol)
                else:  # singular values for the diagnostics only
                    s = _svd(A, compute_uv=False)
                s_all[: s.size] = s
            # The basis is orthonormal, so the minors are O(1) unless they
            # vanish; the floor 1 keeps r = 1 (a single, rank-1 vector) right.
            scale = max(s_all[0], 1.0)
            info["jlv_matrix_shape"] = A.shape
            info["jlv_singular_values"] = s_all[::-1] / scale     # ascending
            if kernel == "repo":
                K, _ = lci.extract_kernel_naive(sparse_projector, dense_basis,
                                                expected_solutions)
                info["jlv_cutoff"] = s_all[0] * np.finfo(float).eps * max(A.shape) / scale
                info["jlv_kernel_dim"] = int(K.shape[1])
            else:
                kdim = int(np.sum(s_all / scale <= tol))
                k = kdim if expected_solutions is None else int(expected_solutions)
                K = vh[ncol - k:].T
                info["jlv_cutoff"] = tol
                info["jlv_kernel_dim"] = kdim
            K = np.array(K, dtype=float)
            info["kernel_tensor_norm"] = float(np.linalg.norm(K / weights[:, None]))
            cache["K"] = K
        return cache["K"].copy(), True

    best = None
    for t in range(max(1, tries)):
        rng = np.random.default_rng(None if seed is None else seed + t)
        coef, (code, jdiag) = lci.demix_subspace(
            B_basis, [n1, n2], [1, 1], kernel_method=kernel_method,
            expected_solutions=r, rng=rng)
        if code != 0 or jdiag is None:
            break  # empty kernel: no rank-1 matrices found, retrying cannot help
        Y, imag = _realify(B_basis @ coef)
        Y /= np.linalg.norm(Y, axis=0, keepdims=True)
        deltas, alphas, ratios = _split_rank_one(Y, (n1, n2))
        G = np.abs(Y.T @ Y)
        np.fill_diagonal(G, 0.0)
        jen_err = float(jdiag[0]) / max(info.get("kernel_tensor_norm", 1.0), 1e-300)
        quality = max(float(ratios.max(initial=0.0)), jen_err)
        cand = dict(deltas=deltas, alphas=alphas, Y=Y, quality=quality,
                    jennrich_rel_error=jen_err, max_rank1_ratio=float(ratios.max(initial=0.0)),
                    max_pairwise_overlap=float(G.max(initial=0.0)),
                    max_imag_part=imag, attempt=t + 1)
        if best is None or quality < best["quality"]:
            best = cand
        if quality <= check_tol:
            break
    if best is None:
        return None, None, None, info
    info.update({k: v for k, v in best.items() if k not in ("deltas", "alphas", "Y")})
    return best["deltas"], best["alphas"], best["Y"], info


# ---------------------------------------------------------------------------
# Steps 4 and 5: the two linear systems
# ---------------------------------------------------------------------------


def solve_tail(T, q, alphas):
    """Step 4: solve T_1 = T[q:, q:, q:] = sum_l x_l alpha_l^{(x)3}, eq. (9).

    Returns (a_tilde, x, relative_residual) with a_tilde[:, l] =
    cbrt(x_l) * alphas[:, l], the real cube root (x_l = beta_l^{-3}).
    """
    T1 = T[q:, q:, q:]
    K1 = sla.khatri_rao(sla.khatri_rao(alphas, alphas), alphas)  # m^3 x r
    x, *_ = np.linalg.lstsq(K1, T1.ravel(), rcond=None)
    rel_res = np.linalg.norm(K1 @ x - T1.ravel()) / max(np.linalg.norm(T1), 1e-300)
    return np.cbrt(x)[None, :] * alphas, x, float(rel_res)


def solve_head(T, q, a_tilde):
    """Step 5: solve T_2 = T[:q, q:, q:] = sum_l a_hat_l (x) a_tilde_l (x) a_tilde_l.

    For each i < q this is the linear system
    T[i, q:, q:] = sum_l (a_hat_l)_i a_tilde_l a_tilde_l^T, whose coefficient
    matrix has columns a_tilde_l (x) a_tilde_l (Lemma 4.1). Returns
    (a_hat, relative_residual), a_hat of shape q x r.
    """
    m = T.shape[0] - q
    K2 = sla.khatri_rao(a_tilde, a_tilde)               # m^2 x r
    T2 = T[:q, q:, q:].reshape(q, m * m).T              # m^2 x q
    H, *_ = np.linalg.lstsq(K2, T2, rcond=None)         # r x q
    rel_res = np.linalg.norm(K2 @ H - T2) / max(np.linalg.norm(T2), 1e-300)
    return H.T, float(rel_res)


# ---------------------------------------------------------------------------
# The full algorithm
# ---------------------------------------------------------------------------


@dataclass
class SymKYFResult:
    """Output of :func:`decompose`.

    components : n x r array; column l is a recovered component a^(l).
    r : number of components used.
    success : False if any of the algorithm's "fail" conditions (or a numerical
        sanity check) was triggered; see ``failures``.
    failures : human-readable list of the checks that failed.
    diagnostics : per-step numbers (ranks, gaps, residuals, timings, ...).
    deltas, alphas : the factors delta^(l), alpha^(l) from Step 3/4.
    B_basis : orthonormal basis of B from Step 2.
    """

    components: np.ndarray
    r: int
    success: bool
    failures: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    deltas: np.ndarray | None = None
    alphas: np.ndarray | None = None
    B_basis: np.ndarray | None = None

    def summary(self):
        d = self.diagnostics
        lines = [
            f"Symmetric KYF decomposition: n = {d.get('n')}, p = {d.get('p')}, "
            f"q = {d.get('q')}, r = {self.r}  ->  "
            f"{'SUCCESS' if self.success else 'FAIL'}",
            f"  Step 1  M is {d.get('M_shape')}, rank(M) = {d.get('rank_M')} "
            f"(expected r*C(q-1,p) = {d.get('expected_rank_M')}); "
            f"sigma_R/sigma_1 = {d.get('M_last_kept', np.nan):.2e}, "
            f"sigma_(R+1)/sigma_1 = {d.get('M_first_dropped', np.nan):.2e}",
            f"  Step 2  dim(colspan M cap Z(x)R^m) = {d.get('intersection_dim')} "
            f"(want r); sines: r-th = {d.get('sine_r', np.nan):.2e}, "
            f"(r+1)-th = {d.get('sine_r_plus_1', np.nan):.2e}",
            f"  Step 3  JLV kernel dim = {d.get('jlv_kernel_dim')} (want r) "
            f"[kernel={d.get('jlv_kernel_mode')}, cutoff {d.get('jlv_cutoff', np.nan):.1e}]; "
            f"max sigma2/sigma1 of recovered = {d.get('max_rank1_ratio', np.nan):.2e}; "
            f"Jennrich rel. error = {d.get('jennrich_rel_error', np.nan):.2e}",
            f"  Step 4  rel. residual T_1 = {d.get('T1_rel_residual', np.nan):.2e}, "
            f"min |x_l| = {d.get('min_abs_x', np.nan):.2e}",
            f"  Step 5  rel. residual T_2 = {d.get('T2_rel_residual', np.nan):.2e}",
            f"  Check   ||sum a(x)3 - T|| / ||T|| = {d.get('reconstruction_rel_error', np.nan):.2e}; "
            f"delta vs (a_1..a_p+1) max sine = {d.get('delta_consistency', np.nan):.2e}",
            "  Time    " + ", ".join(f"{k} {v:.2f}s" for k, v in d.get("timings", {}).items()),
        ]
        if self.failures:
            lines.append("  Failures:")
            lines += [f"    - {f}" for f in self.failures]
        return "\n".join(lines)

    def __str__(self):
        return self.summary()


def decompose(T, p, r=None, *, tol=1e-8, check_tol=1e-6, jlv_kernel="svd",
              jlv_tries=5, seed=None, verbose=False):
    """Algorithm 1: decompose a symmetric n x n x n tensor T = sum_l a_l^{(x)3}.

    Parameters
    ----------
    T : array (n, n, n), real symmetric.
    p : integer >= 1; q = 2p + 1 must be <= n - 2. Larger p allows larger r
        (Theorem 3.1) at the price of a binom(q,p)(n-q) square matrix M.
    r : number of components. If None, it is learned from rank(M)
        (Theorem 3.1). For noisy T, pass r explicitly.
    tol : relative cutoff deciding which singular values count as zero
        (numerical rank of M, JLV kernel); sines of principal angles below
        sqrt(tol) count as zero in Step 2. Only the diagnostics/fail flags
        depend on it when r is given. For noisy T use roughly tol ~ noise.
    check_tol : tolerance for the residual-based sanity checks.
    jlv_kernel : "svd" (default) or "repo"; how Step 3 computes the JLV kernel.
        "repo" uses the repository's own routine unchanged (scipy null_space
        with its machine-precision cutoff); it works for noiseless T but fails
        once T carries relative noise above ~1e-13. See
        :func:`find_rank_one_matrices`.
    jlv_tries : number of random restarts of Jennrich's step inside JLV.
    seed : seed for the randomness in Jennrich's step (reproducibility).
    verbose : print the summary at the end.

    Returns
    -------
    SymKYFResult. ``result.components`` is n x r (columns a^(l), in an
    arbitrary order). ``result.success`` is False when one of the algorithm's
    fail conditions is triggered (the computation is still carried through so
    the diagnostics can be inspected).
    """
    T = _as_cubic_tensor(T)
    n = T.shape[0]
    q, m = _check_params(n, p)
    C = math.comb(q - 1, p)
    failures, diag, timings = [], {"n": n, "p": p, "q": q, "m": m}, {}
    diag["timings"] = timings

    # ---- Step 1: flattening, colspan, rank --------------------------------
    t0 = time.perf_counter()
    M = koszul_young_flattening(T, p)
    U, s, _ = _svd(M)
    s_rel = s / s[0] if s[0] > 0 else s
    rank = int(np.sum(s_rel > tol))
    diag.update(M_shape=M.shape, rank_M=rank, M_singular_values=s_rel)
    if r is None:
        r = rank // C
        if rank % C:
            failures.append(f"Step 1: rank(M) = {rank} is not a multiple of "
                            f"binom(q-1,p) = {C}; using r = {r}.")
    r = int(r)
    R = r * C
    diag["expected_rank_M"] = R
    if r < 1:
        raise ValueError("r = 0: M has (numerically) zero rank.")
    if R > M.shape[1]:
        raise ValueError(f"r*binom(q-1,p) = {R} exceeds the size {M.shape[1]} of M; "
                         "r is too large for this p (increase p or n).")
    if r > (p + 1) * m:
        raise ValueError(f"r = {r} exceeds (p+1)(n-q) = {(p + 1) * m}, the dimension of "
                         "R^{p+1} (x) R^{n-q} that contains B.")
    if rank != R:
        failures.append(f"Step 1: numerical rank(M) = {rank} != r*binom(q-1,p) = {R} "
                        "(condition (v) of Theorem 4.3).")
    diag["M_last_kept"] = float(s_rel[R - 1])
    diag["M_first_dropped"] = float(s_rel[R]) if R < s_rel.size else 0.0
    timings["step1"] = time.perf_counter() - t0

    # ---- Step 2: B = (phi (x) I)(colspan(M) cap Z (x) R^m) -----------------
    t0 = time.perf_counter()
    B_basis, sines = subspace_B(U[:, :R], p, m, r)
    inter_dim = int(np.sum(sines <= np.sqrt(tol)))
    diag.update(intersection_dim=inter_dim, sines=sines,
                sine_r=float(sines[r - 1]),
                sine_r_plus_1=float(sines[r]) if r < sines.size else np.nan)
    if inter_dim != r:
        failures.append(f"Step 2: dim(colspan(M) cap Z(x)R^m) = {inter_dim} != r = {r} "
                        "(Lemma 4.4 / condition (vi) fails numerically).")
    timings["step2"] = time.perf_counter() - t0

    # ---- Step 3: rank-1 matrices in B via JLV ------------------------------
    t0 = time.perf_counter()
    deltas, alphas, Y, jinfo = find_rank_one_matrices(
        B_basis, (p + 1, m), r, kernel=jlv_kernel, tol=tol, check_tol=check_tol,
        tries=jlv_tries, seed=seed)
    diag.update(jinfo)
    if jinfo["jlv_kernel_dim"] != r:
        failures.append(f"Step 3: JLV found {jinfo['jlv_kernel_dim']} rank-1 directions, "
                        f"not r = {r} (output 'fail').")
    if deltas is None or deltas.shape[1] != r:
        # Nothing (or the wrong number) came back: Steps 4-5 cannot run.
        failures.append("Step 3: JLV returned no set of r rank-1 matrices; stopping here.")
        timings["step3"] = time.perf_counter() - t0
        res = SymKYFResult(components=np.full((n, r), np.nan), r=r, success=False,
                           failures=failures, diagnostics=diag, B_basis=B_basis)
        if verbose:
            print(res.summary())
        return res
    if jinfo["max_rank1_ratio"] > check_tol or jinfo["jennrich_rel_error"] > check_tol:
        failures.append("Step 3: the elements returned by JLV are not numerically rank-1 "
                        f"(max sigma2/sigma1 = {jinfo['max_rank1_ratio']:.1e}, "
                        f"Jennrich rel. error = {jinfo['jennrich_rel_error']:.1e}).")
    if jinfo["max_pairwise_overlap"] > 1 - check_tol:
        failures.append("Step 3: two recovered rank-1 elements coincide, so fewer than r "
                        "distinct ones were found.")
    timings["step3"] = time.perf_counter() - t0

    # ---- Step 4: T_1 = sum x_l alpha_l^(x)3 -> a_tilde ---------------------
    t0 = time.perf_counter()
    a_tilde, x, res1 = solve_tail(T, q, alphas)
    diag.update(x=x, T1_rel_residual=res1, min_abs_x=float(np.min(np.abs(x))))
    if res1 > check_tol:
        failures.append(f"Step 4: T_1 system has no (numerical) solution, rel. residual {res1:.1e}.")
    if np.min(np.abs(x)) <= check_tol * np.max(np.abs(x)):
        failures.append("Step 4: some x_l = 0, so beta_l^{-3} is not defined.")
    timings["step4"] = time.perf_counter() - t0

    # ---- Step 5: T_2 = sum a_hat_l (x) a_tilde_l (x) a_tilde_l -> a_hat ----
    t0 = time.perf_counter()
    a_hat, res2 = solve_head(T, q, a_tilde)
    diag["T2_rel_residual"] = res2
    if res2 > check_tol:
        failures.append(f"Step 5: T_2 system has no (numerical) solution, rel. residual {res2:.1e}.")
    timings["step5"] = time.perf_counter() - t0

    # ---- Output and checks -------------------------------------------------
    A = np.vstack([a_hat, a_tilde])                     # n x r
    rec = np.linalg.norm(symmetric_cp_tensor(A) - T) / max(np.linalg.norm(T), 1e-300)
    diag["reconstruction_rel_error"] = float(rec)
    if rec > check_tol:
        failures.append(f"Final check: ||sum_l a_l^(x)3 - T|| / ||T|| = {rec:.1e}.")
    # delta^(l) should be parallel to d^(l) = (a_1..a_{p+1}) of the same component.
    diag["delta_consistency"] = float(_sines_between(deltas, a_hat[: p + 1]).max())

    res = SymKYFResult(components=A, r=r, success=not failures, failures=failures,
                       diagnostics=diag, deltas=deltas, alphas=alphas, B_basis=B_basis)
    if verbose:
        print(res.summary())
    return res


# ---------------------------------------------------------------------------
# Utilities: instances, evaluation, Theorem 4.3 hypotheses, parameter bounds
# ---------------------------------------------------------------------------


def symmetric_cp_tensor(A):
    """T = sum_l a_l (x) a_l (x) a_l for the columns a_l of the n x r array A."""
    A = np.asarray(A, dtype=float)
    return np.einsum("il,jl,kl->ijk", A, A, A, optimize=True)


def random_instance(n, r, seed=None):
    """Gaussian components A (n x r) and T = sum_l a_l^(x)3."""
    A = np.random.default_rng(seed).standard_normal((n, r))
    return symmetric_cp_tensor(A), A


def match_components(A_true, A_est):
    """Match recovered components to the true ones (no sign or scale freedom).

    Returns (perm, rel_errors): column perm[l] of A_est is matched to column l
    of A_true, and rel_errors[l] = ||a_l - a_est_perm[l]|| / ||a_l||.
    """
    A_true = np.asarray(A_true, float)
    A_est = np.asarray(A_est, float)
    if not np.all(np.isfinite(A_est)):          # e.g. the NaN output of a failed run
        return np.full(A_true.shape[1], -1), np.full(A_true.shape[1], np.inf)
    cost = np.linalg.norm(A_true[:, :, None] - A_est[:, None, :], axis=0)
    rows, cols = linear_sum_assignment(cost)
    perm = np.empty(A_true.shape[1], dtype=int)
    perm[rows] = cols
    err = cost[rows, cols] / np.linalg.norm(A_true[:, rows], axis=0)
    out = np.empty(A_true.shape[1])
    out[rows] = err
    return perm, out


def _rank_report(X, tol):
    """(has full column rank, sigma_min / sigma_max) for a matrix X."""
    if X.shape[1] == 0:
        return True, 1.0
    if X.shape[0] < X.shape[1]:
        return False, 0.0
    s = _relative_singular_values(X)
    return bool(s[-1] > tol), float(s[-1])


def check_uniqueness_conditions(A, p, tol=1e-8):
    """Evaluate hypotheses (i)-(vii) of Theorem 4.3 for components A (n x r).

    N and P are the matrices of [KMW24, Definition 2.6] with b -> a_tilde and
    n_2 -> n - q (as in the project note ``definition_NP_symmetric.tex``):

    * N: rows (S, j) with S not inside {0..p}, columns (U, l) with 0 in U and
      U != {0..p}; entry A(a_hat_l; p)_{S,U} * a_tilde_l[j].
    * P: rows (i1 < i2 <= p, j1 < j2 < m), columns (l1 < l2); entry
      (d_l1[i1] d_l2[i2] - d_l1[i2] d_l2[i1]) (a~_l1[j1] a~_l2[j2] - a~_l1[j2] a~_l2[j1]),
      which equals the four-term formula of the definition.

    Returns a dict {label: (holds, measure)}; for rank conditions the measure
    is sigma_min / sigma_max of the relevant matrix.
    """
    A = np.asarray(A, dtype=float)
    n, r = A.shape
    q, m = _check_params(n, p)
    D, Ah, At = A[: p + 1], A[:q], A[q:]
    out = {}

    a1 = np.abs(A[0])
    out["(i) a_1 != 0"] = (bool(a1.min() > tol * np.abs(A).max()), float(a1.min()))

    if r >= 2:
        l1, l2 = np.array(list(itertools.combinations(range(r), 2))).T
        min_sine = float(_sines_between(D[:, l1], D[:, l2]).min())
    else:
        min_sine = 1.0
    out["(ii) d_l pairwise lin. indep."] = (min_sine > tol, min_sine)

    out["(iii) {d_l (x) a~_l} lin. indep."] = _rank_report(sla.khatri_rao(D, At), tol)
    out["(iv) {a~_l (x) a~_l} lin. indep."] = _rank_report(sla.khatri_rao(At, At), tol)

    C = math.comb(q - 1, p)
    M = koszul_young_flattening(symmetric_cp_tensor(A), p)
    s = _relative_singular_values(M)
    rank = int(np.sum(s > tol))
    out["(v) rank M = r C(q-1,p)"] = (rank == r * C,
                                      float(s[r * C - 1]) if r * C <= s.size else 0.0)

    rows, cols, *_ = _koszul_pattern(q, p)
    V = tuple(range(p + 1))
    rkeep = [k for k, S in enumerate(rows) if not set(S) <= set(V)]
    ckeep = [k for k, U in enumerate(cols) if 0 in U and U != V]
    N = np.hstack([np.kron(koszul_matrix(Ah[:, l], p)[np.ix_(rkeep, ckeep)], At[:, l:l + 1])
                   for l in range(r)])
    out["(vi) N full column rank"] = _rank_report(N, tol)

    if r >= 2 and p >= 1 and m >= 2:
        i1, i2 = np.array(list(itertools.combinations(range(p + 1), 2))).T
        j1, j2 = np.array(list(itertools.combinations(range(m), 2))).T
        l1, l2 = np.array(list(itertools.combinations(range(r), 2))).T
        Dw = D[i1][:, l1] * D[i2][:, l2] - D[i2][:, l1] * D[i1][:, l2]
        Aw = At[j1][:, l1] * At[j2][:, l2] - At[j2][:, l1] * At[j1][:, l2]
        out["(vii) P full column rank"] = _rank_report(sla.khatri_rao(Dw, Aw), tol)
    else:
        out["(vii) P full column rank"] = (True, 1.0)
    return out


def parameter_bounds(n, p):
    """Bounds on r for given (n, p), for choosing parameters.

    Keys:
      thm31        : (2 - 1/(p+1)) n - (6p+2), the sufficient condition of
                     Theorem 3.1 (rank detection).
      jlv_pdf      : p (n-q-1)/4, the [JLV23] bound quoted in Section 4.
      count_M      : (n-q)(2p+1)/(p+1); r*binom(q-1,p) must not exceed the size
                     of M (necessary for condition (v)).
      count_N      : (binom(q,p)-p-1)(n-q)/(binom(q-1,p)-1); N needs at least as
                     many rows as columns (necessary for condition (vi)).
      count_P      : largest r with r(r-1)/2 <= binom(p+1,2) binom(n-q,2); P needs
                     at least as many rows as columns (necessary for (vii)); this
                     is also the conjectured threshold ~ (p+1)(n-q)/sqrt(2) of the
                     JLV step from the repo's companion paper (arXiv:2504.17947).
      count_aa     : binom(n-q+1, 2), Lemma 4.1 (condition (iv)).
    The count_* entries are necessary dimension counts, not guarantees.
    """
    q = 2 * p + 1
    m = n - q
    rows_P = math.comb(p + 1, 2) * math.comb(m, 2)
    count_P = int((1 + math.isqrt(1 + 8 * rows_P)) // 2)
    while count_P * (count_P - 1) // 2 > rows_P:
        count_P -= 1
    C, Cq = math.comb(q - 1, p), math.comb(q, p)
    return {
        "thm31": (2 - 1 / (p + 1)) * n - (6 * p + 2),
        "jlv_pdf": p * (m - 1) / 4,
        "count_M": m * q / (p + 1),
        "count_N": (Cq - p - 1) * m / (C - 1) if C > 1 else float(m),
        "count_P": count_P,
        "count_aa": math.comb(m + 1, 2),
    }


if __name__ == "__main__":  # quick self-test
    n, p, r = 30, 2, 33
    T, A_true = random_instance(n, r, seed=0)
    res = decompose(T, p, r, seed=0, verbose=True)
    _, err = match_components(A_true, res.components)
    print(f"max relative component error: {err.max():.2e}")
