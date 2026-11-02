"""Unit tests for symmetric_kyf.py (run: python -m unittest test_symmetric_kyf -v).

The first group checks the building blocks against the statements in the
draft (Lemma 2.1, Lemma 2.2, phi(z^(V)) = d, Theorem 3.1, Lemma 4.1,
Lemma 4.4). The second group runs Algorithm 1 end to end.
"""

import itertools
import math
import unittest

import numpy as np
from scipy import linalg as sla

import symmetric_kyf as sk


def principal_sines(X, Y):
    """Sines of the principal angles between colspan(X) and colspan(Y)
    (singular values of the part of Y orthogonal to X; accurate near 0)."""
    Qx, Qy = sla.orth(X), sla.orth(Y)
    return sla.svdvals(Qy - Qx @ (Qx.T @ Qy))


class TestBuildingBlocks(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_lemma_2_2_rank_of_A(self):
        for p in (1, 2, 3):
            q = 2 * p + 1
            a = self.rng.standard_normal(q)
            A = sk.koszul_matrix(a, p)
            self.assertEqual(A.shape, (math.comb(q, p), math.comb(q, p + 1)))
            self.assertEqual(np.linalg.matrix_rank(A), math.comb(q - 1, p))
            self.assertEqual(np.linalg.matrix_rank(sk.koszul_matrix(np.zeros(q), p)), 0)
            # columns with 0 in U span colspan(A); rows with 0 not in S span rowspan(A)
            rows, cols = sk._koszul_pattern(q, p)[:2]
            cU = [k for k, U in enumerate(cols) if 0 in U]
            rS = [k for k, S in enumerate(rows) if 0 not in S]
            self.assertEqual(len(cU), math.comb(q - 1, p))
            self.assertEqual(np.linalg.matrix_rank(A[:, cU]), math.comb(q - 1, p))
            self.assertEqual(np.linalg.matrix_rank(A[rS]), math.comb(q - 1, p))

    def test_flattening_matches_eq_1_entrywise(self):
        # direct transcription of eq. (1) with 1-based indices, as in the draft
        from itertools import combinations
        for p, n in ((1, 6), (2, 9)):
            q, m = 2 * p + 1, n - 2 * p - 1
            T, _ = sk.random_instance(n, 3, seed=5)
            rows = list(combinations(range(1, q + 1), p))
            cols = list(combinations(range(1, q + 1), p + 1))
            M = np.zeros((len(rows) * m, len(cols) * m))
            for a, S in enumerate(rows):
                for j in range(1, m + 1):
                    for b, U in enumerate(cols):
                        for k in range(1, m + 1):
                            val = 0.0
                            for i in range(1, q + 1):
                                if set(U) == set(S) | {i} and i not in S:
                                    sgn = (-1) ** len([x for x in U if x < i])
                                    val += sgn * T[i - 1, j + q - 1, k + q - 1]
                            M[a * m + j - 1, b * m + k - 1] = val
            np.testing.assert_allclose(sk.koszul_young_flattening(T, p), M, atol=1e-13)

    def test_lemma_2_1_rank_one_flattening(self):
        for p, n in ((1, 7), (2, 10), (3, 12)):
            q = 2 * p + 1
            a = self.rng.standard_normal(n)
            T = np.einsum("i,j,k->ijk", a, a, a)
            M = sk.koszul_young_flattening(T, p)
            expected = np.kron(sk.koszul_matrix(a[:q], p), np.outer(a[q:], a[q:]))
            np.testing.assert_allclose(M, expected, atol=1e-12)

    def test_phi_maps_zV_to_d(self):
        for p in (1, 2, 3):
            q = 2 * p + 1
            a = self.rng.standard_normal(q)
            rows, S_of_i = sk._Z_blocks(p)
            cols = sk._koszul_pattern(q, p)[1]
            zV = sk.koszul_matrix(a, p)[:, cols.index(tuple(range(p + 1)))]
            # z^(V) is supported on the rows S inside V = {0..p}
            support = np.flatnonzero(zV)
            self.assertTrue(set(support) <= set(S_of_i))
            phi_zV = np.array([(-1) ** i * zV[S_of_i[i]] for i in range(p + 1)])
            np.testing.assert_allclose(phi_zV, a[: p + 1], atol=1e-14)

    def test_theorem_3_1_rank_of_M(self):
        for p, n, r in ((1, 14, 10), (2, 24, 26), (3, 30, 32)):
            T, _ = sk.random_instance(n, r, seed=1)
            r_est, rank, _ = sk.detect_rank(T, p)
            self.assertEqual(rank, r * math.comb(2 * p, p))
            self.assertEqual(r_est, r)

    def test_lemma_4_1_symmetric_khatri_rao(self):
        m = 4
        r_max = math.comb(m + 1, 2)
        X = self.rng.standard_normal((m, r_max))
        self.assertEqual(np.linalg.matrix_rank(sla.khatri_rao(X, X)), r_max)
        X = self.rng.standard_normal((m, r_max + 1))
        self.assertEqual(np.linalg.matrix_rank(sla.khatri_rao(X, X)), r_max)

    def test_lemma_4_4_B_is_span_of_d_times_atilde(self):
        p, n, r = 2, 30, 33
        q, m = 2 * p + 1, n - 2 * p - 1
        T, A = sk.random_instance(n, r, seed=2)
        M = sk.koszul_young_flattening(T, p)
        U = sla.svd(M)[0][:, : r * math.comb(q - 1, p)]
        B, sines = sk.subspace_B(U, p, m, r)
        self.assertLess(sines[r - 1], 1e-10)        # r directions in the intersection
        self.assertGreater(sines[r], 1e-4)          # ... and no more (no spurious ones)
        truth = sla.khatri_rao(A[: p + 1], A[q:])   # columns d_l (x) a~_l (row-major)
        self.assertLess(principal_sines(B, truth).max(), 1e-9)


class TestAlgorithm1(unittest.TestCase):
    def assertRecovers(self, p, n, r, seed=0, r_input="given"):
        T, A = sk.random_instance(n, r, seed=seed)
        res = sk.decompose(T, p, r if r_input == "given" else None, seed=seed)
        self.assertTrue(res.success, msg=res.summary())
        self.assertEqual(res.r, r)
        _, err = sk.match_components(A, res.components)
        self.assertLess(err.max(), 1e-8, msg=res.summary())
        return res

    def test_undercomplete(self):
        self.assertRecovers(1, 12, 6)

    def test_overcomplete_p2(self):
        self.assertRecovers(2, 30, 33)          # r > n

    def test_overcomplete_p3(self):
        self.assertRecovers(3, 30, 36)          # r > n

    def test_rank_learned_from_M(self):
        self.assertRecovers(2, 30, 32, r_input="learned")

    def test_small_r(self):
        self.assertRecovers(1, 5, 1)
        self.assertRecovers(2, 8, 2)

    def test_conditions_of_theorem_4_3_hold_generically(self):
        _, A = sk.random_instance(30, 33, seed=3)
        cond = sk.check_uniqueness_conditions(A, 2)
        for name, (holds, _) in cond.items():
            self.assertTrue(holds, msg=name)

    def test_fails_beyond_dimension_count_of_N(self):
        p, n = 2, 30
        r = int(sk.parameter_bounds(n, p)["count_N"]) + 1   # = 36
        T, A = sk.random_instance(n, r, seed=4)
        cond = sk.check_uniqueness_conditions(A, p)
        self.assertFalse(cond["(vi) N full column rank"][0])
        res = sk.decompose(T, p, r, seed=0, jlv_tries=1)
        self.assertFalse(res.success)
        self.assertGreater(res.diagnostics["intersection_dim"], r)

    def test_repo_kernel_machine_precision_noiseless(self):
        # Step 3 with the repo's own kernel routine (null_space, eps cutoff)
        T, A = sk.random_instance(30, 33, seed=0)
        res = sk.decompose(T, 2, 33, jlv_kernel="repo", seed=0)
        self.assertTrue(res.success, msg=res.summary())
        self.assertEqual(res.diagnostics["jlv_kernel_dim"], 33)
        _, err = sk.match_components(A, res.components)
        self.assertLess(err.max(), 1e-8)

    def test_repo_kernel_on_noisy_input(self):
        T, A = sk.random_instance(30, 33, seed=0)
        G = np.random.default_rng(1).standard_normal(T.shape)
        G = sum(G.transpose(s) for s in itertools.permutations(range(3)))
        T = T + 1e-10 * np.linalg.norm(T) * G / np.linalg.norm(G)
        res = sk.decompose(T, 2, 33, jlv_kernel="repo", seed=0)
        self.assertFalse(res.success)                      # fails cleanly, no exception
        self.assertEqual(res.diagnostics["jlv_kernel_dim"], 0)
        self.assertTrue(np.isnan(res.components).all())
        res = sk.decompose(T, 2, 33, seed=0, tol=1e-6, check_tol=1e-4)   # default kernel
        self.assertTrue(res.success, msg=res.summary())
        _, err = sk.match_components(A, res.components)
        self.assertLess(err.max(), 1e-5)

    def test_invalid_parameters(self):
        T, _ = sk.random_instance(6, 2, seed=0)
        with self.assertRaises(ValueError):
            sk.decompose(T, 3)                   # q = 7 > n - 2
        T, _ = sk.random_instance(12, 4, seed=0)
        with self.assertRaises(ValueError):
            sk.decompose(T, 1, 4, jlv_kernel="other")


if __name__ == "__main__":
    unittest.main()
