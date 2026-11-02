# Symmetric Koszul–Young decomposition (Algorithm 1)

This is a Python implementation of **Algorithm 1** from *Symmetric Extension of Overcomplete Tensor Decomposition via Koszul–Young Flattenings* (Luo & Wein, draft). It recovers the components of a real symmetric tensor `T = Σ_ℓ a_ℓ ⊗ a_ℓ ⊗ a_ℓ` (n×n×n), including the overcomplete case r > n.

Step 3 (finding the rank-1 matrices in the subspace B) uses the JLV implementation from [lemniscate8/linear-intersect-varieties](https://github.com/lemniscate8/linear-intersect-varieties) through its function `linear_conic_intersect.demix_subspace`.

| File | What it is |
|---|---|
| `symmetric_kyf.py` | The implementation: Algorithm 1, rank detection, a checker for the hypotheses of Theorem 4.3, and parameter bounds |
| `test_symmetric_kyf.py` | 17 unit tests: eq. (1) entry by entry, Lemmas 2.1, 2.2, 4.1, 4.4, φ(z^(V)) = d, Theorem 3.1, end-to-end recovery, and both Step 3 kernel options (about 5 s) |
| `demo_symmetric_kyf.py` | Worked examples, a threshold sweep, a noise test, and a comparison of the two Step 3 kernel options (about 20 s; add `--big` for p = 4) |

## Setup

```bash
pip install "numpy>=2" scipy        # the JLV code uses np.bool, which needs NumPy 2.x
git clone https://github.com/lemniscate8/linear-intersect-varieties
export JLV_REPO_PATH=$PWD/linear-intersect-varieties   # or put the clone next to symmetric_kyf.py
python -m unittest test_symmetric_kyf -v
python demo_symmetric_kyf.py
```

You can also call `symmetric_kyf.set_jlv_path("/path/to/linear-intersect-varieties")` from Python. The JLV code is imported, not copied. That repository is GPL-3.0.

## Usage

```python
import symmetric_kyf as sk

T, A_true = sk.random_instance(n=30, r=33, seed=0)   # T = sum of 33 cubes in R^30
res = sk.decompose(T, p=2, r=33)                     # r=None learns r from rank(M) (Thm 3.1)
# res = sk.decompose(T, p=2, r=33, jlv_kernel="repo")  # Step 3 with the repo's own null-space cutoff
print(res.summary())                                  # per-step diagnostics
A = res.components                                    # 30 x 33, columns a_l (any order)
_, err = sk.match_components(A_true, A)               # exact recovery, no sign/scale freedom
```

## How the code follows the draft

| Draft | Function | What it does |
|---|---|---|
| eq. (1), Step 1 | `koszul_young_flattening(T, p)` | Builds `M(T;p)`. Block (S, U=S⊔{i}) is `σ(U,i)·T[i, q:, q:]` |
| eq. (3) | `koszul_matrix(a_hat, p)` | Builds `A(â;p)`. Column U is `z^(U)(â;p)` |
| Theorem 3.1 | `detect_rank(T, p)` | Returns `r = rank(M) / binom(q-1,p)` |
| eq. (7), Step 2 | `subspace_B(Q, p, m, r)` | Computes `colspan(M) ∩ (Z_{p,q} ⊗ R^{n-q})`, then applies `φ_{p,q}: e_{[p+1]\{i}} ↦ (-1)^{i-1} e_i` |
| Step 3, [JLV23] | `find_rank_one_matrices(B, (p+1, n-q), r, kernel=...)` | Calls `lci.demix_subspace`. Fails if the kernel dimension (the number of rank-1 tensors found) is not r |
| Step 4, eq. (9) | `solve_tail(T, q, alphas)` | Solves `T_1 = Σ x_ℓ α̃_ℓ^{⊗3}` and sets `ã_ℓ = cbrt(x_ℓ) α̃_ℓ` |
| Step 5, Lemma 4.1 | `solve_head(T, q, a_tilde)` | Solves `T_2 = Σ â_ℓ ⊗ ã_ℓ ⊗ ã_ℓ` |
| everything | `decompose(T, p, r)` | Runs Steps 1–5, then checks `Σ a_ℓ^{⊗3} = T` |
| Theorem 4.3 (i)–(vii) | `check_uniqueness_conditions(A, p)` | Evaluates the hypotheses. N and P follow `definition_NP_symmetric.tex` |

Indices are 0-based in the code. Paper coordinate i is array index i−1, and `ã_j = a_{q+j}` is `a[q+j]`. The (p+1)×(n−q) matrices in B are vectorized row-major, which is the JLV code's convention for `dims=[p+1, n-q]`.

**The kernel in Step 3: two options.** `demix_subspace` takes a `kernel_method`, and `decompose(..., jlv_kernel=...)` chooses it:

- `jlv_kernel="repo"` calls the repo's own `extract_kernel_naive`, unchanged. That function uses `scipy.linalg.null_space` with its machine-precision cutoff, σ_max·ε·max(matrix shape). The number of rank-1 matrices found is however many singular values fall below that cutoff.
- `jlv_kernel="svd"` (default) keeps the r smallest right singular vectors of the same matrix. It counts the singular values below `tol` as the number found.

The rest of Step 3 is the repo's code in both cases: the symmetric lift, the 2×2-minor quadrics and Jennrich's simultaneous diagonalization.

The two options give the same result on noiseless input. The repo's cutoff found exactly r kernel vectors in all 26 test runs (p = 1–4, including r at the threshold). The r-th singular value was 2e-16 to 6e-14, against a cutoff of 5e-15 to 8e-13, with the next singular value at 4e-3 or more. The margin was sometimes small, though, as little as a factor of 2.3. With noise the two options differ (p = 2, n = 30, r = 33, relative noise in T):

| Noise | `"repo"` kernel found | `"repo"` result | `"svd"` result |
|---|---|---|---|
| 1e-14 | 33/33 | ok | ok |
| 1e-13 | 19–22/33 | flagged as fail (components still accurate to within 1e-9) | ok |
| 1e-12 or more | 0/33 | fails, no components | ok |

**Diagnostics.** `res.diagnostics` holds the quantities the draft relies on:

- rank(M) and the singular-value gap
- dim of the Step 2 intersection, with the sines of the principal angles
- the JLV kernel dimension and the Jennrich error
- the residuals of the T₁ and T₂ systems
- the reconstruction error
- a consistency check that δ_ℓ ∥ (a₁, …, a_{p+1})_ℓ

`res.success` is False if any check fails, and `res.failures` says which one. The computation still runs to the end, so you can inspect every step.

## Results (random Gaussian components, `demo_symmetric_kyf.py`)

| p | n | r | r/n | size of M | max relative component error | time |
|---|---|---|---|---|---|---|
| 1 | 12 | 6 | 0.50 | 27 | 4e-15 | <0.1 s |
| 2 | 30 | 33 | 1.10 | 250 | 2e-12 | 0.3 s |
| 3 | 30 | 36 | 1.20 | 805 | 3e-12 | 0.7 s |
| 3 | 40 | 50 | 1.25 | 1155 | 6e-13 | 4 s |
| 4 | 40 | 54 | 1.35 | 3906 | 3e-12 | 29 s |

With noise (p = 2, n = 30, r = 30, r given), a relative noise of 1e-8, 1e-6 or 1e-4 gives a median component error of 1.9e-7, 1.6e-5 or 1.5e-3. For noisy input, pass r explicitly and loosen `tol` and `check_tol`.

## Observations relevant to the draft

1. **Condition (vi) on N is what limits the algorithm in practice.** Success holds exactly up to the dimension count for N, `r ≤ (binom(q,p) − p − 1)(n−q) / (binom(q−1,p) − 1)`. For p = 2, n = 30 that is 35: r = 35 works and r = 36 fails. For p = 3, n = 30 it is 37.5: r = 37 works and r = 38 fails. Past that point, (vi) is the only hypothesis of Theorem 4.3 that fails. The Step 2 intersection then has dimension `r + (#cols N − #rows N)` (41 and 47 in these examples). Rank detection (Theorem 3.1) keeps working past this point. The count is 1.40, 1.63, 1.75 and 1.82 times (n−q) for p = 2, 3, 4, 5, approaching `(2 − 1/(p+1))(n−q)`. For comparison, the bound for N from [KMW24, Lemma 3.6] gives `(2 − 3/(p+1))(n−q) − q³/4`, so a generic-success theorem stated with that bound would be conservative.
2. **The JLV term in the bounds is far from tight.** Section 4 uses r ≤ p(n−q−1)/4 for the JLV step, which is 12 for p = 2, n = 30, while the algorithm succeeds up to 35. The repo's companion paper (Dastidar–Weicht–Wein, arXiv:2504.17947) proves roughly (p+1)(n−q)/2 asymptotically and conjectures (p+1)(n−q)/√2. In these runs JLV was never the step that failed.
3. **M is (skew-)symmetric up to a signed permutation.** Index the columns by the complement [q]∖U, a p-subset, and multiply row (S, j) by (−1)^{ΣS}. The result is symmetric for even p and skew-symmetric for odd p (checked numerically for p = 1–4). This gives a one-line argument for why the switched flattening M′ equals M, and Step 1 could use an eigendecomposition for even p.
4. **Notation.** The code reads Lemmas 4.4–4.5 and the end of the proof of Theorem 4.3 with ℝ^{n−q} and d^{(ℓ)} ⊗ ã^{(ℓ)}. The current PDF still has ℝ^{n−(p+1)}, φ ⊗ Π and â ⊗ ã there. N and P appear in Theorem 4.3 but are not defined in this PDF.

**Cost.** The SVD of the N×N matrix M, with N = binom(q,p)(n−q), dominates the run time at O(N³). For p = 4, n = 40, N = 3906, it takes about 25 s. Step 2 only works with the (p+1)(n−q) rows of Z, so it is cheap.
