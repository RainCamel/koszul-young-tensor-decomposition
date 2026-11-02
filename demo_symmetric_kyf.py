"""Demo of Algorithm 1 (symmetric Koszul-Young decomposition).

    python demo_symmetric_kyf.py          # about 30 seconds
    python demo_symmetric_kyf.py --big    # adds p = 4, n = 40, r = 54 (~30 s more)

Set JLV_REPO_PATH to your clone of lemniscate8/linear-intersect-varieties
(or put the clone next to this file).
"""

import sys
import time

import numpy as np

import symmetric_kyf as sk


def run(p, n, r, seed=0, noise=0.0, **kw):
    T, A = sk.random_instance(n, r, seed=seed)
    if noise:
        G = np.random.default_rng(seed + 1).standard_normal(T.shape)
        G = sum(G.transpose(s) for s in [(0, 1, 2), (0, 2, 1), (1, 0, 2),
                                          (1, 2, 0), (2, 0, 1), (2, 1, 0)])
        T = T + noise * np.linalg.norm(T) * G / np.linalg.norm(G)
    t = time.perf_counter()
    res = sk.decompose(T, p, r, seed=seed, **kw)
    t = time.perf_counter() - t
    _, err = sk.match_components(A, res.components)
    return res, err, t, A


def header():
    print(f"{'p':>2} {'n':>3} {'r':>3} {'r/n':>5} | {'M size':>6} {'rank M':>9} {'dim int':>7} "
          f"{'JLV ker':>7} | {'max err':>8} {'result':>7} {'time':>6}")


def row(p, n, r, res, err, t):
    d = res.diagnostics
    print(f"{p:>2} {n:>3} {r:>3} {r / n:>5.2f} | {d['M_shape'][0]:>6} "
          f"{d['rank_M']:>4}/{d['expected_rank_M']:<4} {d['intersection_dim']:>3}/{r:<3} "
          f"{d['jlv_kernel_dim']:>3}/{r:<3} | {err.max():>8.1e} "
          f"{'ok' if res.success else 'FAIL':>7} {t:>5.1f}s")


def main(big=False):
    sk._jlv()  # fail early with a clear message if the JLV repo is missing

    print("=" * 78)
    print("1. One run in detail (n = 30, r = 33 > n, p = 2)")
    print("=" * 78)
    res, err, _, _ = run(2, 30, 33)
    print(res.summary())
    print(f"  max_l ||a_l - a_l(recovered)|| / ||a_l|| = {err.max():.2e}\n")

    print("=" * 78)
    print("2. Several sizes (r given)")
    print("=" * 78)
    header()
    cases = [(1, 12, 6), (1, 20, 14), (2, 20, 18), (2, 30, 33), (3, 30, 36), (3, 40, 50)]
    if big:
        cases.append((4, 40, 54))
    for p, n, r in cases:
        res, err, t, _ = run(p, n, r)
        row(p, n, r, res, err, t)

    print("\n" + "=" * 78)
    print("3. Where does it stop working?  Sweep r around the bounds")
    print("=" * 78)
    for p, n in [(2, 30), (3, 30)]:
        b = sk.parameter_bounds(n, p)
        print(f"\n  n = {n}, p = {p}: Thm 3.1 bound {b['thm31']:.1f}, [JLV23] bound in draft "
              f"{b['jlv_pdf']:.1f}, count for N {b['count_N']:.1f}, count for M "
              f"{b['count_M']:.1f}, count for P {b['count_P']}")
        header()
        r_N = int(b["count_N"])
        for r in range(r_N - 1, r_N + 3):
            res, err, t, A = run(p, n, r, jlv_tries=1)
            row(p, n, r, res, err, t)
            if not res.success:
                failed = [k for k, (ok, _) in sk.check_uniqueness_conditions(A, p).items() if not ok]
                print(f"{'':>17}Theorem 4.3 conditions that fail: {', '.join(failed) or 'none'}")

    print("\n" + "=" * 78)
    print("4. Noise (p = 2, n = 30, r = 30; r given; loose tolerances)")
    print("=" * 78)
    for eps in [1e-10, 1e-8, 1e-6, 1e-4]:
        res, err, _, _ = run(2, 30, 30, noise=eps, tol=1e-4, check_tol=1e-2)
        print(f"  relative noise {eps:.0e}: max component error {err.max():.1e}, "
              f"median {np.median(err):.1e}, residual {res.diagnostics['reconstruction_rel_error']:.1e}")

    print("\n" + "=" * 78)
    print("5. Step 3 kernel: the repo's machine-precision cutoff vs. the default")
    print("   (p = 2, n = 30, r = 33)")
    print("=" * 78)
    for eps in [0.0, 1e-14, 1e-13, 1e-12]:
        for mode in ("repo", "svd"):
            res, err, _, _ = run(2, 30, 33, noise=eps, jlv_kernel=mode,
                                 tol=1e-6 if eps else 1e-8, check_tol=1e-4 if eps else 1e-6)
            d = res.diagnostics
            print(f"  noise {eps:.0e}  jlv_kernel={mode!r:6}: kernel {d['jlv_kernel_dim']:>2}/33 "
                  f"(cutoff {d['jlv_cutoff']:.0e}), {'ok  ' if res.success else 'FAIL'}, "
                  f"max component error {err.max():.1e}")


if __name__ == "__main__":
    main(big="--big" in sys.argv)
