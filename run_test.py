import symmetric_kyf as sk

T, A_true = sk.random_instance(n=30, r=33, seed=0)  # or your own n×n×n symmetric numpy array
res = sk.decompose(T, p=2, r=33)   # r=None → rank is detected from M (Thm 3.1)
print(res.summary())               # same per-step report as in demo section 1
print(res.success, res.failures)   # True, [] if all checks passed
A = res.components                 # n × r, columns are the a_ℓ (in any order)
_, err = sk.match_components(A_true, A)
#print(err)
