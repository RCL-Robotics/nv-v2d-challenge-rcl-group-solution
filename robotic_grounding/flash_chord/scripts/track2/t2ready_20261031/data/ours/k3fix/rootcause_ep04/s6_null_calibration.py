"""MODEL-ONLY (no data): null distribution of the FIX-A run statistic max_{1<=k<n} |S_k|^2/(k q^2) for a pure 3D Gaussian
random walk (the frozen bias model). The statistic is scale-free, so q drops out. The pointwise chi2_3(99%)=11.34 used in
fixA is the per-lag quantile; the run test takes a max over lags, so its false-rejection rate is much higher.
Writes s6_null_q99.npz (q99[n] for n = 0..1200) and prints the false-rejection rate of 11.34."""
import numpy as np
from pathlib import Path
OUT = Path("/root/autodl-tmp/haoran.zhang/tmp_analysis/tier2/k3_rootcause/ep04")
N, NS, B = 1200, 20000, 1000
rng = np.random.default_rng(12345)
cm = np.empty((NS, N), np.float32)
k = np.arange(1, N + 1)
for i in range(0, NS, B):
    S = np.cumsum(rng.standard_normal((B, N, 3)), axis=1)
    st = (S ** 2).sum(2) / k
    cm[i:i + B] = np.maximum.accumulate(st, axis=1)
# a run of n frames has lags 1..n-1 -> column n-2
q99 = np.zeros(N + 1); q95 = np.zeros(N + 1); fr = np.zeros(N + 1)
for n in range(2, N + 1):
    col = cm[:, n - 2]
    q99[n] = np.percentile(col, 99); q95[n] = np.percentile(col, 95); fr[n] = (col >= 11.344867).mean()
np.savez(OUT / "s6_null_q99.npz", q99=q99, q95=q95, false_reject_11_34=fr)
for n in (10, 15, 23, 30, 40, 63, 91, 97, 115, 138, 153, 300, 600, 1000):
    print(f"n={n:5d}  q99={q99[n]:6.2f}  q95={q95[n]:6.2f}  P(stat>=11.34 | pure walk)={fr[n]:.3f}")
