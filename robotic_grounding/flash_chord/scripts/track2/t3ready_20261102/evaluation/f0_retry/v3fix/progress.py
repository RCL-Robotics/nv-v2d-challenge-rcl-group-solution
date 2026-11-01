"""progress.py: read-only progress of the v3 retry trainers. Per launched clip: gpu, pid alive, last logged iteration
(metrics.jsonl step / 98304), samples appended to v3fix/progress_samples.tsv; rate = it/h over the last >= 10 min of
samples (or since the first sample past the resume point). Starts nothing, signals nothing."""
import json, os, time
from pathlib import Path
RA = Path("/root/autodl-tmp/haoran.zhang/tmp_analysis/tier3/f0_retry")
RUNS = Path("/root/autodl-tmp/haoran.zhang/fc_runs/C3r2_tier3_f0_retry")
SPI, N0, END = 4096 * 24, 44922, 90000
S = RA / "v3fix/progress_samples.tsv"
now = time.time()
rows = {}
for line in (RA / "pids.tsv").read_text().splitlines():
    f = line.split("\t")
    if len(f) >= 6 and f[0].isdigit():
        rows[f[0]] = {"pid": int(f[1]), "gpu": int(f[2]), "t": int(f[5])}
def last_iter(ep):
    p = RUNS / f"ep{ep}" / "metrics.jsonl"
    try:
        with p.open("rb") as fh:
            fh.seek(0, 2); size = fh.tell(); fh.seek(max(0, size - 65536))
            lines = fh.read().decode(errors="ignore").strip().splitlines()
        for l in reversed(lines):
            try:
                return round(json.loads(l)["step"] / SPI)
            except (ValueError, KeyError):
                continue
    except OSError:
        pass
    return None
hist = {}
if S.is_file():
    for l in S.read_text().splitlines():
        ep, t, it = l.split("\t"); hist.setdefault(ep, []).append((float(t), int(it)))
out = []
with S.open("a") as fh:
    for ep, r in sorted(rows.items()):
        alive = Path(f"/proc/{r['pid']}").exists()
        it = last_iter(ep)
        if it is not None:
            fh.write(f"{ep}\t{now:.0f}\t{it}\n"); hist.setdefault(ep, []).append((now, it))
        h = [x for x in hist.get(ep, []) if x[1] > N0]
        rate = None
        if len(h) >= 2:
            old = [x for x in h if now - x[0] >= 600]
            a = old[-1] if old else h[0]
            if now - a[0] > 120 and it is not None and it > a[1]:
                rate = (it - a[1]) / ((now - a[0]) / 3600)
        eta = (END - it) / rate if rate and it else None
        fin = (RUNS / f"ep{ep}" / "final.json").is_file()
        out.append(f"ep{ep} gpu{r['gpu']} pid {r['pid']} {'alive' if alive else 'GONE'} age {int((now - r['t']) / 60)}m it {it} "
                   f"(+{(it or N0) - N0} past resume) rate {f'{rate:.0f} it/h' if rate else '-'} eta {f'{eta:.1f} h' if eta else '-'}"
                   f"{' final.json' if fin else ''}{' CRASHED' if (RA / f'CRASHED_ep{ep}').exists() else ''}")
print("\n".join(out))
