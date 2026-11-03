"""waitfor.py N MAXS: block (read-only) until pids.tsv has >= N rows, or a CRASHED/LATE marker appears, or the driver
dies, or MAXS seconds pass. Prints the reason."""
import os, sys, time
from pathlib import Path
RA = Path("/root/autodl-tmp/haoran.zhang/tmp_analysis/tier3/f0_retry")
n, maxs = int(sys.argv[1]), float(sys.argv[2])
t0 = time.time()
while True:
    rows = [l for l in (RA / "pids.tsv").read_text().splitlines() if l[:2].isdigit()]
    bad = [p.name for p in RA.iterdir() if p.name.startswith(("CRASHED_ep", "LATE_ep"))]
    try:
        alive = Path(f"/proc/{int((RA / 'driver.pid').read_text())}").exists()
    except (OSError, ValueError):
        alive = False
    if len(rows) >= n or bad or not alive or time.time() - t0 > maxs:
        print(time.strftime("%T"), f"launched {len(rows)}", f"markers {bad}", f"driver {'alive' if alive else 'DEAD'}",
              f"waited {int(time.time() - t0)} s")
        break
    time.sleep(15)
