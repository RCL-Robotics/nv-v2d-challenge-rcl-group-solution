import fcntl, os, json
from pathlib import Path
SB = Path("/root/autodl-tmp/haoran.zhang/tmp_analysis/tier3/f0_retry/r3fix/sandbox/flocktest")
(SB / "real").mkdir(parents=True, exist_ok=True)
(SB / "real/gpu0.lock").touch()
link = SB / "link"
if not link.is_symlink():
    link.symlink_to(SB / "real")
def tl(p):
    fd = os.open(p, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB); return fd
    except OSError:
        os.close(fd); return None
a = tl(SB / "real/gpu0.lock"); b = tl(SB / "link/gpu0.lock")
print("flock real:", a is not None, "| same file via symlink in the same process:", b is not None, "(False = self-conflict)")
rep = json.loads(Path("/root/autodl-tmp/haoran.zhang/tmp_analysis/tier3/f0_retry/dryrun/dry_report.json").read_text())
L = rep["lane_lock_files_gpu0"]
print("retry lane_lock_files(0):", len(L), "files;", len({os.path.realpath(p) for p in L}), "distinct real files")
for p in L: print("  ", p, "->", os.path.realpath(p) if os.path.islink(os.path.dirname(p)) else "")
