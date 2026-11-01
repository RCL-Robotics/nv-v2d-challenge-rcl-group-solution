"""make_v3.py: build t3retry_v3.py and launch_v3.sh as NEW files from the v2 t3retry.py / launch.sh (which stay unchanged
and keep matching RETRY_FILES.sha256). Anchored replacements, each must match exactly once."""
import os, sys
from pathlib import Path
RA = Path("/root/autodl-tmp/haoran.zhang/tmp_analysis/tier3/f0_retry")
SRC, DST = RA / "t3retry.py", RA / "t3retry_v3.py"
src = SRC.read_text()
R = []
R.append(('''"""t3retry.py: HELD retry queue for failing Tier-3 f0 clips (tmp_analysis/tier3/f0_retry/). BUILT 2026-10-26, NOT LAUNCHED.
''', '''"""t3retry_v3.py: Tier-3 retry queue (C90 continuations), v3 of t3retry.py (tmp_analysis/tier3/f0_retry/).

v3 (2026-10-31, governed by PREREG_v3.md; built by v3fix/make_v3.py from the unchanged v2 t3retry.py). Changes vs v2:
  (1) hashes: PREREG_v3.md / PREREG_v3.sha256 / RETRY_FILES_v3.sha256 (the v2 files stay unchanged and keep matching
      their own pins; RETRY_FILES_v3.sha256 pins them too);
  (2) GPU allowlist of this wave: ALLOWED_GPUS = (0, 1, 2, 5, 6, 7) (GPU 3 removed; GPU 4 never). The same tuple is set on
      the read-only t3f0_helper module object after import (scripts/t3f0_helper.py itself is NOT edited: it is pinned by
      tmp_analysis/tier3/f0/build_files_f0.sha256); plan() reads it at call time. The per-GPU cap stays 3 trainings of
      anybody before + this start <= 3 (asserted equal to the helper's MAX_TRAIN_PER_GPU).
  Nothing else changes: candidate rule, recipe rules, cfgcheck, LATE gate, start spacing, episode order, post, acceptance,
  chosen_run.
v2 header follows.

t3retry.py: HELD retry queue for failing Tier-3 f0 clips (tmp_analysis/tier3/f0_retry/). BUILT 2026-10-26, NOT LAUNCHED.
'''))
R.append(('''PREREG, PREREG_SHA, FILES_SHA = RA / "PREREG_retry.md", RA / "PREREG_retry.sha256", RA / "RETRY_FILES.sha256"
''', '''PREREG, PREREG_SHA, FILES_SHA = RA / "PREREG_v3.md", RA / "PREREG_v3.sha256", RA / "RETRY_FILES_v3.sha256"  # v3
'''))
R.append(('''ALLOWED_GPUS = (0, 1, 2, 3, 5, 6, 7)  # never GPU 4
''', '''ALLOWED_GPUS = (0, 1, 2, 5, 6, 7)  # v3 (PREREG_v3, this wave): GPUs 3 and 4 never
MAX_TRAIN_PER_GPU = 3  # v3: trainings of anybody per GPU after a start (the t3f0 fill-rule value; asserted against the helper)
'''))
R.append(('''        spec.loader.exec_module(mod)
''', '''        spec.loader.exec_module(mod)
        # v3: apply this wave's allowlist to the module object only (the helper file is not edited); plan() reads it at call
        # time. The cap must be the helper's own value.
        if int(mod.MAX_TRAIN_PER_GPU) != MAX_TRAIN_PER_GPU:
            raise RuntimeError(f"t3f0_helper MAX_TRAIN_PER_GPU {mod.MAX_TRAIN_PER_GPU} != {MAX_TRAIN_PER_GPU}")
        mod.ALLOWED_GPUS = tuple(ALLOWED_GPUS)
'''))
R.append(('''    return got == want, f"sha256(PREREG_retry.md) {str(got)[:16]}... {'==' if got == want else '!='} {want[:16]}..."
''', '''    return got == want, f"sha256({PREREG.name}) {str(got)[:16]}... {'==' if got == want else '!='} {want[:16]}..."
'''))
R.append(('''        return False, "RETRY_GO does not name the current PREREG_retry sha256 (line 'prereg_sha256 <sha>')"
''', '''        return False, f"RETRY_GO does not name the current {PREREG.name} sha256 (line 'prereg_sha256 <sha>')"
'''))
R.append(('''        raise RuntimeError(f"gpu {gpu} not allowed (never 4)")
''', '''        raise RuntimeError(f"gpu {gpu} not allowed (v3 allowlist {ALLOWED_GPUS})")
'''))
for old, new in R:
    n = src.count(old)
    if n != 1:
        sys.exit(f"anchor count {n} != 1 for: {old[:90]!r}")
    src = src.replace(old, new)
compile(src, str(DST), "exec")
if DST.exists():
    sys.exit(f"{DST} exists: refusing to overwrite")
tmp = RA / f".{DST.name}.tmp"
tmp.write_text(src)
os.chmod(tmp, 0o444)
os.replace(tmp, DST)
print("wrote", DST, len(R), "replacements")
L = (RA / "launch.sh").read_text()
L2 = L.replace("t3retry.py run", "t3retry_v3.py run").replace("$RA/t3retry.py", "$RA/t3retry_v3.py")
L2 = L2.replace("env -u PYTHONPATH CUDA_VISIBLE_DEVICES=", "env -u PYTHONPATH -u CUDA_MPS_PIPE_DIRECTORY -u CUDA_MPS_LOG_DIRECTORY CUDA_VISIBLE_DEVICES=")
L2 = L2.replace("# launch.sh: start the HELD Tier-3 retry driver (tmp_analysis/tier3/f0_retry/t3retry.py run).",
                "# launch_v3.sh: start the Tier-3 retry driver v3 (tmp_analysis/tier3/f0_retry/t3retry_v3.py run; PREREG_v3.md).")
L2 = L2.replace("(never GPU 4, < 3 trainings per GPU)", "(v3: GPUs 0 1 2 5 6 7 only, < 3 trainings of anybody per GPU before a start)")
L2 = L2.replace('echo "t3retry driver started', 'echo "t3retry_v3 driver started')
assert L2.count("t3retry_v3.py") >= 2 and "CUDA_MPS_PIPE_DIRECTORY" in L2, L2
LD = RA / "launch_v3.sh"
if LD.exists():
    sys.exit(f"{LD} exists")
tmp = RA / ".launch_v3.sh.tmp"
tmp.write_text(L2)
os.chmod(tmp, 0o555)
os.replace(tmp, LD)
print("wrote", LD)
