import os, sys
from pathlib import Path
RA = Path("/root/autodl-tmp/haoran.zhang/tmp_analysis/tier3/f0_retry")
def patch(F, R):
    src = F.read_text()
    for old, new in R:
        n = src.count(old)
        if n != 1:
            sys.exit(f"{F.name}: anchor count {n} != 1 for: {old[:90]!r}")
        src = src.replace(old, new)
    if F.suffix == ".py":
        compile(src, str(F), "exec")
    os.chmod(F, 0o644); F.write_text(src); os.chmod(F, 0o444)
    print("patched", F.name, len(R))
patch(RA / "t3retry.py", [("""        if str(p) not in seen and p.is_file() and d.resolve() != OWN_EVALLOCK.resolve():
            seen.add(str(p))
            out.append(p)""", """        real = os.path.realpath(p)  # v2: dedup by real file (queue_t1fix*/t2fix evallock are symlinks to queue_runin's;
        if real not in seen and p.is_file() and d.resolve() != OWN_EVALLOCK.resolve():  # a 2nd flock of it would self-conflict)
            seen.add(real)
            out.append(p)""")])
patch(RA / "PREREG_retry.md", [("""- section 6 (the route into the packed file);
- the files list;
- disclosure 6.
""", """- section 6 (the route into the packed file);
- the files list;
- disclosure 6;
- a lane-lock bug fix in section 5. The lock list is now deduplicated by real file. In v1, the
  `fc_runs/queue_t1fix*` / `queue_t2fix` evallock symlinks to `queue_runin/evallock` made the same gpuN.lock appear up
  to 7 times. Our own second flock of it then always failed, so a post on GPUs 0, 1, 2, 5, 6 or 7 would have waited
  forever.
""")])
