"""classify_han_edits.py <repo> <from> <to> <out.tsv>: for every file under hoi_estimation/ MODIFIED between <from> and <to>,
classify each changed line pair. Classes: path (built-in path -> variable / relative path), import-os (added 'os' to an
import line), subset (V2D_EPISODES opt-in), optional-diag (diag_scores.json made optional), doc (markdown), other."""
import re, subprocess, sys
repo, a, b, out = sys.argv[1:5]
def git(*x): return subprocess.run(["git", "-C", repo] + list(x), capture_output=True, text=True, check=True).stdout
files = git("diff", "--name-only", "--diff-filter=M", a, b, "--", "hoi_estimation").split()
PATH = re.compile(r"V2D_ROOT|V2D_RUN|V2D_WORK|V2D_OUT|V2D_LOGS|V2D_DATASET|V2D_WEIGHTS|V2D_SAM2|V2D_CARI4D|V2D_GDINO|V2D_VDA|V2D_DEPTHPRO|V2D_RECON|V2D_SUPPORT_OUT|"
                  r"/mnt/nas|/home/hanzhou|/l/users/han\.zhou|/tmp/claude-|os\.path\.dirname\(os\.path\.abspath\(__file__\)\)|"
                  r"\$\(dirname \"\$0\"\)|\$\(cd \"\$\(dirname|APPTAINER|\$HOME/\.local|CLAUDE_BIN|CODEX_BIN|expanduser|HERE/\.\./\.\.|T2_PY|"
                  r"miniconda3|\{MESH\}|s\[\"mesh\"\]|\bS=sys\.argv\[1\]|\bREPO\b|-B \$H/\.\.")
rows = []
for f in files:
    d = git("diff", "-U0", a, b, "--", f)
    minus = [l[1:] for l in d.splitlines() if l.startswith("-") and not l.startswith("---")]
    plus = [l[1:] for l in d.splitlines() if l.startswith("+") and not l.startswith("+++")]
    cls = set(); other = []
    if f.endswith(".md"):
        cls.add("doc")
    else:
        for l in minus + plus:
            s = l.strip()
            if not s: continue
            if PATH.search(s): cls.add("path")
            elif re.match(r"^import\s", s) and ("os" in re.split(r"[\s,]+", s)): cls.add("import-os")
            elif re.match(r"^import\s", s) and l in minus: cls.add("import-os")
            elif "V2D_EPISODES" in s or s.startswith("for i in EPS") or s == "for i in range(30):": cls.add("subset")
            elif "diag_scores" in s or "diag AUC" in s: cls.add("optional-diag")
            else:
                cls.add("other"); other.append(s[:160])
    rows.append((f, len(minus), len(plus), ",".join(sorted(cls)), " | ".join(other)))
with open(out, "w") as fh:
    fh.write("path\tlines_removed\tlines_added\tclasses\tother_lines\n")
    for r in rows: fh.write("\t".join(map(str, r)) + "\n")
from collections import Counter
print(len(rows), "files;", Counter(r[3] for r in rows).most_common())
for r in rows:
    if "other" in r[3]: print("OTHER", r[0], "::", r[4][:400])
