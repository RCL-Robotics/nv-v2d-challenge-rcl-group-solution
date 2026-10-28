"""Offline checks of one packed Track-2 Tier-1 CSV (no upload, no Kaggle API call, no credentials read):
1. the kit uploader's own validate_submission() (submit/v2d_submit.py), called directly;
2. provenance: uniform evaluator / packer digests, the commit URL on every row equals the expected URL, one
   checkpoint_sha256 per episode equal to the evaluated file of that episode in the trial's set.
Run with the kit env and -B:  <v2dkit python> -B t1topup_validate.py CSV EXPECTED.json COMMIT_URL OUT.json"""
import sys

sys.dont_write_bytecode = True

import importlib.util  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

ME = Path("/root/autodl-tmp/haoran.zhang")
KIT = ME / "v2d_submission_kit_src/v2d_submission_kit"
sys.path.insert(0, str(KIT))
import pandas as pd  # noqa: E402,F401
from v2dlb import report_schema as rs  # noqa: E402

TRACK = "track_2_tier1"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    csv_path, exp_path, url, out_path = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
    expected = {int(k): v for k, v in json.loads(exp_path.read_text()).items()}
    config = json.loads((KIT / "config/tracks.json").read_text())
    out = {"csv": str(csv_path), "track": TRACK, "problems": []}
    up = load("kit_v2d_submit", KIT / "submit/v2d_submit.py")
    frame = up.validate_submission(csv_path, TRACK, config)  # raises SystemExit on a malformed file
    eps = frame[rs.ROW_ID_COLUMN].astype(str).str.split("/").str[1].nunique()
    out["uploader_validate_submission"] = {"ok": True, "rows": int(len(frame)), "episodes": int(eps)}
    prov = {}
    for c in rs.UNIFORM_PROVENANCE_COLUMNS:
        v = sorted(frame[c].astype(str).str.strip().unique().tolist())
        prov[c] = v
        if len(v) != 1:
            out["problems"].append(f"{c} not uniform: {v[:3]}")
    if prov.get("code_commit_url") != [url]:
        out["problems"].append(f"code_commit_url {prov.get('code_commit_url')} != {url}")
    frame["_ep"] = frame[rs.ROW_ID_COLUMN].astype(str).str.split("/").str[1].str[1:].astype(int)
    per_ep = frame.groupby("_ep")["checkpoint_sha256"].agg(lambda s: sorted(set(s.astype(str))))
    bad = [e for e in range(30) if e not in per_ep or per_ep[e] != [expected[e]]]
    if len(per_ep) != 30 or bad:
        out["problems"].append(f"checkpoint_sha256 per episode differs from the set's evaluated files for episodes {bad}")
    out["provenance"] = prov
    out["checkpoints_match_set"] = not bad and len(per_ep) == 30
    out["distinct_checkpoints"] = int(frame["checkpoint_sha256"].nunique())
    out["ok"] = not out["problems"]
    out_path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"ok": out["ok"], "rows": out["uploader_validate_submission"]["rows"], "problems": out["problems"]}))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
