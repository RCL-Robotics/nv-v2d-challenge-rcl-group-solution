#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Score a packed Track-2 Tier-1 CSV with the kit's code-only Kaggle metrics (metric_code/track_2/{AUC,SP-SR,MP-SR,
MPPE}.py) against the clean reference built from the challenge dataset, as the leaderboard does.

Same solution build and metric calls as the server scorer t1best20_score.py; the kit and dataset roots are
arguments here. Run with the kit's Python 3.11 venv and -B (nothing is written into the kit):
  <kit python> -B score_csv.py --csv FILE.csv --out OUT.json [--episodes all|28|"3 24"] [--kit DIR] [--data DIR]
--episodes all: overall float32 and float64 scores (the whole 30-episode solution) plus per-episode float32 scores.
Otherwise: per-episode float32 scores of the listed episodes and their mean (no overall).
--kit / --data default to $V2D_KIT / $V2D_DATA (the dataset root that contains track_2/).
"""

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import base64  # noqa: E402
import importlib.util  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

METRICS = ["AUC", "SP-SR", "MP-SR", "MPPE"]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--episodes", default="all")
    p.add_argument("--kit", type=Path, default=os.environ.get("V2D_KIT"))
    p.add_argument("--data", type=Path, default=os.environ.get("V2D_DATA"))
    a = p.parse_args()
    if a.kit is None or a.data is None:
        p.error("--kit and --data (or V2D_KIT / V2D_DATA) are required")
    kit, data = a.kit.absolute(), a.data.absolute()
    sys.path.insert(0, str(kit))
    import numpy as np
    import pandas as pd
    from v2dlb import object_pose as op
    from v2dlb.dataset import episode_indices, participant_reference_root
    from v2dlb.trajectory import read_track
    from v2dlb.vertices import mesh_path_for, sample_vertices

    def build_solution(dtype):
        config = json.loads((kit / "config/tracks.json").read_text())
        spec = config["tracks"]["track_2_tier1"]
        root = participant_reference_root(data, spec)
        eps = episode_indices(root)
        ref = {t.episode_index: t for t in read_track(root, eps)}
        rows = []
        for e in eps:
            t = ref[e]
            for step, frame in enumerate(t.frame_index):
                for slot, name in enumerate(t.object_names):
                    rows.append((f"track_2/e{e:06d}/w000/f{int(frame):06d}/o{slot}", e, 0, int(frame), slot, name,
                                 *t.pose_xyzw[step, slot]))
        sol = pd.DataFrame(rows, columns=["row_id", "episode_index", "world_index", "frame_index", "object_slot",
                                          "object_name", *op._REF_COLUMNS])
        names = sorted(sol["object_name"].unique())
        verts = {n: sample_vertices(mesh_path_for(root / "mesh", n)).astype(dtype) for n in names}
        buf = io.BytesIO()
        np.savez_compressed(buf, **verts)
        return sol, base64.b64encode(buf.getvalue()).decode("ascii"), str(root)

    def with_blob(sol, blob):
        s = sol.copy()
        s["auc_vertex_samples_b64"] = ""
        s.iloc[0, s.columns.get_loc("auc_vertex_samples_b64")] = blob
        return s

    mods = {m: load_module(f"metric_{m.replace('-', '_')}", kit / f"metric_code/track_2/{m}.py") for m in METRICS}
    sub = pd.read_csv(a.csv)
    result = {"csv": str(a.csv), "metric_files": {m: str(kit / f"metric_code/track_2/{m}.py") for m in METRICS}}
    full = a.episodes.strip() == "all"
    for dtype in ("float32", "float64") if full else ("float32",):
        sol, blob, root = build_solution(dtype)
        result["reference_root"] = root
        if full:
            s = with_blob(sol, blob)
            result[f"overall_{dtype}"] = {m: float(mods[m].score(s.copy(), sub.copy(), "row_id")) for m in METRICS}
        if dtype == "float32":
            wanted = None if full else {int(x) for x in a.episodes.replace(",", " ").split()}
            pe = {}
            for e, g in sol.groupby("episode_index"):
                if wanted is not None and int(e) not in wanted:
                    continue
                s = with_blob(g.reset_index(drop=True), blob)
                pe[str(int(e))] = {m: float(mods[m].score(s.copy(), sub.copy(), "row_id")) for m in METRICS}
            result["per_episode_float32"] = pe
            result["mean_of_per_episode_float32"] = (
                {m: float(np.mean([v[m] for v in pe.values()])) for m in METRICS} if pe else {})
    tmp = a.out.with_name(f".{a.out.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(result, indent=1))
    os.replace(tmp, a.out)
    print(json.dumps({k: result[k] for k in result if k.startswith(("overall", "mean_of"))}), flush=True)


if __name__ == "__main__":
    main()
