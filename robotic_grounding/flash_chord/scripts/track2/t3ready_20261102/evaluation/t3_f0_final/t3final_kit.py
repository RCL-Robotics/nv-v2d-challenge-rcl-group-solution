"""t3final_kit.py: kit-side checks for fc_runs/submissions/t3_f0_final (kit env only: conda_envs/v2dkit python -I -B).

  validate --csv F --out J [--track track_2_tier3] [--expect E.json] [--partial]
      The kit uploader's own validate_submission() (submit/v2d_submit.py, called directly: offline, no credentials,
      Kaggle never contacted) + provenance: uniform evaluator digests / packer / uploader / commit URL, one checkpoint
      digest per episode equal to the expected eval file, header, pose-columns sha256 (to prove a later re-pack with the
      real commit URL carries the same poses). With --partial the uploader check is expected to fail (partial row set);
      'ok' then means "no provenance problem". Writes no clean number.
  score --csv F --out J [--track track_2_tier3] [--episodes all|"0 3 7"]
      The official metric_code/track_2/{AUC,SP-SR,MP-SR,MPPE}.py score() against the clean multi-view reference (the
      solution construction of fc_runs/submissions/t1_best20_20261026/t1best20_score.py). REPORT ONLY: --out must be a
      *_LOCAL_NOT_FOR_UPLOAD file; nothing is printed except the episode count.
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import base64  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

ME = Path("/root/autodl-tmp/haoran.zhang")
KIT = ME / "v2d_submission_kit_src/v2d_submission_kit"
DATA = ME / "data/video_to_data_challenge"
sys.path.insert(0, str(KIT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from v2dlb import object_pose as op  # noqa: E402
from v2dlb import report_schema as rs  # noqa: E402
from v2dlb.dataset import episode_indices, participant_reference_root  # noqa: E402
from v2dlb.trajectory import read_track  # noqa: E402
from v2dlb.vertices import mesh_path_for, sample_vertices  # noqa: E402

METRICS = ["AUC", "SP-SR", "MP-SR", "MPPE"]


def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_json(path: Path, data) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=1, default=str) + "\n")
    os.replace(tmp, path)


def pose_columns_sha(csvp: Path, npose: int) -> tuple[str, list[str]]:
    h = hashlib.sha256()
    header = None
    with csvp.open() as fp:
        for i, line in enumerate(fp):
            parts = line.rstrip("\n").split(",")
            if i == 0:
                header = parts
            h.update((",".join(parts[: 1 + npose]) + "\n").encode())
    return h.hexdigest(), header or []


def cmd_validate(a) -> int:
    config = json.loads((KIT / "config/tracks.json").read_text())
    up = load_module("kit_v2d_submit", KIT / "submit/v2d_submit.py")
    csvp = Path(a.csv)
    res = {"csv": str(csvp), "csv_sha256": sha(csvp), "track": a.track, "partial": a.partial, "problems": []}
    try:
        frame = up.validate_submission(csvp, a.track, config)
        res["uploader_validate_submission"] = "pass"
    except SystemExit as err:
        res["uploader_validate_submission"] = f"FAIL: {str(err)[:400]}"
        frame = pd.read_csv(csvp)
        if not a.partial:
            res["problems"].append("uploader validate_submission failed")
    res["rows"] = int(len(frame))
    uniform = {c: sorted(frame[c].astype(str).str.strip().unique()) for c in rs.UNIFORM_PROVENANCE_COLUMNS}
    res["uniform_columns"] = uniform
    for c, v in uniform.items():
        if len(v) != 1:
            res["problems"].append(f"{c} not uniform: {v[:3]}")
    eps = frame[rs.ROW_ID_COLUMN].astype(str).str.split("/").str[1].str[1:].astype(int)
    per_ep = frame.groupby(eps)["checkpoint_sha256"].agg(lambda s: sorted(s.astype(str).str.strip().unique()))
    res["episodes"] = sorted(int(e) for e in per_ep.index)
    res["checkpoints"] = {str(int(e)): v for e, v in per_ep.items()}
    if a.expect:
        exp = json.loads(Path(a.expect).read_text())
        want = {int(e): s for e, s in exp["episodes"].items()}
        if sorted(want) != res["episodes"]:
            res["problems"].append(f"episode set {res['episodes']} != expected {sorted(want)}")
        for e, v in per_ep.items():
            if len(v) != 1 or v[0] != want.get(int(e)):
                res["problems"].append(f"episode {int(e)}: checkpoint digest {v} != expected {str(want.get(int(e)))[:16]}")
        for c in ("code_commit_url", "eval_script_sha256", "eval_code_sha256", "packer_script_sha256", "uploader_script_sha256"):
            if c in exp and uniform.get(c, [None])[0] != exp[c]:
                res["problems"].append(f"{c} {uniform.get(c)} != expected {exp[c]}")
    man = json.loads((KIT / "KIT_MANIFEST.json").read_text())["files"]
    for c, rel in (("packer_script_sha256", "tools/pack_submission.py"), ("uploader_script_sha256", "submit/v2d_submit.py")):
        if uniform.get(c, [None])[0] != man.get(rel):
            res["problems"].append(f"{c} != KIT_MANIFEST {rel}")
    npose = len(op._POSE_COLUMNS)
    res["pose_columns_sha256"], header = pose_columns_sha(csvp, npose)
    if header[: 1 + npose] != [rs.ROW_ID_COLUMN, *op._POSE_COLUMNS]:
        res["problems"].append(f"unexpected header {header[: 1 + npose]}")
    res["ok"] = not res["problems"] and (a.partial or res["uploader_validate_submission"] == "pass")
    write_json(Path(a.out), res)
    print(f"validate: uploader {res['uploader_validate_submission'][:120]}; rows {res['rows']}; episodes "
          f"{len(res['episodes'])}; problems {len(res['problems'])}; ok {res['ok']}")
    return 0 if res["ok"] else 1


def build_solution(track: str, dtype, wanted):
    config = json.loads((KIT / "config/tracks.json").read_text())
    spec = config["tracks"][track]
    root = participant_reference_root(DATA, spec)
    eps = episode_indices(root) if wanted is None else sorted(wanted)
    ref = {t.episode_index: t for t in read_track(root, eps)}
    tn = rs.track_number(track)
    rows = []
    for e in eps:
        t = ref[e]
        for step, frame in enumerate(t.frame_index):
            for slot, name in enumerate(t.object_names):
                rows.append((f"track_{tn}/e{e:06d}/w000/f{int(frame):06d}/o{slot}", e, 0, int(frame), slot, name,
                             *t.pose_xyzw[step, slot]))
    sol = pd.DataFrame(rows, columns=["row_id", "episode_index", "world_index", "frame_index", "object_slot",
                                      "object_name", *op._REF_COLUMNS])
    names = sorted(sol["object_name"].unique())
    verts = {n: sample_vertices(mesh_path_for(root / "mesh", n)).astype(dtype) for n in names}
    buf = io.BytesIO()
    np.savez_compressed(buf, **verts)
    blob = base64.b64encode(buf.getvalue()).decode("ascii")
    return sol, blob, str(root), spec


def with_blob(sol, blob):
    s = sol.copy()
    s["auc_vertex_samples_b64"] = ""
    s.iloc[0, s.columns.get_loc("auc_vertex_samples_b64")] = blob
    return s


def cmd_score(a) -> int:
    out = Path(a.out)
    if "LOCAL_NOT_FOR_UPLOAD" not in out.name:
        print("score: --out must be a *_LOCAL_NOT_FOR_UPLOAD file (clean numbers are report-only)")
        return 2
    mods = {m: load_module(f"metric_{m.replace('-', '_')}", KIT / f"metric_code/track_2/{m}.py") for m in METRICS}
    sub = pd.read_csv(a.csv)
    full = a.episodes.strip() == "all"
    wanted = None if full else {int(x) for x in a.episodes.replace(",", " ").split()}
    result = {"csv": a.csv, "csv_sha256": sha(a.csv), "track": a.track,
              "metric_files": {m: str(KIT / f"metric_code/track_2/{m}.py") for m in METRICS},
              "metric_sha256": {m: sha(KIT / f"metric_code/track_2/{m}.py") for m in METRICS},
              "note": "LOCAL_NOT_FOR_UPLOAD: clean reference; report only, never used for a Tier-3 decision"}  # fmt: skip
    for dtype in ("float32", "float64") if full else ("float32",):
        sol, blob, root, spec = build_solution(a.track, dtype, wanted)
        result["reference_root"] = root
        result["align_initial_pose"] = bool(spec.get("align_initial_pose", False))
        if full:
            s = with_blob(sol, blob)
            result[f"overall_{dtype}"] = {m: float(mods[m].score(s.copy(), sub.copy(), "row_id")) for m in METRICS}
        if dtype == "float32":
            pe = {}
            for e, g in sol.groupby("episode_index"):
                s = with_blob(g.reset_index(drop=True), blob)
                pe[str(int(e))] = {m: float(mods[m].score(s.copy(), sub.copy(), "row_id")) for m in METRICS}
            result["per_episode_float32"] = pe
            result["mean_of_per_episode_float32"] = {m: float(np.mean([v[m] for v in pe.values()])) for m in METRICS} if pe else {}
    write_json(out, result)
    print(f"score: {len(result.get('per_episode_float32', {}))} episodes scored -> {out.name} (numbers not printed)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--csv", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--track", default="track_2_tier3")
    v.add_argument("--expect")
    v.add_argument("--partial", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("--csv", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--track", default="track_2_tier3")
    s.add_argument("--episodes", default="all")
    a = p.parse_args()
    if not sys.flags.dont_write_bytecode or not sys.flags.isolated:
        print("run with the kit python -I -B")
        return 2
    return {"validate": cmd_validate, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
