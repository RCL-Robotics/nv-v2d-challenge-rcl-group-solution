#!/usr/bin/env python3
"""k3full_gates.py (NOISY-ONLY, CPU only): the per-episode INPUT RULE of the full K3fix arm (PREREG_v3): the four
pre-training gates G1-G4 of the K3fix pilot on ALL 30 Tier-2 episodes, computed on the K3fix track that
data/human_motion_data_tier2_k3fix was built from.
Run:  CUDA_VISIBLE_DEVICES= nice -n 19 conda_envs/v2d/bin/python -I -B scripts/k3full_gates.py gates [--no-write]

Gate code: scripts/k3fix_build.py gate_work, imported read-only (sha256 pinned 364d27ed, the file pinned in
k3fix_pilot/build_files_v2.sha256); same Tier-2-only constants (envelope 49.23 deg / 4.62 cm):
  G1 envelope     no visible frame with |K3fix - raw| rotation > 49.23 deg or position > 4.62 cm
  G2 penetration  no visible frame where the K3fix object is > 4.62 cm deeper in the support stage / below the floor
                  (G1 training world) than the raw object
  G3 rest         no rest run flagged by the calibrated random-walk test still has zero angular velocity forced
  G4 contact      among frames where the raw object is < 3 cm from a hand point, the share where K3fix is > 3 cm farther
                  than raw <= 5 %, evaluated with >= 50 such frames (fewer: not evaluable, passes = exempt)
  episode PASS iff G1-G4 all pass -> K3fix candidate; FAIL -> the raw C3r2_tier2 run is used for that episode.
Checks (all must hold, else exit 1 and nothing is decided):
  - k3fix_build.py sha256 = pinned; build/k3fix_tracks.npz sha256 = tracks.json = data_build.json = the K3fix root's
    READY text (so the gated track is the track the data root was built from);
  - per episode: K3fix-root parquet sha256 = READY entry = data_build.json root_sha256; unchanged episodes byte-identical
    to data/human_motion_data_tier2_k3 (changed = {3, 4, 9, 17} exactly);
  - replication: the K3fix results for ep03/04/09/17 equal build/gates.json, for ep07/18 build/gates_v2.json; the
    frozen-K3 G4 numbers equal the reviewer's all-30 table (k3_rootcause/review/r4_contact_gate_all.json).
Reads no clean data (no Tier-1 path, no *LOCAL_NOT_FOR_UPLOAD* file). Writes only tmp_analysis/tier2/k3full/build/
(gates_all30.json, gates_all30.txt).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

ME = Path("/root/autodl-tmp/haoran.zhang")
OUTDIR = ME / "tmp_analysis/tier2/k3full/build"
OUT, TXT = OUTDIR / "gates_all30.json", OUTDIR / "gates_all30.txt"
PB = ME / "tmp_analysis/tier2/k3fix_pilot/build"
BUILD_PY = ME / "scripts/k3fix_build.py"
BUILD_PY_SHA = "364d27ed1b0e0a250456ff2fa42b130c7f73bfb4755f88c75237dc79148dff7b"
FIXROOT, K3ROOT, RAWROOT = (ME / f"data/{x}" for x in ("human_motion_data_tier2_k3fix", "human_motion_data_tier2_k3",
                                                        "human_motion_data_tier2"))
READY_SHA = "1233398f02e61b51fb9bc13072b1607ec3964556cc20828ac56f571b1640dac8"
R4 = ME / "tmp_analysis/tier2/k3_rootcause/review/r4_contact_gate_all.json"
CHANGED = [3, 4, 9, 17]
GATES = ("G1_envelope", "G2_penetration", "G3_rest", "G4_contact")


def sha256(path) -> str | None:
    try:
        h = hashlib.sha256()
        with Path(path).open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def load_build():
    got = sha256(BUILD_PY)
    if got != BUILD_PY_SHA:
        raise SystemExit(f"{BUILD_PY} sha256 {got} != pinned {BUILD_PY_SHA}")
    spec = importlib.util.spec_from_file_location("k3fix_build", BUILD_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["k3fix_build"] = mod
    spec.loader.exec_module(mod)
    return mod


def ready_table() -> dict[str, str]:
    out = {}
    for line in (FIXROOT / "READY").read_text().splitlines()[1:]:
        parts = line.strip().split("  ")
        if len(parts) == 2 and len(parts[0]) == 64:
            out[parts[1]] = parts[0]
    return out


def same(a, b, tol=1e-9) -> bool:
    """Recursive equality of two gate records (floats within tol; 'seconds' ignored)."""
    if isinstance(a, dict) and isinstance(b, dict):
        keys = (set(a) | set(b)) - {"seconds"}
        return all(k in a and k in b and same(a[k], b[k], tol) for k in keys)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a is None or b is None:
            return a is b
        if math.isnan(float(a)) and math.isnan(float(b)):
            return True
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))
    return a == b


def jsonable(x):
    return json.loads(json.dumps(x, default=float))


def main(write: bool) -> int:
    t0 = time.time()
    kb = load_build()
    checks: list[tuple[str, bool, str]] = []

    def add(name, ok, detail):
        checks.append((name, bool(ok), detail))

    tr = json.loads((PB / "tracks.json").read_text())
    db = json.loads((PB / "data_build.json").read_text())
    npz_sha = sha256(PB / "k3fix_tracks.npz")
    ready_text = (FIXROOT / "READY").read_text()
    add("build_code", True, f"k3fix_build.py sha256 {BUILD_PY_SHA[:16]}... (pinned)")
    add("ready_marker", sha256(FIXROOT / "READY") == READY_SHA, f"{FIXROOT}/READY sha256 {str(sha256(FIXROOT / 'READY'))[:16]}... (pinned {READY_SHA[:8]})")
    add("tracks_npz_same_everywhere", npz_sha == tr.get("k3fix_tracks_npz_sha256") == db.get("k3fix_tracks_npz_sha256")
        and npz_sha[:16] in ready_text, f"k3fix_tracks.npz {str(npz_sha)[:16]}... = tracks.json = data_build.json = READY text")
    add("tracks_changed", tr.get("changed_episodes") == CHANGED and tr.get("all_ok") is True,
        f"tracks.json changed {tr.get('changed_episodes')} (expected {CHANGED}), all_ok {tr.get('all_ok')}")
    ready = ready_table()
    add("ready_30", len(ready) == 30, f"READY lists {len(ready)} parquets")
    g1old = json.loads((PB / "gates.json").read_text())
    g2old = json.loads((PB / "gates_v2.json").read_text())
    r4 = {int(r[0]): r for r in json.loads(R4.read_text())}
    res, root = {}, {}
    for ep in range(30):
        t1 = time.time()
        r = kb.gate_work(ep)
        r["seconds"] = round(time.time() - t1, 1)
        res[ep] = r
        seq = r["sequence_id"]
        rel = f"whole_body/soma/sequence_id={seq}/robot_name=g1/data.parquet"
        s_fix, s_k3, s_raw = sha256(FIXROOT / rel), sha256(K3ROOT / rel), sha256(RAWROOT / rel)
        de = db["episodes"][f"{ep:02d}"]
        changed = ep in CHANGED
        ok = (s_fix is not None and s_fix == ready.get(rel) == de.get("root_sha256") and de.get("ok") is True
              and de.get("changed") is changed and s_fix != s_raw and de.get("sequence_id") == seq
              and ((s_fix != s_k3) if changed else (s_fix == s_k3)))
        root[ep] = {"sequence_id": seq, "parquet_rel": rel, "k3fix_root_sha256": s_fix, "k3_root_sha256": s_k3,
                    "raw_tier2_sha256": s_raw, "changed_by_k3fix": changed, "byte_identical_to_k3_root": s_fix == s_k3,
                    "ready_entry_matches": s_fix == ready.get(rel), "data_build_matches": s_fix == de.get("root_sha256"),
                    "ok": bool(ok)}
        add(f"ep{ep:02d}_data_root", ok, f"K3fix root parquet {str(s_fix)[:12]} = READY = data_build; "
            f"{'changed: differs from K3 root' if changed else 'unchanged: = K3 root'}; != raw")
        # frozen-K3 G4 vs the reviewer's all-30 table
        fz = r["K3_frozen"]["G4_contact"]
        rr = r4[ep]
        okr = fz["n_raw_contact_frames"] == rr[2] and abs((fz["share_farther_gt3cm"] or 0.0) - rr[4]) <= 1e-12
        add(f"ep{ep:02d}_r4_replication", okr, f"frozen-K3 G4 n {fz['n_raw_contact_frames']}/{rr[2]} share "
            f"{fz['share_farther_gt3cm']}/{rr[4]}")
        if ep in (3, 4, 9, 17):
            old = g1old["clips"][f"{ep:02d}"]
            add(f"ep{ep:02d}_gates_json_replication", same(jsonable(r["K3fix"]), old["K3fix"]), "K3fix record = build/gates.json")
        if ep in (7, 18):
            old = g2old["clips"][f"{ep:02d}"]
            add(f"ep{ep:02d}_gates_v2_replication", same(jsonable(r["K3fix"]), old["K3fix"]), "K3fix record = build/gates_v2.json")
        print(f"ep{ep:02d} gated in {r['seconds']} s", flush=True)
    rows, passing, failing = [], [], []
    for ep in range(30):
        k = res[ep]["K3fix"]
        g1, g2, g3, g4 = (k[g] for g in GATES)
        exempt = not g4["evaluable"]
        rows.append({"episode": ep, "object": res[ep]["object"], "G1": g1["pass"], "G2": g2["pass"], "G3": g3["pass"],
                     "G4": g4["pass"], "G4_exempt": exempt, "all_pass": k["all_pass"],
                     "g1_rot_max_deg": g1["max_rot_dev_deg"], "g1_pos_max_cm": g1["max_pos_dev_cm"], "g1_frames_beyond": g1["frames_beyond"],
                     "g2_frames_deeper": g2["frames_deeper_than_raw_by_gt_env"], "g2_max_excess_cm": g2["max_excess_depth_cm"],
                     "g3_flagged_runs": g3["flagged_runs"], "g3_w0_forced": g3["w0_frames_forced_in_flagged"],
                     "g4_n_contact": g4["n_raw_contact_frames"], "g4_share": g4["share_farther_gt3cm"]})
        (passing if k["all_pass"] else failing).append(ep)
    bad = [c for c in checks if not c[1]]
    lines = [f"K3fix full arm - noisy-only input rule G1-G4 on all 30 Tier-2 episodes (envelope {kb.ENV_R:.2f} deg / {kb.ENV_P:.2f} cm; "
             f"G4 share <= {kb.MAX_SHARE} with >= {kb.MIN_CONTACT} contact frames)",
             f"checks {len(checks) - len(bad)}/{len(checks)} ok" + (f"; FAILED: {' '.join(c[0] for c in bad)}" if bad else ""), ""]
    lines.append("ep  object               G1   G2   G3   G4         rot_max  pos_max  beyond  deeper  excess_cm  G4 n/share      -> input")
    for x in rows:
        g4t = "exempt" if x["G4_exempt"] else ("pass" if x["G4"] else "FAIL")
        share = "-" if x["g4_share"] is None else f"{x['g4_share']:.4f}"
        lines.append(f"{x['episode']:02d}  {x['object']:<20} {'pass' if x['G1'] else 'FAIL':<4} {'pass' if x['G2'] else 'FAIL':<4} "
                     f"{'pass' if x['G3'] else 'FAIL':<4} {g4t:<10} {x['g1_rot_max_deg']:7.1f}  {x['g1_pos_max_cm']:7.2f}  {x['g1_frames_beyond']:6d}  "
                     f"{x['g2_frames_deeper']:6d}  {x['g2_max_excess_cm']:9.2f}  {x['g4_n_contact']:4d}/{share:<8} -> "
                     f"{'K3fix candidate' if x['all_pass'] else 'RAW (gate fail)'}")
    lines += ["", f"PASS (K3fix candidates, {len(passing)}): {' '.join(f'{e:02d}' for e in passing)}",
              f"FAIL (raw fallback, {len(failing)}): {' '.join(f'{e:02d}' for e in failing)}",
              f"G4 exempt (< {kb.MIN_CONTACT} contact frames): " + " ".join(f"{x['episode']:02d}" for x in rows if x["G4_exempt"])]
    for name, ok, detail in checks:
        if not ok:
            lines.append(f"CHECK FAIL {name}: {detail}")
    text = "\n".join(lines) + "\n"
    print(text, end="")
    out = {"written": time.strftime("%F %T"), "noisy_only": True, "gate_code": str(BUILD_PY), "gate_code_sha256": BUILD_PY_SHA,
           "k3fix_tracks_npz_sha256": npz_sha, "k3fix_root": str(FIXROOT), "k3fix_root_ready_sha256": sha256(FIXROOT / "READY"),
           "envelope_rot_deg": kb.ENV_R, "envelope_pos_cm": kb.ENV_P, "g4_max_share": kb.MAX_SHARE, "g4_min_contact": kb.MIN_CONTACT,
           "rule": "episode PASS iff G1-G4 all pass (G4 exempt with < 50 raw-contact frames) -> K3fix candidate; FAIL -> raw C3r2_tier2 run",
           "checks": [{"name": n, "ok": o, "detail": d} for n, o, d in checks], "checks_ok": not bad,
           "table": rows, "pass_episodes": passing, "fail_episodes": failing,
           "g4_exempt_episodes": [x["episode"] for x in rows if x["G4_exempt"]], "data_root": {f"{e:02d}": v for e, v in root.items()},
           "clips": {f"{e:02d}": jsonable(r) for e, r in res.items()}, "seconds": round(time.time() - t0, 1)}
    if write:
        OUTDIR.mkdir(parents=True, exist_ok=True)
        for path, body in ((OUT, json.dumps(out, indent=1, default=float)), (TXT, text)):
            p = path.resolve()
            assert str(p).startswith(str(OUTDIR.resolve()) + "/")
            tmp = p.with_name(p.name + ".tmp")
            tmp.write_text(body)
            tmp.replace(p)
        print(f"wrote {OUT} and {TXT}")
    return 0 if not bad else 1


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "gates":
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main("--no-write" not in sys.argv[2:]))
