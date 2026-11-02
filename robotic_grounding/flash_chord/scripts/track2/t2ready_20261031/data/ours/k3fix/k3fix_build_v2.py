#!/usr/bin/env python3
"""k3fix_build_v2.py (NOISY-ONLY, CPU only): the build step the STAGED K3fix pilot (PREREG_v2.md) adds to
scripts/k3fix_build.py: the four noisy-only gates G1-G4 on the two stage-1 clips that K3fix does NOT change,
ep07 iron and ep18 blue_trash_can, plus the identity checks of their training input.
Run:  CUDA_VISIBLE_DEVICES= nice -n 19 conda_envs/v2d/bin/python -I -B scripts/k3fix_build_v2.py gates [--no-write]

Gate code: scripts/k3fix_build.py gate_work (imported read-only, unchanged; sha256 pinned in build_files.sha256), the
same G1-G4 rules and Tier-2-only constants as build/gates.json (envelope 49.23 deg / 4.62 cm). Each gate is also run
on the FROZEN K3 track and compared with the root-cause numbers (k3_rootcause/fixes/gate2.json variant K3,
review/r4_contact_gate_all.json), as gates.json does for ep03/04/09/17.
Identity checks per clip (K3fix leaves these clips unchanged, so their training input is the K3 input):
  - build/tracks.json: changed false, X3 bit-identical, no flagged rest run, no released w=0 frame, ok;
  - K3fix full track vs frozen K3 full track (kalman/eval/k3_fullframe.npz) max |diff| < 1e-12;
  - data/human_motion_data_tier2_k3fix parquet sha256 == data/human_motion_data_tier2_k3 parquet sha256 (byte-identical)
    == READY entry == build/data_build.json root_sha256, and != the raw Tier-2 parquet;
  - frame_refs_k3fix/episode_<ep>.npz sha256 == data_build.json frame_ref_sha256 and the kit packer verdicts recorded
    there (accepted vs frame_refs_k3fix, REJECTED vs the released raw input).
Reads no clean data. Writes only tmp_analysis/tier2/k3fix_pilot/build/gates_v2.json.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

ME = Path("/root/autodl-tmp/haoran.zhang")
K = ME / "tmp_analysis/tier2/k3fix_pilot"
BUILD = K / "build"
OUT = BUILD / "gates_v2.json"
BUILD_PY = ME / "scripts/k3fix_build.py"
BUILD_PY_SHA = "364d27ed1b0e0a250456ff2fa42b130c7f73bfb4755f88c75237dc79148dff7b"   # build_files.sha256
FIXROOT, K3ROOT, RAWROOT = (ME / f"data/{x}" for x in ("human_motion_data_tier2_k3fix", "human_motion_data_tier2_k3", "human_motion_data_tier2"))
RC = ME / "tmp_analysis/tier2/k3_rootcause"
KOUT = ME / "tmp_analysis/tier2/kalman"
CLIPS = (7, 18)


def sha256(path: Path) -> str | None:
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
        raise SystemExit(f"{BUILD_PY} sha256 {got} != build_files.sha256 {BUILD_PY_SHA}")
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


def main(write: bool) -> int:
    kb = load_build()
    tr = json.loads((BUILD / "tracks.json").read_text())
    db = json.loads((BUILD / "data_build.json").read_text())["episodes"]
    if tr.get("k3fix_tracks_npz_sha256") != sha256(BUILD / "k3fix_tracks.npz"):
        raise SystemExit("build/k3fix_tracks.npz changed since tracks.json")
    g2j = json.loads((RC / "fixes/gate2.json").read_text())["gate"]
    r4 = {int(r[0]): r for r in json.loads((RC / "review/r4_contact_gate_all.json").read_text())}
    ff = np.load(KOUT / "eval/k3_fullframe.npz")
    trk = np.load(BUILD / "k3fix_tracks.npz")
    ready = ready_table()
    res, repl, ident, rep_ok, id_ok = {}, {}, {}, True, True
    for ep in CLIPS:
        t0 = time.time()
        r = kb.gate_work(ep)
        seq = r["sequence_id"]
        rel = f"whole_body/soma/sequence_id={seq}/robot_name=g1/data.parquet"
        s_fix, s_k3, s_raw = sha256(FIXROOT / rel), sha256(K3ROOT / rel), sha256(RAWROOT / rel)
        te, de = tr["episodes"][f"{ep:02d}"], db[f"{ep:02d}"]
        fr = K / "frame_refs_k3fix" / f"episode_{ep:06d}.npz"
        d_track = float(np.abs(trk[f"K3fixfull__{ep}"] - ff[f"K3full__{ep}"]).max())
        idc = {"tracks_json_unchanged": te.get("changed") is False and te.get("x3_identical") is True
               and not te.get("flagged_runs_calibrated") and te.get("w0_frames_released") == 0 and te.get("ok") is True,
               "k3fix_vs_frozen_k3_full_track_max_abs": d_track, "track_identical_lt_1e-12": d_track < 1e-12,
               "parquet_sha256_k3fix_root": s_fix, "parquet_sha256_k3_root": s_k3, "parquet_sha256_raw": s_raw,
               "k3fix_parquet_byte_identical_to_k3": s_fix is not None and s_fix == s_k3,
               "k3fix_parquet_differs_from_raw": s_fix is not None and s_fix != s_raw,
               "ready_entry_matches": ready.get(rel) == s_fix, "data_build_root_sha256_matches": de.get("root_sha256") == s_fix
               and de.get("k3_root_sha256") == s_k3 and de.get("raw_tier2_sha256") == s_raw and de.get("changed") is False,
               "frame_ref_sha256_matches": sha256(fr) == de.get("frame_ref_sha256"),
               "packer_vs_frame_refs_k3fix": de.get("vs_frame_refs_k3fix"), "packer_vs_released_raw": de.get("vs_released_raw"),
               "packer_ok": de.get("packer_ok") is True and de.get("vs_frame_refs_k3fix") == "accepted"
               and de.get("vs_released_raw") == "REJECTED"}
        idc["all_ok"] = all(idc[k] for k in ("tracks_json_unchanged", "track_identical_lt_1e-12", "k3fix_parquet_byte_identical_to_k3",
                                             "k3fix_parquet_differs_from_raw", "ready_entry_matches",
                                             "data_build_root_sha256_matches", "frame_ref_sha256_matches", "packer_ok"))
        id_ok &= idc["all_ok"]
        ident[f"{ep:02d}"] = idc
        k3 = next(c for c in g2j[str(ep)]["K3g2"]["checks"] if c["variant"] == "K3")
        fz = r["K3_frozen"]
        rows = {"rot_max_deg": (fz["G1_envelope"]["max_rot_dev_deg"], k3["bias_rot_max_deg"], 1e-6),
                "pos_max_cm": (fz["G1_envelope"]["max_pos_dev_cm"], k3["bias_pos_max_cm"], 1e-6),
                "pen_p95_cm": (fz["G2_penetration"]["pen_p95_cm"], k3["pen_p95_cm"], 2e-5),
                "pen_p95_raw_cm": (fz["G2_penetration"]["pen_p95_raw_cm"], k3["pen_p95_raw_cm"], 2e-5),
                "r4_n_contact": (fz["G4_contact"]["n_raw_contact_frames"], r4[ep][2], 0),
                "r4_literal_share": (fz["G4_contact"]["share_farther_gt3cm"], r4[ep][4], 1e-12)}
        repl[f"{ep:02d}"] = {k: {"here": a, "root_cause": b, "match": abs(a - b) <= tol} for k, (a, b, tol) in rows.items()}
        rep_ok &= all(v["match"] for v in repl[f"{ep:02d}"].values())
        r["seconds"] = round(time.time() - t0, 1)
        res[f"{ep:02d}"] = r
    out = {"written": time.strftime("%F %T"), "noisy_only": True, "purpose": "PREREG_v2 stage-1 clips ep07 / ep18 (K3fix = frozen K3 there)",
           "gate_code": str(BUILD_PY), "gate_code_sha256": BUILD_PY_SHA,
           "k3fix_tracks_npz_sha256": tr["k3fix_tracks_npz_sha256"], "envelope_rot_deg": kb.ENV_R, "envelope_pos_cm": kb.ENV_P,
           "rules": "as build/gates.json (G1 envelope, G2 penetration literal per-frame, G3 calibrated rest runs, G4 contact share "
                    "<= 0.05 with >= 50 contact frames); clip PASS iff G1..G4 all pass",
           "replication_frozen_K3_vs_root_cause": repl, "replication_ok": bool(rep_ok),
           "identity": ident, "identity_ok": bool(id_ok), "clips": res,
           "pass_clips": [int(e) for e, r in res.items() if r["K3fix"]["all_pass"]],
           "fail_clips": [int(e) for e, r in res.items() if not r["K3fix"]["all_pass"]]}
    print(f"envelope {kb.ENV_R:.3f} deg / {kb.ENV_P:.3f} cm; replication of root-cause numbers for frozen K3: {'OK' if rep_ok else 'MISMATCH'}; "
          f"identity checks: {'OK' if id_ok else 'FAILED'}")
    for k, v in repl.items():
        print(f"  ep{k} " + "  ".join(f"{n} {x['here']:.5g}/{x['root_cause']:.5g}{'' if x['match'] else ' MISMATCH'}" for n, x in v.items()))
    for k, v in ident.items():
        print(f"  ep{k} identity: K3fix parquet {str(v['parquet_sha256_k3fix_root'])[:16]} == K3 {str(v['parquet_sha256_k3_root'])[:16]}: "
              f"{v['k3fix_parquet_byte_identical_to_k3']}; != raw {str(v['parquet_sha256_raw'])[:16]}: {v['k3fix_parquet_differs_from_raw']}; "
              f"track |K3fix-K3| {v['k3fix_vs_frozen_k3_full_track_max_abs']:.1e}; READY {v['ready_entry_matches']}; data_build "
              f"{v['data_build_root_sha256_matches']}; frame ref {v['frame_ref_sha256_matches']}; packer {v['packer_vs_frame_refs_k3fix']}"
              f"/{v['packer_vs_released_raw']} -> {'ok' if v['all_ok'] else 'FAIL'}")
    for k, r in res.items():
        for name in ("K3fix", "K3_frozen"):
            o = r[name]
            g1, g2, g3, g4 = o["G1_envelope"], o["G2_penetration"], o["G3_rest"], o["G4_contact"]
            print(f"ep{k} {r['object']:<18} {name:<9} G1 {'pass' if g1['pass'] else 'FAIL'} (rot max {g1['max_rot_dev_deg']:.1f} "
                  f"pos max {g1['max_pos_dev_cm']:.2f}, {g1['frames_beyond']} frames beyond) | G2 {'pass' if g2['pass'] else 'FAIL'} "
                  f"({g2['frames_deeper_than_raw_by_gt_env']} frames, max excess {g2['max_excess_depth_cm']:.1f} cm; p95 {g2['pen_p95_cm']:.2f} vs raw "
                  f"{g2['pen_p95_raw_cm']:.2f}, F5 form fires {g2['f5_p95_form_fires']}; {r['n_support_prims']} support prims) | G3 "
                  f"{'pass' if g3['pass'] else 'FAIL'} (flagged {g3['flagged_runs']} w0 forced {g3['w0_frames_forced_in_flagged']}) | G4 "
                  f"{'pass' if g4['pass'] else 'FAIL'} (n {g4['n_raw_contact_frames']}, share {g4['share_farther_gt3cm']:.4f}) -> "
                  f"{'PASS' if o['all_pass'] else 'FAIL'}")
    print(f"stage-1 unchanged clips passing all gates: {out['pass_clips']}; failing: {out['fail_clips']}")
    if write:
        p = OUT.resolve()
        assert str(p).startswith(str(BUILD.resolve()) + "/")
        p.write_text(json.dumps(out, indent=1, default=float))
        print(f"wrote {OUT}")
    return 0 if (rep_ok and id_ok) else 1


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "gates":
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main("--no-write" not in sys.argv[2:]))
