"""Build a G1 reference data root whose object track is the G2 filter output (NOISY-ONLY; reads no clean data).

The maths is the teammate's scripts/k3fix_build_data.py build_one (copied, his files read only): the Tier-2 G1 parquet
of each episode with ONLY object_body_position/wxyz and object_root_position/axis_angle replaced by T o G2, T being the
rigid map from the released Tier-2 world to the G1 world fitted on the first visible frame (residual checked). The
human and robot columns stay the raw Tier-2 ones (G2's human root-drift estimate is not applied). Also writes, per
episode, the frame references for the kit packer (G2 pose on the visible frames, released Tier-2 world), and checks
with the packer's own functions that the new parquet is accepted against them and rejected against the raw input.

    envs/kit/bin/python -I tmp/t2/g2data/build_g2_data.py <out root> <episode> [<episode> ...]
"""
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ME = Path("/root/autodl-tmp/haoran.zhang")
W = Path("/root/autodl-tmp/zuxing.lu")
T2G1 = ME / "data/human_motion_data_tier2"
G2OUT = W / "tmp/tier2_noise/kfix/gen/G2_surface_only/out"
OBJ_COLS = ("object_body_position", "object_body_wxyz", "object_root_position", "object_root_axis_angle")
spec = importlib.util.spec_from_file_location("e2ro", ME / "tmp_analysis/tier2/kalman/eval/e2_build_eval_inputs.py")
e2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e2)  # loads npz files and the kit modules, read only
ps = e2.ps


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def build_one(ep, seq, full, dst):
    raw = e2.read_episode(e2.episode_parquet(e2.T2, ep), ep)
    idx = np.asarray(raw.frame_index, dtype=np.int64)
    rel = raw.pose_xyzw[:, 0]
    tab = pq.read_table(T2G1 / f"whole_body/soma/sequence_id={seq}/robot_name=g1/data.parquet")
    bp, bq = e2.col(tab, "object_body_position")[:, 0], e2.col(tab, "object_body_wxyz")[:, 0]
    assert len(bp) == len(full), (ep, len(bp), len(full))
    rq = e2.xyzw_to_wxyz(rel[:, 3:])
    qT = e2.qmul(bq[idx[0]], e2.qconj(rq[0])); qT /= np.linalg.norm(qT)
    tT = bp[idx[0]] - e2.qrot(qT, rel[0, :3])
    res_p = float(np.linalg.norm(e2.qrot(np.broadcast_to(qT, rq.shape), rel[:, :3]) + tT - bp[idx], axis=-1).max())
    res_a = float(e2.qangle(e2.qmul(np.broadcast_to(qT, rq.shape), rq), bq[idx]).max())
    assert res_p < 1e-5 and res_a < 1e-5, (ep, res_p, res_a)
    kq = e2.xyzw_to_wxyz(full[:, 3:])
    nq = e2.qmul(np.broadcast_to(qT, kq.shape), kq); nq /= np.linalg.norm(nq, axis=-1, keepdims=True)
    npos = e2.qrot(np.broadcast_to(qT, kq.shape), full[:, :3]) + tT
    rp, rr = e2.col(tab, "object_root_position"), e2.aa_to_q(e2.col(tab, "object_root_axis_angle"))
    loc_q = e2.qmul(e2.qconj(bq), rr)
    loc_p = e2.qrot(e2.qconj(bq), rp - bp)
    new_rq = e2.qmul(nq, loc_q); new_rq /= np.linalg.norm(new_rq, axis=-1, keepdims=True)
    new_rp = npos + e2.qrot(nq, loc_p)
    out = tab
    for cname, arr in (("object_body_position", npos[:, None]), ("object_body_wxyz", nq[:, None]),
                       ("object_root_position", new_rp), ("object_root_axis_angle", e2.q_to_aa(new_rq))):
        i = out.schema.get_field_index(cname)
        out = out.set_column(i, out.schema.field(i), pa.array([arr.astype(np.float32).tolist()], type=out.schema.field(i).type))
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, dst)
    back = pq.read_table(dst)
    assert back.num_rows == 1 and back.schema.equals(tab.schema)
    for cname in tab.column_names:
        if cname not in OBJ_COLS:
            assert back.column(cname).equals(tab.column(cname)), cname
    ch_p = np.linalg.norm(npos - bp, axis=-1) * 100
    ch_a = np.degrees(e2.qangle(nq, bq))
    return raw, idx, {"fit_resid_m": res_p, "fit_resid_rad": res_a, "change_vs_raw_mean_cm": float(ch_p.mean()),
                      "change_vs_raw_max_cm": float(ch_p.max()), "change_vs_raw_mean_deg": float(ch_a.mean()),
                      "change_vs_raw_max_deg": float(ch_a.max())}


def main():
    root = Path(sys.argv[1]).resolve()
    assert str(root).startswith(str(W) + "/"), root
    eps = [int(e) for e in sys.argv[2:]]
    manifest = json.loads((T2G1 / "whole_body/soma/track2_manifest.json").read_text())
    ep_to_seq = {int(v["episode_index"]): v["sequence_id"] for v in manifest.values()}
    fref = root / "frame_refs_g2"
    fref.mkdir(parents=True, exist_ok=True)
    (root / "whole_body/reconstructed_stage").mkdir(parents=True, exist_ok=True)
    shutil.copy2(T2G1 / "whole_body/soma/track2_manifest.json", root / "whole_body/soma/track2_manifest.json") \
        if (root / "whole_body/soma").exists() else None
    rows = {}
    for ep in eps:
        seq = ep_to_seq[ep]
        pose = np.load(G2OUT / f"pose_ep{ep:02d}.npy")
        rel_pq = Path(f"whole_body/soma/sequence_id={seq}/robot_name=g1/data.parquet")
        raw = e2.read_episode(e2.episode_parquet(e2.T2, ep), ep)
        idx = np.asarray(raw.frame_index, dtype=np.int64)
        n = len(e2.col(pq.read_table(T2G1 / rel_pq), "object_body_position"))
        assert len(pose) == len(idx), (ep, len(pose), len(idx))
        full = np.empty((n, 7))
        vis = np.zeros(n, bool); vis[idx] = True
        full[idx] = pose
        for t in np.flatnonzero(~vis):  # invisible frames: the nearest visible frame, as G2's input
            full[t] = pose[np.argmin(np.abs(idx - t))]
        _, _, r = build_one(ep, seq, full, root / rel_pq)
        shutil.copy2(T2G1 / rel_pq.parent.parent / "ground_plane.json", root / rel_pq.parent.parent / "ground_plane.json")
        usda = f"whole_body/reconstructed_stage/{seq}_support.usda"
        if (T2G1 / usda).exists():
            shutil.copy2(T2G1 / usda, root / usda)
        fdst = fref / f"episode_{ep:06d}.npz"
        np.savez(fdst, episode_index=np.asarray(ep), coordinate_frame=np.asarray("released_track2_world"),
                 frame_index=idx, pose_xyzw=pose[:, None, :].astype(np.float64))
        back = pq.read_table(root / rel_pq)
        bpn, bqn = e2.col(back, "object_body_position")[:, 0], e2.col(back, "object_body_wxyz")[:, 0]
        ev = np.concatenate([bpn[idx], e2.wxyz_to_xyzw(bqn[idx])], -1)
        try:
            _, pe, ae, _ = ps.convert_track2_frame(ev, ev, ps.input_frame_reference(fref, ep, idx, 1, raw.object_names))
            r.update(vs_frame_refs_g2="accepted", packer_resid_m=pe, packer_resid_rad=ae)
        except ValueError as error:
            r.update(vs_frame_refs_g2="REJECTED", error=str(error)[:200])
        try:
            ps.convert_track2_frame(ev, ev, raw.pose_xyzw[:, 0]); r["vs_released_raw"] = "accepted"
        except ValueError:
            r["vs_released_raw"] = "REJECTED"
        r.update(sequence_id=seq, g2_pose_sha256=sha(G2OUT / f"pose_ep{ep:02d}.npy"), parquet_sha256=sha(root / rel_pq),
                 frame_ref_sha256=sha(fdst), ok=r["vs_frame_refs_g2"] == "accepted" and r["vs_released_raw"] == "REJECTED")
        rows[ep] = r
        print(ep, json.dumps(r))
    (root / "build.json").write_text(json.dumps(rows, indent=1))
    return 0 if all(r["ok"] for r in rows.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
