#!/usr/bin/env python3
"""EVAL INPUT step 2 (NOISY-ONLY, no clean data): everything the single real evaluation of the K3-filtered reference needs.

a) G1 reference copy  KOUT/eval/data_k3/whole_body/...  (layout of tmp_analysis/tier2/hybrid_data, which the evaluator
   already ran on): the Tier-2 G1 parquet with ONLY object_body_position/wxyz and object_root_position/axis_angle replaced.
   The Tier-2 G1 object track is an exact rigid transform T of the released Tier-2 object track (fitted at the first
   visible frame, residual checked on every visible frame); the new track is T o K3 (k3_fullframe.npz, every frame).
   The object root keeps its pose relative to the object body. Robot, human, contact columns are untouched.
b) Frame references  KOUT/eval/frame_refs_k3/episode_XXXXXX.npz  for the kit packer's --frame-references: the frozen K3
   track (outputs/kalman_poses.npz) in the released Tier-2 world, on the released frame grid.
c) Checkpoint copies  KOUT/eval/ckpt/epNN/model_IT.safetensors  of the 30 Tier-2 C3r2 exports in early_scores/shard.json,
   with only cfg.task.parquet in the safetensors header pointing at (a) (as c3r2_on_t2/make_t2_ckpts.py); the tensor
   byte buffer is verified identical; shard_k3.json + ckpt_manifest.json.
d) Packer emulation (no clean data): convert_track2_frame(new G1 track -> K3 frame ref) must pass; against the released
   noisy input it should be rejected (shows the frame reference is required and that the input really changed).
Run: <v2dkit python> -I -B e2_build_eval_inputs.py
"""
import hashlib, importlib.util, json, shutil, struct, sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ME = Path("/root/autodl-tmp/haoran.zhang")
KIT = ME / "v2d_submission_kit_src/v2d_submission_kit"
sys.path.insert(0, str(KIT))
from v2dlb.trajectory import read_episode, episode_parquet  # noqa: E402

_spec = importlib.util.spec_from_file_location("pack_submission_ro", KIT / "tools/pack_submission.py")
ps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ps)

KOUT = ME / "tmp_analysis/tier2/kalman"
EV = KOUT / "eval"
T2 = ME / "data/video_to_data_challenge/track_2/tier_2_synthetic_noise"
G1 = ME / "data/human_motion_data_tier2"
DATA = EV / "data_k3"
FREF = EV / "frame_refs_k3"
CK = EV / "ckpt"
SHARD_SRC = ME / "tmp_analysis/tier2/early_scores/shard.json"
OLD = str(G1).encode() + b"/whole_body/"
NEW = str(DATA).encode() + b"/whole_body/"
CFG_KEY = "user.resolved_config_json"
KP = np.load(KOUT / "outputs/kalman_poses.npz")
FF = np.load(EV / "k3_fullframe.npz")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def qmul(a, b):  # wxyz
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0); w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2], -1)


def qconj(q): return q * np.array([1, -1, -1, -1])


def qrot(q, v):
    qv = np.concatenate([np.zeros(v.shape[:-1] + (1,)), v], -1)
    return qmul(qmul(q, qv), qconj(q))[..., 1:]


def qangle(a, b):
    """Robust geodesic angle (rad) between wxyz quaternions of any norm (atan2 form; arccos is ill-conditioned near 0)."""
    a = a / np.linalg.norm(a, axis=-1, keepdims=True); b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    d = qmul(qconj(a), b)
    return 2 * np.arctan2(np.linalg.norm(d[..., 1:], axis=-1), np.abs(d[..., 0]))


def aa_to_q(aa):
    th = np.linalg.norm(aa, axis=-1, keepdims=True)
    axis = np.where(th > 1e-12, aa / np.maximum(th, 1e-12), np.array([1.0, 0, 0]))
    return np.concatenate([np.cos(th / 2), np.sin(th / 2) * axis], -1)


def q_to_aa(q):
    q = q * np.where(q[..., :1] < 0, -1.0, 1.0)
    th = 2 * np.arccos(np.clip(q[..., :1], -1, 1))
    s = np.sqrt(np.maximum(1 - q[..., :1] ** 2, 0))
    return np.where(s > 1e-9, q[..., 1:] / np.maximum(s, 1e-12) * th, 2 * q[..., 1:])


def col(table, name):
    return np.asarray(table.column(name).to_pylist()[0], dtype=np.float64)


def xyzw_to_wxyz(q): return np.concatenate([q[..., 3:4], q[..., :3]], -1)


def wxyz_to_xyzw(q): return np.concatenate([q[..., 1:], q[..., :1]], -1)


def main():
    manifest = json.loads((G1 / "whole_body/soma/track2_manifest.json").read_text())
    ep_to_seq = {int(v["episode_index"]): v["sequence_id"] for v in manifest.values()}
    (DATA / "whole_body/soma").mkdir(parents=True, exist_ok=True)
    (DATA / "whole_body/reconstructed_stage").mkdir(parents=True, exist_ok=True)
    FREF.mkdir(parents=True, exist_ok=True)
    rows = []
    for ep in range(30):
        seq = ep_to_seq[ep]
        raw = read_episode(episode_parquet(T2, ep), ep)
        idx = np.asarray(raw.frame_index, dtype=np.int64)
        rel = raw.pose_xyzw[:, 0]                                         # released noisy, visible frames, xyzw
        vis = FF[f"vis__{ep}"]
        assert np.array_equal(np.flatnonzero(vis), idx), ep
        k3_vis = KP[f"K3__{ep}"]
        k3_full = FF[f"K3full__{ep}"]
        assert np.abs(k3_full[idx] - k3_vis).max() < 1e-12
        src = G1 / f"whole_body/soma/sequence_id={seq}/robot_name=g1/data.parquet"
        tab = pq.read_table(src)
        bp, bq = col(tab, "object_body_position")[:, 0], col(tab, "object_body_wxyz")[:, 0]
        assert len(bp) == len(vis), (ep, len(bp), len(vis))
        # rigid T: G1 = T o released (first visible frame), residual on all visible frames
        rq = xyzw_to_wxyz(rel[:, 3:])
        qT = qmul(bq[idx[0]], qconj(rq[0])); qT /= np.linalg.norm(qT)
        tT = bp[idx[0]] - qrot(qT, rel[0, :3])
        res_p = float(np.linalg.norm(qrot(np.broadcast_to(qT, rq.shape), rel[:, :3]) + tT - bp[idx], axis=-1).max())
        res_a = float(qangle(qmul(np.broadcast_to(qT, rq.shape), rq), bq[idx]).max())
        assert res_p < 1e-5 and res_a < 1e-5, (ep, res_p, res_a)
        # new body track on every frame
        kq = xyzw_to_wxyz(k3_full[:, 3:])
        nq = qmul(np.broadcast_to(qT, kq.shape), kq); nq /= np.linalg.norm(nq, axis=-1, keepdims=True)
        npos = qrot(np.broadcast_to(qT, kq.shape), k3_full[:, :3]) + tT
        # object root: keep its pose relative to the body
        rp, rr = col(tab, "object_root_position"), aa_to_q(col(tab, "object_root_axis_angle"))
        loc_q = qmul(qconj(bq), rr)
        loc_p = qrot(qconj(bq), rp - bp)
        new_rq = qmul(nq, loc_q); new_rq /= np.linalg.norm(new_rq, axis=-1, keepdims=True)
        new_rp = npos + qrot(nq, loc_p)
        out = tab
        for cname, arr in (("object_body_position", npos[:, None]), ("object_body_wxyz", nq[:, None]),
                           ("object_root_position", new_rp), ("object_root_axis_angle", q_to_aa(new_rq))):
            i = out.schema.get_field_index(cname)
            out = out.set_column(i, out.schema.field(i), pa.array([arr.astype(np.float32).tolist()], type=out.schema.field(i).type))
        dst = DATA / f"whole_body/soma/sequence_id={seq}/robot_name=g1/data.parquet"
        dst.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(out, dst)
        shutil.copy2(src.parent.parent / "ground_plane.json", dst.parent.parent / "ground_plane.json")
        usda = G1 / f"whole_body/reconstructed_stage/{seq}_support.usda"
        shutil.copy2(usda, DATA / "whole_body/reconstructed_stage" / usda.name)
        back = pq.read_table(dst)
        assert back.num_rows == 1 and back.schema.equals(tab.schema)
        for cname in tab.column_names:
            if cname not in ("object_body_position", "object_body_wxyz", "object_root_position", "object_root_axis_angle"):
                assert back.column(cname).equals(tab.column(cname)), cname
        bpn, bqn = col(back, "object_body_position")[:, 0], col(back, "object_body_wxyz")[:, 0]
        assert np.abs(bpn - npos).max() < 1e-5
        # frame reference npz (frozen K3, released frame, released frame grid)
        np.savez(FREF / f"episode_{ep:06d}.npz", episode_index=np.asarray(ep), coordinate_frame=np.asarray("released_track2_world"),
                 frame_index=idx, pose_xyzw=k3_vis[:, None, :].astype(np.float64))
        fr = ps.input_frame_reference(FREF, ep, idx, 1, raw.object_names)
        # packer emulation: evaluator reference = new G1 body track (any rigid frame), achieved = same
        ev = np.concatenate([bpn[idx], wxyz_to_xyzw(bqn[idx])], -1)
        _, pe, ae, _ = ps.convert_track2_frame(ev, ev, fr)
        try:
            ps.convert_track2_frame(ev, ev, rel)
            vs_noisy = "accepted"
        except ValueError:
            vs_noisy = "REJECTED"
        ch_p = np.linalg.norm(npos - bp, axis=-1) * 100
        ch_a = np.degrees(qangle(nq, bq))
        rows.append(dict(episode=ep, sequence_id=seq, frames=int(len(vis)), invisible=int((~vis).sum()),
                         fit_resid_m=res_p, fit_resid_rad=res_a, packer_resid_vs_k3_m=pe, packer_resid_vs_k3_rad=ae,
                         packer_vs_released_noisy=vs_noisy, ref_change_mean_cm=float(ch_p.mean()), ref_change_max_cm=float(ch_p.max()),
                         ref_change_mean_deg=float(ch_a.mean()), ref_change_max_deg=float(ch_a.max()),
                         ref_change_frame0_cm=float(ch_p[0]), ref_change_frame0_deg=float(ch_a[0]), parquet=str(dst)))
        print(f"ep{ep:02d} fit {res_p:.1e}m/{res_a:.1e}rad packer-vs-K3 {pe:.1e}m/{ae:.1e}rad vs-noisy {vs_noisy} "
              f"change mean {ch_p.mean():.2f}cm {ch_a.mean():.1f}deg (frame0 {ch_p[0]:.2f}cm {ch_a[0]:.2f}deg)", flush=True)
    # c) checkpoints
    shard = json.loads(SHARD_SRC.read_text())
    new_shard, cman = {}, []
    for key in sorted(shard, key=int):
        ep = int(key)
        srcp = Path(shard[key])
        rawb = srcp.read_bytes()
        n = struct.unpack("<Q", rawb[:8])[0]
        header, data = rawb[8:8 + n], rawb[8 + n:]
        pos, start = [], 0
        while (i := header.find(OLD, start)) != -1:
            pos.append(i); start = i + 1
        chosen = []
        for i in pos:
            trial = header[:i] + NEW + header[i + len(OLD):]
            cfg = json.loads(json.loads(trial)["__metadata__"][CFG_KEY])
            if cfg["task"]["parquet"].startswith(str(DATA) + "/") and not cfg["reference"]["parquet"].startswith(str(DATA) + "/"):
                chosen.append(trial)
        assert len(chosen) == 1, (ep, len(pos), len(chosen))
        nh = chosen[0].rstrip(b" ")
        nh += b" " * ((-(8 + len(nh))) % 8)
        dst = CK / f"ep{ep:02d}" / srcp.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(struct.pack("<Q", len(nh)) + nh + data)
        # verify
        hb = json.loads(header); hn = json.loads(nh)
        assert set(hb) == set(hn) and all(hb[k] == hn[k] for k in hb if k != "__metadata__")
        mb, mn = hb["__metadata__"], hn["__metadata__"]
        assert set(mb) == set(mn) and all(mb[k] == mn[k] for k in mb if k != CFG_KEY)
        cb, cn = json.loads(mb[CFG_KEY]), json.loads(mn[CFG_KEY])
        newp = cn["task"]["parquet"]; cn["task"]["parquet"] = cb["task"]["parquet"]
        assert cb == cn, ep
        assert Path(newp).is_file() and newp == rows[ep]["parquet"], (ep, newp)
        back = dst.read_bytes(); n2 = struct.unpack("<Q", back[:8])[0]
        assert back[8 + n2:] == data
        new_shard[str(ep)] = str(dst)
        cman.append(dict(episode=ep, source_checkpoint=str(srcp), source_sha256=sha(srcp), k3_checkpoint=str(dst), k3_sha256=sha(dst),
                         tensor_bytes_sha256=hashlib.sha256(data).hexdigest(), tensors_identical=True,
                         old_task_parquet=cb["task"]["parquet"], new_task_parquet=newp, reference_parquet_kept=cb["reference"]["parquet"]))
    (EV / "shard_k3.json").write_text(json.dumps(new_shard, indent=1) + "\n")
    (EV / "ckpt_manifest.json").write_text(json.dumps(cman, indent=1) + "\n")
    (EV / "build_report.json").write_text(json.dumps(rows, indent=1) + "\n")
    agg = dict(n=len(rows), max_fit_resid_m=max(r["fit_resid_m"] for r in rows), max_packer_resid_vs_k3_m=max(r["packer_resid_vs_k3_m"] for r in rows),
               max_packer_resid_vs_k3_rad=max(r["packer_resid_vs_k3_rad"] for r in rows),
               n_rejected_vs_released_noisy=sum(r["packer_vs_released_noisy"] == "REJECTED" for r in rows),
               mean_ref_change_cm=float(np.mean([r["ref_change_mean_cm"] for r in rows])),
               mean_ref_change_deg=float(np.mean([r["ref_change_mean_deg"] for r in rows])),
               invisible_frames=sum(r["invisible"] for r in rows), checkpoints=len(new_shard))
    print(json.dumps(agg, indent=1))
    (EV / "build_summary.json").write_text(json.dumps(agg, indent=1) + "\n")


if __name__ == "__main__":
    main()
