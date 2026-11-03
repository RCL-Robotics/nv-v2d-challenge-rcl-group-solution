#!/usr/bin/env python
"""t3f0_align.py -- Tier-3 f0 step 2 (ALIGN): GT-free hand-to-object contact gap on Han Zhou's frame-0 tracks (f0) vs his
opus_v2 tracks, with the SAME reconstructed human (soma_params.npz is byte-identical in both releases). NEW FILE 2026-10-24.
CPU only (CUDA_VISIBLE_DEVICES=""), v2d env (py-soma-x + robotic_grounding). Reads only:
  data/human_motion_data_tier3_f0   (byte copies of tier3_frame0_supports_20261024 + our retarget parquets)
  data/human_motion_data_tier3_opus (byte copies of tier3_opus_20261021 + our retarget parquets)
Writes only tmp_analysis/tier3/f0/align/. No clean, Tier-1 or Tier-2 data. Nothing here touches the object tracks.

World = the G1 parquet world the trainer uses: the frame chain of scripts/retarget/soma_to_g1.py @ f2923d61 (imported
read-only, same code path as scripts/t3f0_frames.py): OpenCV -> SOMA axes -> frame-0 root anchor -> plane levelling ->
robot axes; object only: + its own constant frame-0 ground lift (per track). Each track is checked against its parquet.

Measures (per frame t, hand side s, track X in {f0, opus}); PRIMARY = j_X (the spec's palm/fingertip points):
  j_X     = min unsigned distance of 6 hand joints (palm centre = mean(wrist, Index1..Pinky1), 5 fingertip *End joints)
            to the object surface (cm).
  e_X     = (secondary, lenient: ANY of ~1500 skin vertices incl. back of hand / wrist counts) max(gap, pen) (cm): gap = min unsigned distance from the SOMA hand skin vertices of side s to the object
            surface; pen = depth reached inside the object by >= 5 % of those vertices (sign from the outward normal of
            the nearest surface sample; meshes are winding-consistent but not watertight, so 5 % guards against a few
            mis-signed vertices). 0 = touching, >0 = the hand floats or sinks in. sd_X = gap if pen == 0 else -pen.
            v1 of this script (sha 1224fd55) used e_X as the rule metric; on ep03 it gave 1.7 -> 2.0 cm although the
            object moved 14.9 cm relative to the hand (some other skin vertex is always near), so the rule metric was
            switched to j_X (= the spec) BEFORE any other episode was run.
  disp    = |T_f0(t) T_opus(t)^-1 h - h| at the hand centroid h: how far the f0 correction moves the object relative
            to the hand (track-only quantity; the human is identical).
  rp_X    = unsigned distance of the ROBOT reference palm link (parquet ee_pose_w, identical in both parquets) to the
            object surface: what the robot palm must bridge, vs termination.palm_position_threshold (0.15 default).
Contact (pre-registered here, fixed before looking at f0-vs-opus numbers; GT-free, uses only the reconstruction):
  moving_X(t)   object centroid speed >= 3 cm/s OR angular speed >= 15 deg/s (central difference over +-3 frames)
  near_X(t,s)   e_X <= 5 cm
  comove_X(t,s) (v3, PRIMARY) the palm centre's net displacement over [t-6, t+6] measured in the object frame of X is
                <= max(2 cm, 0.35 x its net displacement in the world): the hand moves (mostly) WITH the object
  rigid_X(t,s)  (v2, reported as a check only) over [t-6, t+6] the hand-vertex centroid in the object frame stays
                within 2 cm of its window mean. v2 found almost no contact on noisy or fast-rotating tracks (ep04 bat held
                for the whole clip: 3 frames; ep02: 0), so v3 replaced it as the contact cue before the final run.
  contact_X     = moving_X & near_X & comove_X ;  C(s) = contact_opus | contact_f0  (union: neither track is favoured)
  Both tracks are scored on the same side-frames C. static touch (reported only): not moving on either track and
  j <= 2 cm on either track.
Rule metric (main session's pre-registered mitigation): qualifies if median j_f0 over C > 3 cm OR median j_f0 - median
j_opus > 2 cm.
Usage: t3f0_align.py EP [EP ...]   (writes align/epNN.json; summary via --summary)
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json, time, hashlib, importlib.util, pickle
from pathlib import Path
import numpy as np

ME = Path("/root/autodl-tmp/haoran.zhang")
ROOT = {"f0": ME / "data/human_motion_data_tier3_f0/whole_body/soma",
        "opus": ME / "data/human_motion_data_tier3_opus/whole_body/soma"}
OUT = ME / "tmp_analysis/tier3/f0/align"
S2G = ME / "code/nv-v2d-challenge-codebase/robotic_grounding/scripts/retarget/soma_to_g1.py"

FPS = 30.0
K_VEL = 3            # central difference half-width (frames)
V_MIN = 0.03         # m/s
W_MIN = 15.0         # deg/s
NEAR = 0.05          # m
RIGID_HALF = 6       # frames
RIGID_TOL = 0.02     # m
COMOVE_HALF = 6      # frames
COMOVE_TOL = 0.02    # m
COMOVE_FRAC = 0.35   # object-frame net displacement allowed, as a fraction of the world net displacement
STATIC_TOUCH = 0.02  # m
FAR_SKIP = 0.40      # m: hand farther than this from the object's bounding sphere -> coarse distance only
SKIN_UB = 0.40       # m: KD-tree distance_upper_bound for skin vertices (speed); beyond -> >= 40 cm
SKIN_STRIDE = 2      # use every 2nd hand skin vertex (~1500 per hand)
RULE_ABS = 3.0       # cm
RULE_DELTA = 2.0     # cm
MIN_CONTACT = 30     # side-frames; below this the episode is flagged "few contact frames"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_s2g():
    spec = importlib.util.spec_from_file_location("t3f0_align_soma_to_g1", S2G)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def q2m(wxyz):
    from scipy.spatial.transform import Rotation as R
    return R.from_quat(np.asarray(wxyz), scalar_first=True).as_matrix()


def stats(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return dict(n=0, median=None, p90=None, mean=None, frac_le2=None)
    return dict(n=int(x.size), median=round(float(np.median(x)), 2), p90=round(float(np.percentile(x, 90)), 2),
                mean=round(float(x.mean()), 2), frac_le2=round(float((x <= 2.0).mean()), 3))


def segments(mask):
    """contiguous True runs -> list of (start, end_inclusive)"""
    m = np.r_[False, mask, False].astype(np.int8)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0].tolist(), (np.where(d == -1)[0] - 1).tolist()))


def object_chain(s2g, kin, src2rob, norm_transform, G, P_cv, mesh_v):
    object_poses = s2g._convert_object_poses_cv_to_soma(P_cv)
    object_poses = np.einsum("ij,tjk->tik", norm_transform, object_poses)
    object_poses = np.einsum("ij,tjk->tik", G, object_poses)
    p0 = kin.transform_source_position(object_poses[0, :3, 3])
    R0 = kin.transform_world_rotation(object_poses[0, :3, :3])
    lift = s2g.compute_object_ground_lift(
        mesh_v @ R0.T + p0, s2g.ReferencePlane.horizontal(),
        penetration_tolerance=s2g.OBJECT_GROUND_PENETRATION_TOLERANCE_M,
        clearance=s2g.OBJECT_GROUND_CLEARANCE_M, max_lift=s2g.OBJECT_GROUND_MAX_LIFT_M)
    object_poses[:, :3, 3] += src2rob.T @ np.array([0.0, 0.0, lift.applied_lift])
    obj_pos = object_poses[:, :3, 3] @ src2rob.T
    obj_R = np.einsum("ij,tjk->tik", src2rob, object_poses[:, :3, :3])
    return obj_pos, obj_R, float(lift.applied_lift)


def read_parquet(pq_path):
    import pyarrow.parquet as pq
    t = pq.read_table(pq_path, columns=["object_body_position", "object_body_wxyz", "ee_pose_w", "ee_link_names",
                                        "hand_contact_active", "source_payload"])
    r = t.to_pylist()[0]
    pay = pickle.loads(r["source_payload"])   # our own retarget output (soma_to_g1 builder dict)
    return dict(obj_pos=np.asarray(r["object_body_position"], dtype=np.float64)[:, 0],
                obj_wxyz=np.asarray(r["object_body_wxyz"], dtype=np.float64)[:, 0],
                ee=np.asarray(r["ee_pose_w"], dtype=np.float64), ee_names=list(r["ee_link_names"]),
                hca=np.asarray(r["hand_contact_active"], dtype=np.float64),
                soma_joints=np.asarray(pay["soma_joints"], dtype=np.float64))


def main():
    args = sys.argv[1:]
    if args and args[0] == "--summary":
        return summary()
    dump = "--dump" in args
    eps = [int(a) for a in args if a != "--dump"]
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    import trimesh
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation as R
    s2g = load_s2g()
    config = s2g.load_robot_config("g1")
    kin = s2g.WholeBodyKinematics(config=config)
    soma = s2g.SOMA(data_root=None, identity_model_type="mhr", device=torch.device("cpu"))
    src2rob = np.asarray(config.r_world, dtype=np.float64)
    man = json.loads((ROOT["f0"] / "track2_manifest.json").read_text())
    seq_of = {v["episode_index"]: k for k, v in man.items()}
    nthreads = int(os.environ.get("OMP_NUM_THREADS", "4"))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dump").mkdir(exist_ok=True)
    for ep in eps:
        t0 = time.time()
        seq = seq_of[ep]
        fold = {k: ROOT[k] / f"sequence_id={seq}" for k in ROOT}
        same = {n: sha(fold["f0"] / n) == sha(fold["opus"] / n)
                for n in ("soma_params.npz", "ground_plane.json", "reconstructed_mesh/output_aligned.glb")}
        if not all(same.values()):
            raise SystemExit(f"ep{ep}: human/ground/mesh differ between roots: {same}")
        objf = {k: fold[k] / "object" / "textured_mesh.obj" for k in ROOT}
        obj_same = sha(objf["f0"]) == sha(objf["opus"])
        # ---- human: verbatim chain of soma_to_g1.main() (@ f2923d61), as scripts/t3f0_frames.py ----
        motion = soma.load_motion(params_path=str(fold["f0"] / "soma_params.npz"), normalize=True)
        transl_first = motion["first_frame_transl"]; R_first_inv = motion["first_frame_R_inv"]
        norm_transform = np.eye(4); norm_transform[:3, :3] = R_first_inv; norm_transform[:3, 3] = -R_first_inv @ transl_first
        plane = s2g.load_ground_plane_robot_frame(fold["f0"] / "ground_plane.json", cv_to_source=s2g._R_CV_TO_SOMA,
                                                  first_frame_anchor=norm_transform, source_to_robot=src2rob)
        rot_r, tr_r = s2g.compute_plane_leveling_transform(plane)
        rot_s = src2rob.T @ rot_r @ src2rob; tr_s = src2rob.T @ tr_r
        G = np.eye(4); G[:3, :3] = rot_s; G[:3, 3] = tr_s
        joints_src = np.einsum("ij,tkj->tki", rot_s, motion["joints"]) + tr_s
        joints_w = joints_src @ src2rob.T
        verts_w = (np.einsum("ij,tvj->tvi", rot_s, motion["vertices"]) + tr_s) @ src2rob.T
        names = list(motion["joint_names"])
        T = len(joints_w)
        t_fk = time.time() - t0
        # ---- object mesh (mesh-local frame used by the retarget) ----
        om = trimesh.load(str(objf["f0"]), force="mesh", process=False)
        mv = np.asarray(om.vertices, dtype=np.float64)
        mesh_v_lines = []
        with open(objf["f0"]) as f:
            for line in f:
                if line.startswith("v "):
                    p = line.split(); mesh_v_lines.append([float(p[1]), float(p[2]), float(p[3])])
        mesh_v_lines = np.asarray(mesh_v_lines)
        merged = om.copy(); merged.merge_vertices()
        watertight = bool(merged.is_watertight); winding = bool(merged.is_winding_consistent)
        fc = np.asarray(om.triangles_center)
        c_area = (fc * np.asarray(om.area_faces)[:, None]).sum(0) / float(om.area)
        orient = float((np.einsum("ij,ij->i", fc - c_area, np.asarray(om.face_normals)) * np.asarray(om.area_faces)).sum() / 3.0)
        vol = orient                                   # divergence-theorem volume about the area centroid (m^3)
        flip = -1.0 if orient < 0 else 1.0             # make normals point outward
        fn = np.asarray(om.face_normals) * flip
        vn = np.asarray(om.vertex_normals) * flip
        S = np.vstack([mv, fc]); Nrm = np.vstack([vn, fn])
        tree = cKDTree(S)
        edges = np.asarray(om.edges_unique_length) if len(om.faces) else np.zeros(1)
        c_obj = mv.mean(0); r_obj = float(np.linalg.norm(mv - c_obj, axis=1).max())
        # ---- tracks ----
        trk, pqd, lifts, chk = {}, {}, {}, {}
        for k in ROOT:
            P_cv = np.load(fold[k] / "poses.npy")
            pos, Rm, lift = object_chain(s2g, kin, src2rob, norm_transform, G, P_cv, mesh_v_lines)
            trk[k] = (pos, Rm); lifts[k] = lift
            pqd[k] = read_parquet(fold[k] / "robot_name=g1" / "data.parquet")
            Rp = q2m(pqd[k]["obj_wxyz"])
            chk[k] = dict(obj_pos_max_err_m=float(np.abs(pqd[k]["obj_pos"] - pos).max()),
                          obj_rot_max_err=float(np.abs(Rp - Rm).max()),
                          soma_joints_max_err_m=float(np.abs(pqd[k]["soma_joints"] @ src2rob.T - joints_w).max()))
        ee_diff = float(np.abs(pqd["f0"]["ee"][:, :, :3] - pqd["opus"]["ee"][:, :, :3]).max())
        bad = [k for k in ROOT if chk[k]["obj_pos_max_err_m"] > 1e-4 or chk[k]["obj_rot_max_err"] > 1e-3
               or chk[k]["soma_joints_max_err_m"] > 1e-3]
        if bad:
            raise SystemExit(f"ep{ep}: frame chain does not reproduce the parquet for {bad}: {chk}")
        # ---- hand skin vertices (nearest-joint labels at frame 0, distal of the wrist) ----
        jn = np.asarray(names)
        d0 = np.linalg.norm(verts_w[0][:, None, :] - joints_w[0][None, :, :], axis=2)
        lab = d0.argmin(1)
        hands = {}
        for side, pre in (("left", "Left"), ("right", "Right")):
            hj = [i for i, n in enumerate(names) if n.startswith(pre + "Hand")]
            wr = names.index(pre + "Hand"); m1 = names.index(pre + "HandMiddle1")
            sel = np.isin(lab, hj)
            ax = joints_w[0, m1] - joints_w[0, wr]; ax /= np.linalg.norm(ax)
            sel &= ((verts_w[0] - joints_w[0, wr]) @ ax) >= -0.005
            jidx = [wr] + [names.index(pre + f"Hand{f}1") for f in ("Index", "Middle", "Ring", "Pinky")]
            tips = [names.index(pre + f"Hand{f}End") for f in ("Thumb", "Index", "Middle", "Ring", "Pinky")]
            hands[side] = dict(vidx=np.where(sel)[0], palm_j=jidx, tips=tips)
        ee_names = pqd["f0"]["ee_names"]
        res = {}
        for k in ROOT:
            pos, Rm = trk[k]
            cw = pos + np.einsum("tij,j->ti", Rm, c_obj)       # object centroid, world
            lo = np.clip(np.arange(T) - K_VEL, 0, T - 1); hi = np.clip(np.arange(T) + K_VEL, 0, T - 1)
            dt = np.maximum(hi - lo, 1) / FPS
            vel = np.linalg.norm(cw[hi] - cw[lo], axis=1) / dt
            ang = np.degrees(R.from_matrix(np.einsum("tij,tkj->tik", Rm[hi], Rm[lo])).magnitude()) / dt
            moving = (vel >= V_MIN) | (ang >= W_MIN)
            per = {"vel": vel, "ang": ang, "moving": moving}
            for side in ("left", "right"):
                H = hands[side]
                V = verts_w[:, H["vidx"][::SKIN_STRIDE], :]                    # (T, Nh, 3)
                L = np.einsum("tnj,tjk->tnk", V - pos[:, None, :], Rm)         # object frame
                coarse = np.linalg.norm(L - c_obj, axis=2).min(1) - r_obj
                todo = np.where(coarse < FAR_SKIP)[0]
                e = np.full(T, np.nan); sd = np.full(T, np.nan); ud = np.full(T, np.nan)
                if len(todo):
                    Q = L[todo].reshape(-1, 3)
                    dd, ii = tree.query(Q, workers=nthreads, distance_upper_bound=SKIN_UB)
                    okq = np.isfinite(dd); ii = np.where(okq, ii, 0); dd = np.where(okq, dd, SKIN_UB)
                    sg = np.sign(np.einsum("ij,ij->i", Q - S[ii], Nrm[ii])); sg[(sg == 0) | ~okq] = 1.0
                    sdq = (dd * sg).reshape(len(todo), -1); ddq = dd.reshape(len(todo), -1)
                    gap = ddq.min(1); pen = np.maximum(0.0, -np.quantile(sdq, 0.05, axis=1))
                    ud[todo] = gap; e[todo] = np.maximum(gap, pen); sd[todo] = np.where(pen > 0, -pen, gap)
                far = np.setdiff1d(np.arange(T), todo)
                e[far] = np.maximum(coarse[far], FAR_SKIP); sd[far] = e[far]; ud[far] = e[far]
                # joints (palm centre + 5 tips)
                pj = joints_w[:, H["palm_j"], :].mean(1, keepdims=True)
                J = np.concatenate([pj, joints_w[:, H["tips"], :]], axis=1)
                LJ = np.einsum("tnj,tjk->tnk", J - pos[:, None, :], Rm)
                dj, _ = tree.query(LJ.reshape(-1, 3), workers=nthreads)
                jd = dj.reshape(T, -1).min(1)
                # robot reference palm (identical in both parquets)
                ei = ee_names.index(f"{side}_hand_palm_link")
                rpw = pqd["f0"]["ee"][:, ei, :3]
                LR = np.einsum("tj,tjk->tk", rpw - pos, Rm)
                dr, ir = tree.query(LR, workers=nthreads)
                # rigidity of the hand centroid in the object frame
                hc = L.mean(1)
                dev = np.full(T, np.inf)
                for t in range(T):
                    a, b = max(0, t - RIGID_HALF), min(T, t + RIGID_HALF + 1)
                    w = hc[a:b]
                    dev[t] = np.linalg.norm(w - w.mean(0), axis=1).max()
                near = e <= NEAR
                rigid = dev <= RIGID_TOL
                # co-motion of the palm centre (v3)
                pcw = pj[:, 0, :]
                pco = LJ[:, 0, :]
                l2 = np.clip(np.arange(T) - COMOVE_HALF, 0, T - 1); h2 = np.clip(np.arange(T) + COMOVE_HALF, 0, T - 1)
                Dw = np.linalg.norm(pcw[h2] - pcw[l2], axis=1); Do = np.linalg.norm(pco[h2] - pco[l2], axis=1)
                comove = Do <= np.maximum(COMOVE_TOL, COMOVE_FRAC * Dw)
                per[side] = dict(e=e, sd=sd, ud=ud, jd=jd, rp=dr, dev=dev, near=near, rigid=rigid, comove=comove,
                                 Dw=Dw, Do=Do, contact=moving & near & comove, contact_v2=moving & near & rigid, hc=hc)
            res[k] = per
        # ---- union contact, metrics on the same side-frames ----
        out_sides, pooled = {}, {k: dict(e=[], sd=[], jd=[], rp=[]) for k in ROOT}
        pooled_alt = {a: {k: dict(jd=[], e=[]) for k in ROOT} for a in ("opus_only", "f0_only")}
        disp_all, dvert_all, dhor_all, seg_list = [], [], [], []
        Pf, Rf = trk["f0"]; Po, Ro = trk["opus"]
        def per_side_rows(h):                          # retarget stores hand_contact_active as (sides, T)
            return h if h.shape[0] == 2 and h.shape[-1] == T else h.T
        hca_f0 = per_side_rows(pqd["f0"]["hca"]); hca_op = per_side_rows(pqd["opus"]["hca"])
        for si, side in enumerate(("left", "right")):
            co = res["opus"][side]["contact"]; cf = res["f0"][side]["contact"]
            C = co | cf
            idx = np.where(C)[0]
            mov_any = res["opus"]["moving"] | res["f0"]["moving"]
            st = (~mov_any) & ((res["opus"][side]["jd"] <= STATIC_TOUCH) | (res["f0"][side]["jd"] <= STATIC_TOUCH))
            # displacement of the opus-attached hand point under the f0 correction: T_f0 T_opus^-1 hc_world - hc_world
            hcw = Po + np.einsum("tij,tj->ti", Ro, res["opus"][side]["hc"])
            moved = Pf + np.einsum("tij,tj->ti", Rf, res["opus"][side]["hc"])
            dvec = (moved - hcw) * 100
            disp = np.linalg.norm(dvec, axis=1)
            disp_all += disp[idx].tolist(); dvert_all += np.abs(dvec[idx, 2]).tolist()
            dhor_all += np.linalg.norm(dvec[idx, :2], axis=1).tolist()
            sd_ = {}
            for k in ROOT:
                r_ = res[k][side]
                pooled[k]["e"] += (r_["e"][idx] * 100).tolist(); pooled[k]["sd"] += (r_["sd"][idx] * 100).tolist()
                pooled[k]["jd"] += (r_["jd"][idx] * 100).tolist(); pooled[k]["rp"] += (r_["rp"][idx] * 100).tolist()
                sd_[k] = dict(e=stats(r_["e"][idx] * 100), signed_median=(round(float(np.median(r_["sd"][idx] * 100)), 2)
                                                                           if len(idx) else None),
                              joints=stats(r_["jd"][idx] * 100), robot_palm=stats(r_["rp"][idx] * 100),
                              static_touch_joints=stats(r_["jd"][st] * 100))
            for a, msk in (("opus_only", co), ("f0_only", cf)):
                for k in ROOT:
                    for mname in ("jd", "e"):
                        pooled_alt[a][k][mname] += (res[k][side][mname][msk] * 100).tolist()
            for (a, b) in segments(C):
                seg_list.append(dict(side=side, start=a, end=b, n=b - a + 1,
                                     j_opus_med=round(float(np.median(res["opus"][side]["jd"][a:b + 1]) * 100), 2),
                                     j_f0_med=round(float(np.median(res["f0"][side]["jd"][a:b + 1]) * 100), 2),
                                     e_opus_med=round(float(np.median(res["opus"][side]["e"][a:b + 1]) * 100), 2),
                                     e_f0_med=round(float(np.median(res["f0"][side]["e"][a:b + 1]) * 100), 2),
                                     sd_f0_med=round(float(np.median(res["f0"][side]["sd"][a:b + 1]) * 100), 2),
                                     disp_med=round(float(np.median(disp[a:b + 1])), 2),
                                     rp_f0_med=round(float(np.median(res["f0"][side]["rp"][a:b + 1]) * 100), 2),
                                     rp_opus_med=round(float(np.median(res["opus"][side]["rp"][a:b + 1]) * 100), 2)))
            out_sides[side] = dict(n_hand_verts=int(len(hands[side]["vidx"])), contact_opus=int(co.sum()),
                                   contact_f0=int(cf.sum()), contact_union=int(C.sum()),
                                   contact_both=int((co & cf).sum()), static_touch=int(st.sum()),
                                   retarget_hca_f0=int(hca_f0[si].sum()), retarget_hca_opus=int(hca_op[si].sum()),
                                   retarget_hca_in_C_f0=int(hca_f0[si][idx].sum()), retarget_hca_in_C_opus=int(hca_op[si][idx].sum()),
                                   **{k: sd_[k] for k in ROOT})
        E = {k: stats(pooled[k]["e"]) for k in ROOT}
        Jst = {k: stats(pooled[k]["jd"]) for k in ROOT}
        med_f0 = Jst["f0"]["median"]; med_op = Jst["opus"]["median"]        # rule metric = joints (spec)
        q = None
        if med_f0 is not None:
            q = bool(med_f0 > RULE_ABS or (med_f0 - med_op) > RULE_DELTA)
        alt = {}
        for a in pooled_alt:
            ja = {k: stats(pooled_alt[a][k]["jd"]) for k in ROOT}
            ea = {k: stats(pooled_alt[a][k]["e"]) for k in ROOT}
            alt[a] = dict(n=ja["f0"]["n"], j_f0_median=ja["f0"]["median"], j_opus_median=ja["opus"]["median"],
                          e_f0_median=ea["f0"]["median"], e_opus_median=ea["opus"]["median"],
                          qualifies=(None if ja["f0"]["median"] is None else
                                     bool(ja["f0"]["median"] > RULE_ABS or ja["f0"]["median"] - ja["opus"]["median"] > RULE_DELTA)))
        rp = {k: stats(pooled[k]["rp"]) for k in ROOT}
        v2p = {k: [] for k in ROOT}
        for side in ("left", "right"):
            C2 = res["opus"][side]["contact_v2"] | res["f0"][side]["contact_v2"]
            for k in ROOT:
                v2p[k] += (res[k][side]["jd"][C2] * 100).tolist()
        j2 = {k: stats(v2p[k]) for k in ROOT}
        v2chk = dict(n=j2["f0"]["n"], j_f0_median=j2["f0"]["median"], j_opus_median=j2["opus"]["median"],
                     qualifies=(None if j2["f0"]["median"] is None else
                                bool(j2["f0"]["median"] > RULE_ABS or j2["f0"]["median"] - j2["opus"]["median"] > RULE_DELTA)))
        rec = dict(ep=ep, seq=seq, frames=T, lifts_cm={k: round(lifts[k] * 100, 2) for k in ROOT},
                   checks=dict(inputs_identical=same, obj_mesh_identical=obj_same, chain_vs_parquet=chk,
                               robot_palm_f0_vs_opus_max_m=ee_diff),
                   mesh=dict(n_v=int(len(mv)), n_f=int(len(om.faces)), watertight=watertight, winding_consistent=winding,
                             orient_volume_m3=round(vol, 6), normals_flipped=bool(flip < 0), edge_median_mm=round(float(np.median(edges)) * 1000, 2),
                             edge_p99_mm=round(float(np.percentile(edges, 99)) * 1000, 2), radius_m=round(r_obj, 3)),
                   sides=out_sides,
                   gap_cm={k: dict(e=E[k], signed_median=(round(float(np.median(pooled[k]["sd"])), 2) if pooled[k]["sd"] else None),
                                   joints=stats(pooled[k]["jd"]), robot_palm=rp[k]) for k in ROOT},
                   robot_palm_frac_gt15={k: (round(float((np.asarray(pooled[k]["rp"]) > 15).mean()), 3) if pooled[k]["rp"] else None) for k in ROOT},
                   robot_palm_frac_gt25={k: (round(float((np.asarray(pooled[k]["rp"]) > 25).mean()), 3) if pooled[k]["rp"] else None) for k in ROOT},
                   correction_disp_at_hand_cm=stats(disp_all), correction_disp_vertical_cm=stats(dvert_all),
                   correction_disp_horizontal_cm=stats(dhor_all),
                   delta_median_cm=(None if med_f0 is None else round(med_f0 - med_op, 2)),
                   qualifies=q, few_contact=bool(Jst["f0"]["n"] < MIN_CONTACT), alt_contact_sets=alt, v2_rigid_check=v2chk,
                   segments=seg_list,
                   definition=dict(V_MIN=V_MIN, W_MIN=W_MIN, K_VEL=K_VEL, NEAR=NEAR, RIGID_HALF=RIGID_HALF, RIGID_TOL=RIGID_TOL,
                                   STATIC_TOUCH=STATIC_TOUCH, RULE_ABS=RULE_ABS, RULE_DELTA=RULE_DELTA, MIN_CONTACT=MIN_CONTACT,
                                   FAR_SKIP=FAR_SKIP, SKIN_UB=SKIN_UB, SKIN_STRIDE=SKIN_STRIDE, rule_metric="joints",
                                   COMOVE_HALF=COMOVE_HALF, COMOVE_TOL=COMOVE_TOL, COMOVE_FRAC=COMOVE_FRAC, contact="v3_comove"),
                   script_sha256=sha(__file__), soma_to_g1_sha256=sha(S2G), seconds=round(time.time() - t0, 1),
                   seconds_fk=round(t_fk, 1))
        (OUT / ("dump" if dump else "") / f"ep{ep:02d}.json").write_text(json.dumps(rec, indent=1))
        if dump:
            arr = {}
            for k in ROOT:
                for nm in ("vel", "ang", "moving"):
                    arr[f"{k}_{nm}"] = res[k][nm]
                for side in ("left", "right"):
                    for nm in ("e", "sd", "jd", "rp", "dev", "near", "rigid", "comove", "Dw", "Do", "contact", "contact_v2", "hc"):
                        arr[f"{k}_{side}_{nm}"] = res[k][side][nm]
            arr["palm_w"] = np.stack([joints_w[:, hands[sd_]["palm_j"], :].mean(1) for sd_ in ("left", "right")], 1)
            arr["obj_pos_f0"], arr["obj_R_f0"] = trk["f0"]; arr["obj_pos_opus"], arr["obj_R_opus"] = trk["opus"]
            arr["c_obj"] = c_obj
            np.savez_compressed(OUT / "dump" / f"ep{ep:02d}.npz", **arr)
        print(json.dumps(dict(ep=ep, n=Jst["f0"]["n"], j_opus=med_op, j_f0=med_f0, p90_opus=Jst["opus"]["p90"],
                              p90_f0=Jst["f0"]["p90"], e_opus=E["opus"]["median"], e_f0=E["f0"]["median"],
                              disp=rec["correction_disp_at_hand_cm"]["median"], q=q, s=rec["seconds"])), flush=True)


def summary():
    rows = []
    for p in sorted(OUT.glob("ep[0-9][0-9].json")):
        r = json.loads(p.read_text())
        g = r["gap_cm"]
        rows.append(dict(ep=r["ep"], obj=r["seq"].split("_", 2)[2][:28], n=g["f0"]["joints"]["n"],
                         n_opus=sum(r["sides"][s]["contact_opus"] for s in ("left", "right")),
                         n_f0=sum(r["sides"][s]["contact_f0"] for s in ("left", "right")),
                         j_opus_med=g["opus"]["joints"]["median"], j_opus_p90=g["opus"]["joints"]["p90"],
                         j_f0_med=g["f0"]["joints"]["median"], j_f0_p90=g["f0"]["joints"]["p90"],
                         skin_opus_med=g["opus"]["e"]["median"], skin_f0_med=g["f0"]["e"]["median"],
                         skin_f0_signed=g["f0"]["signed_median"], skin_opus_signed=g["opus"]["signed_median"],
                         delta=r["delta_median_cm"], disp=r["correction_disp_at_hand_cm"]["median"],
                         disp_vert=r["correction_disp_vertical_cm"]["median"],
                         rp_opus=g["opus"]["robot_palm"]["median"], rp_f0=g["f0"]["robot_palm"]["median"],
                         rp_f0_p90=g["f0"]["robot_palm"]["p90"], rp15_f0=r["robot_palm_frac_gt15"]["f0"],
                         rp25_f0=r["robot_palm_frac_gt25"]["f0"],
                         q=r["qualifies"], few=r["few_contact"],
                         q_opus_only=r["alt_contact_sets"]["opus_only"]["qualifies"],
                         q_f0_only=r["alt_contact_sets"]["f0_only"]["qualifies"],
                         n_v2=r["v2_rigid_check"]["n"], q_v2=r["v2_rigid_check"]["qualifies"],
                         hca_opus=sum(r["sides"][s]["retarget_hca_opus"] for s in ("left", "right")),
                         hca_f0=sum(r["sides"][s]["retarget_hca_f0"] for s in ("left", "right")),
                         watertight=r["mesh"]["watertight"]))
    (OUT / "summary.json").write_text(json.dumps(rows, indent=1))
    keys = list(rows[0].keys()) if rows else []
    with open(OUT / "summary.tsv", "w") as f:
        f.write("\t".join(keys) + "\n")
        for r in rows:
            f.write("\t".join(str(r[k]) for k in keys) + "\n")
    for r in rows:
        print("\t".join(str(r[k]) for k in keys))


if __name__ == "__main__":
    main()
