#!/usr/bin/env python
"""t3f0_supports.py -- Tier-3 f0 prep: support USDA files for the 14 support episodes from Han Zhou's support release
(tier3_frame0_supports_20261024/supports/<seq>/: support.json + support_mesh.glb, read from the byte copies in
data/human_motion_data_tier3_f0/han_release_docs/supports/). NEW FILE (2026-10-24). CPU only. GT-free: uses Han's
video-fit furniture poses (T_world_mesh_contact, top_box), the f0 object track (whose frame-0 pose is the allowed true
first-frame pose) and the task text; no Tier-1/Tier-2 support value and no clean trajectory is read.

Trainer format (code/fc_r2 @ 61fa5d3 flash_chord/scene/support.py add_support_surfaces; released files use the same
schema): Z-up, metersPerUnit 1 stage, default prim /support_surfaces (Xform) with Cube children (PhysicsCollisionAPI,
size 1, xformOp:translate = box centre in the G1 parquet world, xformOp:scale = full edge lengths). The loader IGNORES any
rotation, so every box is axis-aligned in the G1 world; rotated / round shapes are written as inscribed axis-aligned
strips (width <= 4 cm). Placement: Han's pose -> G1 world by the retarget's static frame chain M (t3f0_frames.py,
validated against every parquet by t3f0_rcheck.py G10). Han's furniture is upright in the G1 world (tilt 0.000 deg).

Rules (fixed before the physical checks; all numbers are recorded in tmp_analysis/tier3/f0/supports/epNN.json):
  TOP (round_table 02 03 06 07 17 20 25 -> disk of radius min(top_box half extents xy) about the top_box centre;
       white_desk 11 13 16 27 and skinny_wood_chair seat 09 -> the oriented top_box rectangle): 3 cm thick slab.
       Height = Han's contact top (top_box face at T_world_mesh + contact_shift_m along up = the top of the mesh at
       T_world_mesh_contact) REFIT by an anchor-contact residual measured in the trainer's world with the trainer's object
       mesh: top z := median over the anchor run of the object's lowest convex-hull vertex. Anchor run = the still run
       that starts at frame 0 on the top (frame 0 = the true first-frame pose), else the LAST still run on the top
       (ep17 cone, ep27 basket start on the floor). Anchor footprint: if < 95% of the object's contact points (lowest
       1 cm) at the anchor mid frame lie inside the top shrunk by 1 cm, the top is translated horizontally by the smallest
       such vector (cap 10 cm). Reason for the height refit: with Han's shift as given the anchor runs sit 0.3-2.5 cm INSIDE
       the top in the trainer world (Han's own gap estimate is ~0.5-1.5 cm higher than the lowest hull vertex).
  STOOL (tall_bar_stool, leaning: 01 vacuum, 28 squeegee): the stool mesh at T_world_mesh_contact (G1 world) voxelised at
       2 cm (surface voxels + hole filling), kept only within 10 cm of the object at any still frame (the stool parts the
       object leans on; 6 / 4 cm if needed to keep <= 40 boxes), CARVED by the object's reference path (a cell is removed if, at ANY frame, an object surface sample
       lies within 0.5 cm of the cell or the cell centre lies inside the object's filled 1 cm occupancy: the support never
       obstructs the reference object path; needed for ep28, whose frame-0 squeegee blade lies inside Han's stool mesh),
       then greedily merged into axis-aligned boxes; (voxel, near) = the first of STOOL_CONFIGS giving <= 40 boxes with the
       frame-0 contact in [0, 1] cm and the path >= -0.5 cm from every box. Anchor-contact residual: at the frame-0
       rest pose the signed distance between the object surface and the box union must be in [0, 1] cm; otherwise the
       boxes are translated HORIZONTALLY along the closest-approach direction until it is +0.2 cm (no vertical move: the
       stool stands on the floor), and kept only if the reference path then stays >= -0.5 cm from every box.
Usage: t3f0_supports.py EP [EP ...] | all     (all = the 14 + SCENE_SUPPORT.json + SUPPORT_FILES.sha256 + parse checks)
"""
import os, sys, json, hashlib, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import numpy as np
import trimesh
from scipy import ndimage

ME = Path("/root/autodl-tmp/haoran.zhang")
ROOT = ME / "data/human_motion_data_tier3_f0"
DOCS = ROOT / "han_release_docs"
STAGE = ROOT / "whole_body/reconstructed_stage"
A = ME / "tmp_analysis/tier3/f0"
OUT = A / "supports"
SUPPORT_EPS = (1, 2, 3, 6, 7, 9, 11, 13, 16, 17, 20, 25, 27, 28)
STRIP_W = 0.04
THICK = 0.03
STILL_V = 0.02      # m/s (30 fps track)
STILL_MIN = 10      # frames
ON_TOP_BAND = 0.08  # |hull bottom - Han contact top| for a run to count as resting on the top
VOX = 0.02
NEAR = 0.04
STOOL_CONFIGS = ((0.02, 0.10), (0.03, 0.10), (0.02, 0.06), (0.03, 0.06), (0.02, 0.04), (0.03, 0.04), (0.04, 0.04))
MAX_STOOL_BOXES = 40
CARVE_PAD = 0.005
OBJ_VOX = 0.01
LEAN_TARGET = 0.002
FOOT_MARGIN = 0.01
FOOT_MAX = 0.10


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_glb(p):
    m = trimesh.load(str(p), force="scene")
    return m.dump(concatenate=True) if isinstance(m, trimesh.Scene) else m


def episodes():
    return {int(e["episode_index"]): e for e in json.loads((DOCS / "episodes.json").read_text())}


def still_runs(P):
    sp = np.r_[0.0, np.linalg.norm(np.diff(P, axis=0), axis=1) * 30.0]
    st = sp < STILL_V
    runs, a = [], None
    for t in range(len(P) + 1):
        f = t < len(P) and st[t]
        if f and a is None:
            a = t
        if not f and a is not None:
            if t - a >= STILL_MIN:
                runs.append((a, t))
            a = None
    return runs, sp


def box_sdf(pts, c, half):
    q = np.abs(pts - c) - half
    return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(np.max(q, axis=-1), 0.0)


def union_sdf(pts, boxes):
    return np.min(np.stack([box_sdf(pts, np.asarray(b["center"]), np.asarray(b["dims"]) / 2) for b in boxes]), axis=0)


def strips_disk(cx, cy, r, z_top, name):
    """inscribed axis-aligned strips (along y, sliced in x) of a disk."""
    n = int(np.ceil(2 * r / STRIP_W)); w = 2 * r / n
    out = []
    for k in range(n):
        x0, x1 = -r + k * w, -r + (k + 1) * w
        m = max(abs(x0), abs(x1))
        if m >= r:
            continue
        hy = np.sqrt(r * r - m * m)
        if 2 * hy < 0.02:
            continue
        out.append(dict(name=f"{name}_s{k:02d}", center=[cx + (x0 + x1) / 2, cy, z_top - THICK / 2], dims=[w, 2 * hy, THICK]))
    return out


def strips_rect(c, u, v, hu, hv, z_top, name):
    """inscribed axis-aligned strips of the rotated rectangle {c + a u + b v : |a|<=hu, |b|<=hv} (u, v unit, xy)."""
    corners = np.array([c + su * hu * u + sv * hv * v for su in (-1, 1) for sv in (-1, 1)])
    best = None
    for ax in (0, 1):  # slice along x (strips long in y) or along y
        o = 1 - ax
        lo, hi = corners[:, ax].min(), corners[:, ax].max()
        n = int(np.ceil((hi - lo) / STRIP_W)); w = (hi - lo) / n

        def interval(s):
            # {t : point with coord ax = s, coord o = t inside}: |(p-c).u| <= hu, |(p-c).v| <= hv
            tlo, thi = -np.inf, np.inf
            for e, h in ((u, hu), (v, hv)):
                # (s - c[ax]) e[ax] + (t - c[o]) e[o] in [-h, h]
                base = (s - c[ax]) * e[ax]
                if abs(e[o]) < 1e-12:
                    if abs(base) > h:
                        return None
                    continue
                a1, a2 = (-h - base) / e[o] + c[o], (h - base) / e[o] + c[o]
                tlo, thi = max(tlo, min(a1, a2)), min(thi, max(a1, a2))
            return (tlo, thi) if thi > tlo else None
        out = []
        for k in range(n):
            s0, s1 = lo + k * w, lo + (k + 1) * w
            i0, i1 = interval(s0 + 1e-9), interval(s1 - 1e-9)
            if i0 is None or i1 is None:
                continue
            tlo, thi = max(i0[0], i1[0]), min(i0[1], i1[1])
            if thi - tlo < 0.02:
                continue
            cen = [0.0, 0.0, z_top - THICK / 2]; dims = [0.0, 0.0, THICK]
            cen[ax], cen[o] = (s0 + s1) / 2, (tlo + thi) / 2
            dims[ax], dims[o] = w, thi - tlo
            out.append(dict(name=f"{name}_s{k:02d}", center=cen, dims=dims))
        area = sum(b["dims"][0] * b["dims"][1] for b in out)
        if best is None or area > best[0] + 1e-9 or (abs(area - best[0]) < 1e-9 and len(out) < len(best[1])):
            best = (area, out, ax)
    return best[1], 4 * hu * hv, best[0]


def greedy(occ):
    occ = occ.copy(); boxes = []
    nx, ny, nz = occ.shape
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                if not occ[x, y, z]:
                    continue
                x1 = x
                while x1 + 1 < nx and occ[x1 + 1, y, z]:
                    x1 += 1
                y1 = y
                while y1 + 1 < ny and occ[x:x1 + 1, y1 + 1, z].all():
                    y1 += 1
                z1 = z
                while z1 + 1 < nz and occ[x:x1 + 1, y:y1 + 1, z1 + 1].all():
                    z1 += 1
                occ[x:x1 + 1, y:y1 + 1, z:z1 + 1] = False
                boxes.append((x, x1, y, y1, z, z1))
    return boxes


def write_usda(out, boxes, ep, note):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    out = Path(out)
    if out.exists():
        out.unlink()
    stage = Usd.Stage.CreateNew(str(out))
    stage.SetMetadata("metersPerUnit", 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.GetRootLayer().customLayerData = {"t3f0": note}
    root = UsdGeom.Xform.Define(stage, "/support_surfaces")
    stage.SetDefaultPrim(root.GetPrim())
    for b in boxes:
        cube = UsdGeom.Cube.Define(stage, f"/support_surfaces/{b['name']}")
        cube.CreateSizeAttr(1.0)
        cube.CreateDisplayColorAttr([Gf.Vec3f(0.80, 0.72, 0.55)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(x) for x in b["center"]]))
        xf.AddScaleOp().Set(Gf.Vec3f(*[float(x) for x in b["dims"]]))
    stage.GetRootLayer().Save()


def build(ep):
    if (ROOT / "READY").exists():
        raise SystemExit("REFUSED: data root READY (write-once)")
    E = episodes()[ep]
    seq = E["sequence_id"]
    sdir = DOCS / "supports" / seq
    sj = json.loads((sdir / "support.json").read_text())
    fr = np.load(A / "frames" / f"ep{ep:02d}.npz")
    fmeta = json.loads((A / "frames" / f"ep{ep:02d}.json").read_text())
    M = fr["M_cv_to_g1"]
    up = M[:3, :3] @ np.array(sj["up_world"])
    tilt = float(np.degrees(np.arccos(min(1.0, up[2]))))
    assert tilt < 0.05, f"ep{ep}: support up axis tilted {tilt} deg in the G1 world"
    om = trimesh.load(str(A / "frames" / f"ep{ep:02d}_obj" / "textured_mesh.obj"), force="mesh", process=False)
    hull = np.asarray(om.convex_hull.vertices)
    OS, _ = trimesh.sample.sample_surface(om, 20000, seed=0)
    OS = np.vstack([OS, hull])
    P, Rm = fr["obj_pos"], fr["obj_R"]
    T = len(P)
    runs, speed = still_runs(P)
    bottom = np.array([(hull @ Rm[t].T + P[t])[:, 2].min() for t in range(T)])
    rec = dict(ep=ep, seq=seq, object=E["object"], task=E["task"], frames=T, support=sj["support"], han_iou=sj["iou"],
               han_contact_shift_cm=round(100 * sj["contact_shift_m"], 3), up_tilt_deg=tilt,
               han_support_json_sha256=sha(sdir / "support.json"), han_mesh_sha256=sha(sdir / "support_mesh.glb"),
               frames_inputs=fmeta["inputs"], object_lift_cm=round(100 * fmeta["lift_applied"], 3),
               still_runs=[list(r) for r in runs])
    tb = sj["top_box"]
    ax = np.array(tb["axes"])  # columns = box axes (column 2 = up_world)
    c_g = M[:3, :3] @ np.array(tb["centre"]) + M[:3, 3]
    ax_g = M[:3, :3] @ ax
    he = np.array(tb["half_extents"])
    face_vf = float(c_g[2] + he[2] * abs(ax_g[2, 2]))           # top_box face at the video-fit pose
    z_han = face_vf + float(sj["contact_shift_m"]) * float(up[2])  # Han's contact top
    if sj["support"] in ("round_table", "white_desk", "skinny_wood_chair"):
        kind = "disk" if sj["support"] == "round_table" else "rect"
        u = ax_g[:2, 0] / np.linalg.norm(ax_g[:2, 0]); v = ax_g[:2, 1] / np.linalg.norm(ax_g[:2, 1])

        def inside(xy):
            d = xy - c_g[:2]
            if kind == "disk":
                return np.linalg.norm(d, axis=-1) <= min(he[0], he[1])
            return (np.abs(d @ u) <= he[0]) & (np.abs(d @ v) <= he[1])
        on = []
        for a, b in runs:
            m = (a + b) // 2
            W = OS @ Rm[m].T + P[m]
            low = W[W[:, 2] < W[:, 2].min() + 0.01]
            ok = abs(float(np.median(bottom[a:b])) - z_han) < ON_TOP_BAND and inside(low[:, :2]).mean() > 0.5
            on.append(ok)
        top_runs = [r for r, ok in zip(runs, on) if ok]
        anchor = None
        if top_runs and top_runs[0][0] == 0:
            anchor, why = top_runs[0], "still run from frame 0 on the top (frame 0 = true first-frame pose)"
        elif top_runs:
            anchor, why = top_runs[-1], "last still run on the top (the object starts off the support)"
        z_top = float(np.median(bottom[anchor[0]:anchor[1]])) if anchor else z_han
        # anchor footprint: the object's contact points (lowest 1 cm) at the anchor mid frame must lie inside the top
        # shrunk by FOOT_MARGIN; if fewer than 95% do, the top is translated horizontally by the smallest vector that
        # achieves it (cap FOOT_MAX; above the cap nothing is moved and the episode is flagged)
        foot = dict(applied=False, shift_xy_cm=[0.0, 0.0])
        if anchor:
            m_ = (anchor[0] + anchor[1]) // 2
            W = OS @ Rm[m_].T + P[m_]
            low = W[W[:, 2] < W[:, 2].min() + 0.01][:, :2]
            cov0 = float(inside(low).mean())
            foot["anchor_cover_before"] = round(cov0, 3)
            if cov0 < 0.95:
                d = low - c_g[:2]
                if kind == "disk":
                    rr = min(he[0], he[1]) - FOOT_MARGIN
                    sh = np.zeros(2)
                    for _ in range(200):
                        dist = np.linalg.norm(d - sh, axis=1); k = int(np.argmax(dist))
                        if dist[k] <= rr + 1e-5:
                            break
                        sh += (d[k] - sh) / dist[k] * (dist[k] - rr) * 0.5
                else:
                    sh = np.zeros(2)
                    for e_, h_ in ((u, he[0]), (v, he[1])):
                        a_ = d @ e_
                        lo_, hi_ = a_.max() - (h_ - FOOT_MARGIN), a_.min() + (h_ - FOOT_MARGIN)
                        t_ = 0.0 if lo_ <= 0.0 <= hi_ else (lo_ if lo_ > 0 else hi_)
                        sh += t_ * e_
                if np.linalg.norm(sh) <= FOOT_MAX:
                    c_g = c_g.copy(); c_g[:2] += sh
                    foot.update(applied=True, shift_xy_cm=np.round(100 * sh, 2).tolist(), anchor_cover_after=round(float(inside(low).mean()), 3))
                else:
                    foot.update(note=f"needed shift {100 * np.linalg.norm(sh):.1f} cm > cap {100 * FOOT_MAX:.0f} cm: not applied")
        if kind == "disk":
            r = float(min(he[0], he[1]))
            boxes = strips_disk(float(c_g[0]), float(c_g[1]), r, z_top, "tabletop")
            area_true, area_boxes = np.pi * r * r, sum(b["dims"][0] * b["dims"][1] for b in boxes)
            shape = dict(kind="disk", centre_xy=c_g[:2].tolist(), radius_m=r)
        else:
            name = "seat" if sj["support"] == "skinny_wood_chair" else "desktop"
            boxes, area_true, area_boxes = strips_rect(c_g[:2], u, v, float(he[0]), float(he[1]), z_top, name)
            shape = dict(kind="rect", centre_xy=c_g[:2].tolist(), half_extents_m=he[:2].tolist(),
                         yaw_deg=float(np.degrees(np.arctan2(u[1], u[0]))))
        rec.update(kind="top", shape=shape, top_runs=[list(r) for r in top_runs], anchor_run=list(anchor) if anchor else None,
                   anchor_rule=why if anchor else "no still run on the top: Han's contact top kept",
                   z_top_videofit_m=face_vf, z_top_han_contact_m=z_han, z_top_m=z_top,
                   refit_residual_cm=round(100 * (z_top - z_han), 3), total_shift_vs_videofit_cm=round(100 * (z_top - face_vf), 3),
                   strip_area_fraction=round(area_boxes / area_true, 4), anchor_footprint=foot,
                   run_gap_cm=[dict(run=list(r), on_top=bool(ok), bottom_gap_cm=round(100 * (float(np.median(bottom[r[0]:r[1]])) - z_top), 2))
                               for r, ok in zip(runs, on)])
    else:  # tall_bar_stool, leaning
        def stool(VOX, NEAR):
            st = load_glb(sdir / "support_mesh.glb")
            st.apply_transform(M @ np.array(sj["T_world_mesh_contact"]))
            vg = st.voxelized(VOX)
            occ = ndimage.binary_fill_holes(vg.matrix)
            idx = np.argwhere(occ)
            cen = trimesh.transform_points(idx.astype(float), vg.transform)  # voxel centres (G1 world)
            # parts the object can lean on: voxels within NEAR of the object at any still frame
            from scipy.spatial import cKDTree
            pts = np.vstack([OS[::4] @ Rm[t].T + P[t] for a, b in runs for t in range(a, b, max(1, (b - a) // 5))])
            d, _ = cKDTree(pts).query(cen)
            keep = np.zeros_like(occ)
            sel = idx[d <= NEAR]
            keep[sel[:, 0], sel[:, 1], sel[:, 2]] = True
            org, pitch = vg.transform[:3, 3], float(np.ravel(vg.pitch)[0])
            # carve the reference object path: (a) cells within CARVE_PAD of an object surface sample at any frame
            carve = np.zeros_like(occ)
            shp = np.array(occ.shape)
            sub = OS[::3]
            for t in range(T):
                W = sub @ Rm[t].T + P[t]
                for dx in (-CARVE_PAD, CARVE_PAD):
                    for dy in (-CARVE_PAD, CARVE_PAD):
                        for dz in (-CARVE_PAD, CARVE_PAD):
                            ijk = np.floor((W + np.array([dx, dy, dz]) - org) / pitch + 0.5).astype(int)
                            okb = np.all((ijk >= 0) & (ijk < shp), axis=1)
                            carve[ijk[okb, 0], ijk[okb, 1], ijk[okb, 2]] = True
            # (b) cell centres inside the object's filled occupancy at any frame
            ovg = om.voxelized(OBJ_VOX)
            oocc = ndimage.binary_fill_holes(ovg.matrix)
            oinv = np.linalg.inv(ovg.transform)
            cand = np.argwhere(keep & ~carve)
            cc = trimesh.transform_points(cand.astype(float), vg.transform)
            inside_any = np.zeros(len(cand), bool)
            oshp = np.array(oocc.shape)
            for t in range(T):
                loc = (cc - P[t]) @ Rm[t]  # world -> object local
                ijk = np.round(trimesh.transform_points(loc, oinv)).astype(int)
                okb = np.all((ijk >= 0) & (ijk < oshp), axis=1)
                hit = np.zeros(len(cand), bool); hit[okb] = oocc[ijk[okb, 0], ijk[okb, 1], ijk[okb, 2]]
                inside_any |= hit
            carve[cand[inside_any, 0], cand[inside_any, 1], cand[inside_any, 2]] = True
            n_carved = int((keep & carve).sum())
            keep &= ~carve
            gb = greedy(keep)
            boxes = []
            for k, (x0, x1, y0, y1, z0, z1) in enumerate(gb):
                lo = org + pitch * (np.array([x0, y0, z0]) - 0.5); hi = org + pitch * (np.array([x1, y1, z1]) + 0.5)
                lo[2] = max(lo[2], 0.0)
                boxes.append(dict(name=f"stool_b{k:03d}", center=((lo + hi) / 2).tolist(), dims=(hi - lo).tolist()))

            def path_min_sdf(bx, stride=1):
                return min(float(union_sdf(sub @ Rm[t].T + P[t], bx).min()) for t in range(0, T, stride))
            # anchor-contact residual at the frame-0 rest (only if nothing was carved)
            anchor = runs[0] if runs and runs[0][0] == 0 else None
            shift = np.zeros(3)
            sd0 = None
            shift_note = "none"
            lsp = OUT / f"ep{ep:02d}_lean_search.json"
            lsv = json.loads(lsp.read_text()).get("chosen_shift_cm") if lsp.is_file() else None
            if anchor and lsv:
                # settle-verified correction from scripts/t3f0_lean_search.py (relative to Han's pose)
                W0 = OS @ Rm[0].T + P[0]
                sd0 = float(union_sdf(W0, boxes).min())
                v = np.asarray(lsv, float) / 100.0
                bx = [dict(b, center=(np.asarray(b["center"]) + v).tolist()) for b in boxes]
                m0, mp = float(union_sdf(W0, bx).min()), path_min_sdf(bx, 3)
                if 0.0 <= m0 <= 0.01 and mp >= -0.005:
                    shift, boxes = v, bx
                    shift_note = (f"lean_search (settle-verified, {lsp.name}): frame-0 min sdf {100 * m0:.2f} cm, "
                                  f"path min sdf {100 * mp:.2f} cm")
                else:
                    shift_note = f"lean_search shift REJECTED here (frame-0 {100 * m0:.2f} cm, path {100 * mp:.2f} cm): Han's pose kept"
            elif anchor:
                W0 = OS @ Rm[0].T + P[0]
                sd = union_sdf(W0, boxes); sd0 = float(sd.min())
                if not (0.0 <= sd0 <= 0.01):
                    j = int(np.argmin(sd))
                    bi = int(np.argmin([box_sdf(W0[j:j + 1], np.asarray(b["center"]), np.asarray(b["dims"]) / 2)[0] for b in boxes]))
                    cb, hb = np.asarray(boxes[bi]["center"]), np.asarray(boxes[bi]["dims"]) / 2
                    near = np.clip(W0[j], cb - hb, cb + hb)
                    dvec = near - W0[j]; dvec[2] = 0.0
                    if np.linalg.norm(dvec) < 1e-6:
                        dvec = cb - W0[j]; dvec[2] = 0.0
                    dvec /= np.linalg.norm(dvec)
                    sgn = -1.0 if sd0 > 0.01 else 1.0
                    lo_s, hi_s = 0.0, 0.10
                    for _ in range(40):
                        s_ = (lo_s + hi_s) / 2
                        bx = [dict(b, center=(np.asarray(b["center"]) + sgn * s_ * dvec).tolist()) for b in boxes]
                        m_ = float(union_sdf(W0, bx).min())
                        if (m_ > LEAN_TARGET) == (sgn < 0):
                            lo_s = s_
                        else:
                            hi_s = s_
                    cand_shift = sgn * hi_s * dvec
                    bx = [dict(b, center=(np.asarray(b["center"]) + cand_shift).tolist()) for b in boxes]
                    m0, mp = float(union_sdf(W0, bx).min()), path_min_sdf(bx, 3)
                    if 0.0 <= m0 <= 0.01 and mp >= -0.005:
                        shift, boxes, shift_note = cand_shift, bx, f"applied: frame-0 min sdf {100 * m0:.2f} cm, path min sdf {100 * mp:.2f} cm"
                    else:
                        shift_note = f"REJECTED (frame-0 min sdf {100 * m0:.2f} cm, path min sdf {100 * mp:.2f} cm): kept Han's pose"
            sd0_after = None if not anchor else round(100 * float(union_sdf(OS @ Rm[0].T + P[0], boxes).min()), 2)
            sdf_runs = []
            for a, b in runs:
                m = (a + b) // 2
                sdf_runs.append(dict(run=[a, b], min_sdf_cm=round(100 * float(union_sdf(OS @ Rm[m].T + P[m], boxes).min()), 2),
                                     bottom_z_cm=round(100 * float(np.median(bottom[a:b])), 2)))
            info = dict(kind="stool", voxel_m=VOX, near_m=NEAR, carve_pad_m=CARVE_PAD, n_voxels_total=int(occ.sum()),
                       n_voxels_carved=n_carved, n_voxels_kept=int(keep.sum()),
                       anchor_run=list(anchor) if anchor else None, frame0_min_sdf_before_cm=None if sd0 is None else round(100 * sd0, 2),
                       frame0_min_sdf_after_cm=sd0_after, shift_note=shift_note,
                       horizontal_shift_cm=np.round(100 * shift, 2).tolist(), horizontal_shift_norm_cm=round(100 * float(np.linalg.norm(shift)), 2),
                       path_min_sdf_cm=round(100 * path_min_sdf(boxes, 1), 2), run_min_sdf_cm=sdf_runs)
            return boxes, info
        tried = []
        for VOX_, NEAR_ in STOOL_CONFIGS:  # first config with <= MAX_STOOL_BOXES boxes and frame-0 contact in [0, 1] cm
            boxes, info = stool(VOX_, NEAR_)
            f0 = info['frame0_min_sdf_after_cm']
            tried.append(dict(voxel_m=VOX_, near_m=NEAR_, n_boxes=len(boxes), frame0_min_sdf_after_cm=f0, path_min_sdf_cm=info['path_min_sdf_cm']))
            if len(boxes) <= MAX_STOOL_BOXES and f0 is not None and 0.0 <= f0 <= 1.0 and info['path_min_sdf_cm'] >= -0.5:
                break
        info['configs_tried'] = tried
        rec.update(info)
    rec["boxes"] = boxes
    rec["n_boxes"] = len(boxes)
    STAGE.mkdir(parents=True, exist_ok=True)
    u = STAGE / f"{seq}_support.usda"
    write_usda(u, boxes, ep, f"Tier-3 f0 support ep{ep:02d} ({sj['support']}) from Han Zhou's support release "
               f"(T_world_mesh_contact/top_box/mesh) + anchor-contact residual (scripts/t3f0_supports.py); axis-aligned boxes "
               f"in the G1 parquet world")
    rec["support_file"] = f"whole_body/reconstructed_stage/{u.name}"
    rec["support_sha256"] = sha(u)
    rec["written"] = time.strftime("%F %T")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"ep{ep:02d}.json").write_text(json.dumps(rec, indent=1, default=float))
    brief = {k: rec.get(k) for k in ("ep", "support", "kind", "n_boxes", "anchor_run", "refit_residual_cm", "total_shift_vs_videofit_cm",
                                      "strip_area_fraction", "frame0_min_sdf_before_cm", "frame0_min_sdf_after_cm",
                                      "n_voxels_carved", "path_min_sdf_cm", "horizontal_shift_norm_cm")}
    print(json.dumps(brief, default=float), flush=True)
    return rec


def finish():
    sys.path.insert(0, str(ME / "code/fc_r2/src"))
    import newton
    from flash_chord.scene.support import add_support_surfaces
    from flash_chord.assets.registry import support_usda_for_parquet
    E = episodes()
    scene, lines, checks = {}, [], []
    for ep in range(30):
        seq = E[ep]["sequence_id"]
        u = STAGE / f"{seq}_support.usda"
        future = ROOT / "whole_body/soma" / f"sequence_id={seq}" / "robot_name=g1" / "data.parquet"
        got = support_usda_for_parquet(str(future), robot_name="g1")
        if ep in SUPPORT_EPS:
            rec = json.loads((OUT / f"ep{ep:02d}.json").read_text())
            ok_sha = sha(u) == rec["support_sha256"]
            b = newton.ModelBuilder()
            binding = add_support_surfaces(b, u)
            ncube = sum(1 for line in u.read_text().splitlines() if line.strip().startswith('def Cube "'))
            checks.append((f"ep{ep:02d}", ok_sha and got is not None and Path(got) == u and len(binding.shapes) == ncube == rec["n_boxes"],
                           f"sha {ok_sha}; trainer resolves {got}; trainer parses {len(binding.shapes)} shapes = {ncube} cubes = {rec['n_boxes']} boxes"))
            lines.append(f"{rec['support_sha256']}  whole_body/reconstructed_stage/{u.name}")
            scene[f"{ep:02d}"] = dict(sequence_id=seq, scene_support=True, support_file=rec["support_file"],
                                      support_sha256=rec["support_sha256"], support=rec["support"], kind=rec["kind"],
                                      n_boxes=rec["n_boxes"], detail=f"tmp_analysis/tier3/f0/supports/ep{ep:02d}.json")
        else:
            txt = u.read_text()
            geo = "def Cube" in txt or "def Cylinder" in txt
            checks.append((f"ep{ep:02d}", not geo and got is not None and Path(got) == u, f"floor-only geometry {geo}; resolves {got}"))
            scene[f"{ep:02d}"] = dict(sequence_id=seq, scene_support=False, support_file=f"whole_body/reconstructed_stage/{u.name}",
                                      support_sha256=sha(u), kind="floor-only")
    bad = [c for c in checks if not c[1]]
    for c in checks:
        print(("ok   " if c[1] else "FAIL ") + c[0] + " " + c[2])
    if bad:
        raise SystemExit(f"{len(bad)} support checks failed")
    (ROOT / "SUPPORT_FILES.sha256").write_text("\n".join(lines) + "\n")
    (ROOT / "SCENE_SUPPORT.json").write_text(json.dumps(dict(
        note="Tier-3 f0 scene table. scene_support=true -> train with scene.support=true (the trainer resolves "
             "whole_body/reconstructed_stage/<seq>_support.usda from the parquet path); false -> floor-only (geometry-free "
             "file, scene.support=false). Supports: scripts/t3f0_supports.py from Han Zhou's support release.",
        episodes=scene), indent=1) + "\n")
    print("SCENE_SUPPORT.json + SUPPORT_FILES.sha256 written")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["all"]:
        for e in SUPPORT_EPS:
            build(e)
        finish()
    elif a == ["finish"]:
        finish()
    else:
        for e in a:
            build(int(e))
