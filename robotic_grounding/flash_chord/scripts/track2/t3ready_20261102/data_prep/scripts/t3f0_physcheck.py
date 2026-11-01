#!/usr/bin/env python
"""t3f0_physcheck.py -- Tier-3 f0 prep: physical checks of the 14 support files (CPU only, no GPU). NEW FILE (2026-10-24);
the settle and G1 parts are copies of scripts/t3sup_settle.py / t3sup_g1_check.py (left untouched) with the f0 paths.
No clean data, no Tier-1/Tier-2 support values. Writes only tmp_analysis/tier3/f0/physcheck/.

  rest EP    object vs support from the reference track (trainer world): per still run the bottom gap (tops) / min signed
             distance (stools) and the footprint coverage (share of the object's lowest points that lie over a support
             box); the whole reference path's min signed distance to the boxes (summary: frames deeper than 1 cm); the
             SOMA human's min distance to the object at the run's mid frame (a run with a hand within 3 cm is held).
  human EP   the SOMA human (all mesh vertices, t3f0_frames.py) vs the boxes, every frame. A vertex violates when it is
             inside a box deeper than 2 cm AND more than 2.5 cm below that box's top (resting a hand on a top is fine).
  g1 EP      the trainer's own scene (code/fc_r2 @ 61fa5d3, experiment g1_recon_body_coord_rsi_r2, collision 'physical' =
             robot_support_collision true, the support USDA through add_support_surfaces) posed at every reference frame by
             the trainer's RSI reset; every robot collision shape vs every box; violation = deeper than 1 cm (non-hand links)
             / 2 cm (hand links) AND more than 2.5 cm below the box top (t3sup_g1_check rule).
  settle EP  passive settle of the object ALONE (MuJoCo-Warp on the Warp CPU device, training sim settings, 10 s) spawned at
             the mid frame of every still run that rests on / leans against the support; control: the anchor run without
             the support (must move). PASS: final |dp| <= 2 cm, final rotation <= 5 deg, max |dp| <= 3 cm (t3sup rule).
  summary    physcheck_summary.json + flags per episode.
Usage: t3f0_physcheck.py rest|human|g1|settle EP [EP ...] | summary
"""
import os, sys, json, time, hashlib
from pathlib import Path
ME = Path("/root/autodl-tmp/haoran.zhang")
A = ME / "tmp_analysis/tier3/f0"
OUT = A / "physcheck"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["FLASH_CHORD_CACHE_DIR"] = str(A / "convex_cache")
sys.path.insert(0, str(ME / "code/fc_r2/src"))
import numpy as np
import trimesh

ROOT = ME / "data/human_motion_data_tier3_f0"
SUPPORT_EPS = (1, 2, 3, 6, 7, 9, 11, 13, 16, 17, 20, 25, 27, 28)
BODY_THR, HAND_THR, TOP_BAND, HUMAN_THR = 0.01, 0.02, 0.025, 0.02


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def box_sdf(pts, c, half):
    q = np.abs(pts - c) - half
    return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(np.max(q, axis=-1), 0.0)


def load(ep):
    rec = json.loads((A / "supports" / f"ep{ep:02d}.json").read_text())
    fr = np.load(A / "frames" / f"ep{ep:02d}.npz")
    boxes = [(np.asarray(b["center"]), np.asarray(b["dims"]) / 2) for b in rec["boxes"]]
    return rec, fr, boxes


def write(name, ep, res):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}_ep{ep:02d}.json").write_text(json.dumps(res, indent=1, default=float))


def rest(ep):
    rec, fr, boxes = load(ep)
    om = trimesh.load(str(A / "frames" / f"ep{ep:02d}_obj" / "textured_mesh.obj"), force="mesh", process=False)
    OS, _ = trimesh.sample.sample_surface(om, 8000, seed=0)
    OS = np.vstack([OS, om.convex_hull.vertices])
    P, Rm = fr["obj_pos"], fr["obj_R"]
    T = len(P)
    sdf = np.empty(T)
    for t in range(T):
        W = OS @ Rm[t].T + P[t]
        sdf[t] = min(float(box_sdf(W, c, h).min()) for c, h in boxes)
    from scipy.spatial import cKDTree
    HV = fr["verts_w"] if "verts_w" in fr.files else None
    runs = []
    for a, b in rec["still_runs"]:
        m = (a + b) // 2
        W = OS @ Rm[m].T + P[m]
        low = W[W[:, 2] < W[:, 2].min() + 0.01]
        cover = float(np.mean([any(np.all(np.abs(p[:2] - c[:2]) <= h[:2]) for c, h in boxes) for p in low]))
        held = None if HV is None else round(100 * float(cKDTree(HV[m].astype(np.float64)).query(W)[0].min()), 2)
        runs.append(dict(run=[a, b], held_min_dist_cm=held, min_sdf_cm=round(100 * float(sdf[a:b].min()), 2), median_sdf_cm=round(100 * float(np.median(sdf[a:b])), 2),
                         bottom_z_m=round(float(W[:, 2].min()), 4), footprint_cover=round(cover, 3),
                         near_support=bool(np.median(sdf[a:b]) < 0.03)))
    deep = np.where(sdf < -0.015)[0]
    res = dict(ep=ep, kind=rec["kind"], frames=T, path_min_sdf_cm=round(100 * float(sdf.min()), 2),
               path_min_sdf_frame=int(np.argmin(sdf)), frames_deeper_1p5cm=int(len(deep)),
               deep_runs=[[int(x[0]), int(x[-1])] for x in np.split(deep, np.where(np.diff(deep) > 1)[0] + 1)] if len(deep) else [],
               runs=runs, per_frame_sdf_cm=np.round(100 * sdf, 2).tolist())
    write("rest", ep, res)
    print(json.dumps({k: v for k, v in res.items() if k != "per_frame_sdf_cm"}), flush=True)


def human(ep):
    rec, fr, boxes = load(ep)
    V = fr["verts_w"]; J = fr["joints_w"]; names = [str(x) for x in fr["joint_names"]]
    T = len(V)
    tops = [c[2] + h[2] for c, h in boxes]
    viol, minsd = [], np.full(T, np.inf)
    for t in range(T):
        worst = None
        for (c, h), top in zip(boxes, tops):
            sd = box_sdf(V[t].astype(np.float64), c, h)
            minsd[t] = min(minsd[t], float(sd.min()))
            bad = np.where((sd < -HUMAN_THR) & (V[t][:, 2] < top - TOP_BAND))[0]
            if len(bad):
                i = bad[np.argmin(sd[bad])]
                if worst is None or sd[i] < worst[0]:
                    worst = (float(sd[i]), int(i))
        if worst:
            j = int(np.argmin(np.linalg.norm(J[t] - V[t][worst[1]], axis=1)))
            viol.append(dict(frame=t, depth_cm=round(-100 * worst[0], 2), nearest_joint=names[j]))
    fr_ = np.array([v["frame"] for v in viol], int)
    res = dict(ep=ep, frames=T, rule=f"vertex inside a box deeper than {HUMAN_THR} m and > {TOP_BAND} m below its top",
               min_sdf_cm=round(100 * float(minsd.min()), 2), violating_frames=len(viol),
               violating_runs=[[int(x[0]), int(x[-1])] for x in np.split(fr_, np.where(np.diff(fr_) > 1)[0] + 1)] if len(fr_) else [],
               worst=sorted(viol, key=lambda v: -v["depth_cm"])[:10],
               joints_involved=sorted({v["nearest_joint"] for v in viol}))
    write("human", ep, res)
    print(json.dumps(res), flush=True)


def g1(ep):
    import warp as wp
    wp.config.quiet = True
    import newton
    from scipy.spatial.transform import Rotation as R
    from hydra import compose, initialize_config_dir
    from flash_chord.configuration import instantiate_typed
    from flash_chord.embodiments.base import Embodiment
    from flash_chord.scene.collision import CollisionPolicy
    from flash_chord.assets.registry import support_usda_for_parquet
    from flash_chord.data import load_reference
    from flash_chord.scene.builder import build_scene
    from flash_chord.lifecycle.reset import ReferenceResetTable, reset_worlds
    rec = json.loads((A / "supports" / f"ep{ep:02d}.json").read_text())
    parquet = str(ROOT / "whole_body/soma" / f"sequence_id={rec['seq']}" / "robot_name=g1/data.parquet")
    usda = support_usda_for_parquet(parquet, robot_name="g1")
    assert usda is not None and sha(usda) == rec["support_sha256"], "trainer does not resolve the recorded support file"
    with initialize_config_dir(config_dir=str(ME / "code/fc_r2/src/flash_chord/configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=["experiment=g1_recon_body_coord_rsi_r2", f"task.parquet='{parquet}'"])
    emb = instantiate_typed(cfg.embodiment, Embodiment)
    col = instantiate_typed(cfg.collision, CollisionPolicy)
    assert col.robot_support_collision, "experiment collision policy must collide robot and support"
    ref = load_reference(parquet, control_fps=float(cfg.sim.fps) if "sim" in cfg else 50.0, motion_speed=1.0,
                         source_frame_playback=bool(cfg.task.get("source_frame_playback", False)))
    scene = build_scene(emb, ref, world_count=1, support_usda=usda, collision=col, decompose_objects=True,
                        contact_friction=1.0, object_free_joint_damping=0.01)
    m = scene.model
    lay = scene.collision_layout
    stype = m.shape_type.numpy(); sbody = m.shape_body.numpy(); sxf = m.shape_transform.numpy(); sscale = m.shape_scale.numpy()
    labels = getattr(m, "body_label", None) or getattr(m, "body_key", None)
    boxes = []
    for s in range(lay.support.start, lay.support.stop):
        assert int(stype[s]) == int(newton.GeoType.BOX), "support shapes must be boxes"
        boxes.append((np.asarray(sxf[s][:3], float), np.asarray(sscale[s][:3], float)))
    assert len(boxes) == rec["n_boxes"], f"scene has {len(boxes)} support boxes, file {rec['n_boxes']}"
    tops = [c[2] + h[2] for c, h in boxes]
    shapes = []
    for s in range(lay.robot.start, lay.robot.stop):
        t_ = int(stype[s]); b = int(sbody[s]); name = str(labels[b]) if labels is not None else str(b)
        if t_ == int(newton.GeoType.SPHERE):
            loc, rad = np.zeros((1, 3)), float(sscale[s][0])
        elif t_ == int(newton.GeoType.CAPSULE):
            hh = float(sscale[s][1]); loc = np.c_[np.zeros(21), np.zeros(21), np.linspace(-hh, hh, 21)]; rad = float(sscale[s][0])
        elif t_ in (int(newton.GeoType.MESH), int(newton.GeoType.CONVEX_MESH)):
            loc = np.asarray(m.shape_source[s].vertices, float) * np.asarray(sscale[s][:3], float); rad = 0.0
        elif t_ == int(newton.GeoType.BOX):
            h3 = np.asarray(sscale[s][:3], float)
            g = np.linspace(-1, 1, 5)
            loc = np.array([[a, b_, c_] for a in g for b_ in g for c_ in g]) * h3; rad = 0.0
        else:
            raise SystemExit(f"unhandled robot shape type {t_}")
        hand = any(k in name.lower() for k in ("hand", "thumb", "index", "middle", "palm", "finger"))
        shapes.append((s, b, sxf[s], loc, rad, hand, name))
    table = ReferenceResetTable.build(scene, ref, device=m.device)
    state = m.state()
    T = ref.num_frames
    body_cl, hand_cl = np.full(T, np.inf), np.full(T, np.inf)
    viol = []
    for f in range(T):
        reset_worlds(m, table, state, frame_id=f)
        bq = state.body_q.numpy()
        worst = {}
        for s, b, xf, loc, rad, hand, name in shapes:
            Rb = R.from_quat(bq[b][3:7]); Rs = R.from_quat(np.asarray(xf[3:7], float))
            Pp = Rb.apply(Rs.apply(loc) + np.asarray(xf[:3], float)) + bq[b][:3]
            for (c, h), top in zip(boxes, tops):
                sd = box_sdf(Pp, c, h) - rad
                i = int(np.argmin(sd)); d = float(sd[i])
                if hand:
                    hand_cl[f] = min(hand_cl[f], d)
                else:
                    body_cl[f] = min(body_cl[f], d)
                thr = HAND_THR if hand else BODY_THR
                if d < -thr and Pp[i, 2] - rad < top - TOP_BAND and d < worst.get("d", 0):
                    worst = dict(frame=f, link=name, d=d, depth_cm=round(-100 * d, 2), hand=hand)
        if worst:
            worst.pop("d"); viol.append(worst)
    fr_ = np.array([v["frame"] for v in viol], int)
    res = dict(ep=ep, parquet=parquet, parquet_sha256=sha(parquet), usda=str(usda), frames=T, n_boxes=len(boxes),
               robot_shapes=len(shapes), min_body_clear_cm=round(100 * float(body_cl.min()), 2),
               min_hand_clear_cm=round(100 * float(hand_cl.min()), 2), violating_frames=len(viol),
               violating_runs=[[int(x[0]), int(x[-1])] for x in np.split(fr_, np.where(np.diff(fr_) > 1)[0] + 1)] if len(fr_) else [],
               worst=sorted(viol, key=lambda v: -v["depth_cm"])[:10], links_involved=sorted({v["link"] for v in viol}))
    write("g1", ep, res)
    print(json.dumps(res), flush=True)


URDF_TMPL = """<?xml version="1.0"?>
<robot name="retarget_object">
  <link name="object">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_path}"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_path}"/>
      </geometry>
    </collision>
  </link>
</robot>
"""


def settle(ep, usda_override=None, only_anchor=False, save=True):
    import warp as wp
    wp.config.quiet = True
    import newton
    from dataclasses import replace
    from scipy.spatial.transform import Rotation as R
    from flash_chord.scene.support import add_support_surfaces
    from flash_chord.scene.objects import _object_builder, _decompose_object_collision
    from flash_chord.scene.collision import ShapeSpan
    from flash_chord.runtime.sim import SimConfig, build_force_contacts
    SIM = SimConfig(fps=50.0, substeps=4, solver="newton", integrator="implicitfast", cone="pyramidal", njmax=1024,
                    nconmax=512, impratio=1.0, iterations=100, ls_iterations=50, use_mujoco_contacts=True)
    rec, fr, _ = load(ep)
    rr = json.loads((OUT / f"rest_ep{ep:02d}.json").read_text())
    usda = Path(usda_override) if usda_override else ROOT / rec["support_file"]
    if not usda_override:
        assert sha(usda) == rec["support_sha256"]
    mesh = A / "frames" / f"ep{ep:02d}_obj" / "textured_mesh.obj"
    pmesh = ROOT / "whole_body/soma" / f"sequence_id={rec['seq']}" / "object" / "textured_mesh.obj"
    same_mesh = pmesh.is_file() and sha(pmesh) == sha(mesh)
    urdf = OUT / f"settle_ep{ep:02d}_object.urdf"
    OUT.mkdir(parents=True, exist_ok=True)
    txt = URDF_TMPL.format(mesh_path=str(mesh.resolve()))
    if not urdf.is_file() or urdf.read_text() != txt:
        urdf.write_text(txt)

    def build(pos, quat_wxyz, with_support):
        cp = newton.ModelBuilder()
        cp.default_shape_cfg.ke = 1.0e3; cp.default_shape_cfg.kd = 1.0e2
        cp.default_shape_cfg.margin = 0.0; cp.default_shape_cfg.gap = 0.0
        newton.solvers.SolverMuJoCo.register_custom_attributes(cp)
        comp = _object_builder(cp)
        comp.add_urdf(source=str(urdf), floating=True, enable_self_collisions=False, collapse_fixed_joints=True,
                      ignore_inertial_definitions=False)
        q = R.from_quat(quat_wxyz, scalar_first=True).as_quat()
        comp.joint_q[0:3] = [float(v) for v in pos]; comp.joint_q[3:7] = [float(v) for v in q]
        nh = _decompose_object_collision(comp, ShapeSpan(0, comp.shape_count))
        cp.add_builder(comp)
        if with_support:
            add_support_surfaces(cp, usda)
        collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
        for s in range(cp.shape_count):
            if int(cp.shape_flags[s]) & collide:
                cp.shape_material_mu[s] = 1.0
        damp = cp.custom_attributes.get("mujoco:dof_passive_damping")
        for dof in range(6):
            damp.values[dof] = 0.01
        b = newton.ModelBuilder()
        b.replicate(cp, 1)
        b.add_ground_plane(cfg=replace(b.default_shape_cfg, mu=1.0))
        return b.finalize(device="cpu"), nh

    def run(model, seconds=10.0):
        solver = newton.solvers.SolverMuJoCo(model, solver=SIM.solver, integrator=SIM.integrator, njmax=SIM.njmax,
                                             nconmax=SIM.nconmax, impratio=SIM.impratio, cone=SIM.cone,
                                             iterations=SIM.iterations, ls_iterations=SIM.ls_iterations,
                                             use_mujoco_contacts=True, use_mujoco_cpu=False)
        s0, s1 = model.state(), model.state()
        ctrl = model.control()
        contacts = build_force_contacts(model, solver, SIM)
        newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
        traj = []
        for _ in range(int(seconds * SIM.fps)):
            for _ in range(SIM.substeps):
                s0.clear_forces()
                solver.step(s0, s1, ctrl, contacts, SIM.sim_dt)
                s0, s1 = s1, s0
            traj.append(s0.body_q.numpy()[0].copy())
        return np.array(traj)

    def metrics(traj, pos0, q0):
        R0 = R.from_quat(q0, scalar_first=True)
        dp = np.linalg.norm(traj[:, :3] - pos0, axis=1)
        rot = np.degrees((R.from_quat(traj[:, 3:7]) * R0.inv()).magnitude())
        return dict(final_dp_cm=round(100 * float(dp[-1]), 3), max_dp_cm=round(100 * float(dp.max()), 3),
                    final_rot_deg=round(float(rot[-1]), 3), max_rot_deg=round(float(rot.max()), 3),
                    final_z_change_cm=round(100 * float(traj[-1, 2] - pos0[2]), 3),
                    final_disp_cm=np.round(100 * (traj[-1, :3] - pos0), 2).tolist(),
                    disp_at_0p2_1_2s_cm=[np.round(100 * (traj[k, :3] - pos0), 2).tolist() for k in (9, 49, 99) if k < len(traj)],
                    ok=bool(dp[-1] <= 0.02 and rot[-1] <= 5.0 and dp.max() <= 0.03))
    tests = []
    anchor = rec.get("anchor_run")
    for r in rr["runs"]:
        a, b = r["run"]
        if not r["near_support"]:
            continue
        if only_anchor and not (anchor and [a, b] == list(anchor)):
            continue
        t = (a + b) // 2
        pos0, q0 = fr["obj_pos"][t], fr["obj_wxyz"][t]
        row = dict(run=[a, b], frame=int(t), anchor=bool(anchor and [a, b] == list(anchor)), held_min_dist_cm=r.get("held_min_dist_cm"))
        model, nh = build(pos0, q0, True)
        row["with_support"] = metrics(run(model), pos0, q0); row["hulls"] = nh
        if row["anchor"] and not only_anchor:
            model, _ = build(pos0, q0, False)
            row["no_support_control"] = metrics(run(model), pos0, q0)
        tests.append(row)
        print(json.dumps(row), flush=True)
    res = dict(ep=ep, usda=str(usda), mesh=str(mesh), mesh_equal_parquet_mesh=same_mesh, sim=SIM.__dict__,
               pass_rule="10 s: final |dp|<=2cm, final rot<=5deg, max |dp|<=3cm", tests=tests,
               all_ok=bool(tests) and all(x["with_support"]["ok"] for x in tests),
               anchor_ok=any(x["anchor"] and x["with_support"]["ok"] for x in tests),
               anchor_control_moves=any(x["anchor"] and not x.get("no_support_control", {}).get("ok", True) for x in tests))
    if save:
        write("settle", ep, res)
    print("settle all_ok", res["all_ok"], "anchor_ok", res["anchor_ok"], "control moves", res["anchor_control_moves"], flush=True)
    return res


def summary():
    out = {}
    for ep in SUPPORT_EPS:
        k = f"{ep:02d}"
        rec = json.loads((A / "supports" / f"ep{k}.json").read_text())
        row = dict(support=rec["support"], kind=rec["kind"], n_boxes=rec["n_boxes"], flags=[])
        for name in ("rest", "human", "g1", "settle"):
            p = OUT / f"{name}_ep{k}.json"
            row[name] = json.loads(p.read_text()) if p.is_file() else None
            if row[name] is None:
                row["flags"].append(f"{name.upper()}_MISSING")
        r = row["rest"]
        if r:
            pf = np.asarray(r.pop("per_frame_sdf_cm", []), float)
            deep = np.where(pf < -1.0)[0]
            r["frames_deeper_1cm"] = int(len(deep))
            r["deep_runs_1cm"] = [[int(x[0]), int(x[-1])] for x in np.split(deep, np.where(np.diff(deep) > 1)[0] + 1)] if len(deep) else []
            for x in r["runs"]:
                if x["near_support"] and (x["median_sdf_cm"] > 2.0 or x["median_sdf_cm"] < -1.5):
                    row["flags"].append(f"REST_GAP run {x['run']} median sdf {x['median_sdf_cm']} cm")
                if rec["kind"] == "top" and x["near_support"] and x["footprint_cover"] < 0.8:
                    row["flags"].append(f"FOOTPRINT run {x['run']} cover {x['footprint_cover']}")
            if r["frames_deeper_1cm"]:
                row["flags"].append(f"PATH object deeper than 1 cm in a box on {r['frames_deeper_1cm']} frames {r['deep_runs_1cm'][:4]} (min {r['path_min_sdf_cm']} cm; a 3 cm slab allows at most 1.5)")
        h = row["human"]
        if h and h["violating_frames"]:
            row["flags"].append(f"HUMAN {h['violating_frames']} frames {h['violating_runs'][:4]} worst {h['worst'][0]['depth_cm']} cm ({h['joints_involved'][:4]})")
        g = row["g1"]
        if g and g["violating_frames"]:
            row["flags"].append(f"G1 {g['violating_frames']} frames {g['violating_runs'][:4]} worst {g['worst'][0]['depth_cm']} cm ({g['links_involved'][:4]})")
        s = row["settle"]
        if s:
            if not s["anchor_ok"]:
                row["flags"].append("SETTLE anchor run not stable")
            held_of = {tuple(x["run"]): x.get("held_min_dist_cm") for x in (r["runs"] if r else [])}
            for x in s["tests"]:
                if not x["with_support"]["ok"]:
                    hd = x.get("held_min_dist_cm")
                    hd = held_of.get(tuple(x["run"])) if hd is None else hd
                    tag = "SETTLE" if hd is None or hd > 3.0 else "settle (human hand within 3 cm: held, not a free rest)"
                    row["flags"].append(f"{tag} run {x['run']} dp {x['with_support']['final_dp_cm']} cm rot {x['with_support']['final_rot_deg']} deg; human-object {hd} cm")
            if not s["anchor_control_moves"]:
                row["flags"].append("SETTLE control without support did not move (support not needed at the anchor?)")
            if not s["mesh_equal_parquet_mesh"]:
                row["flags"].append("SETTLE mesh != parquet mesh")
        out[k] = row
    res = dict(written=time.strftime("%F %T"), episodes=out)
    (OUT / "physcheck_summary.json").write_text(json.dumps(res, indent=1, default=float))
    for k, v in out.items():
        print(f"ep{k} {v['support']:18s} boxes {v['n_boxes']:3d} flags: {v['flags'] or 'none'}")


if __name__ == "__main__":
    cmd, eps = sys.argv[1], [int(x) for x in sys.argv[2:]]
    if cmd == "summary":
        summary()
    else:
        for e in eps:
            dict(rest=rest, human=human, g1=g1, settle=settle)[cmd](e)
