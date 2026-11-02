"""Per-episode cache of the Tier-2 inputs: noisy object track (invisible frames filled by the nearest visible one),
Tier-2 SOMA joints in the OpenCV world, floor, ZUPT rest mask, mesh samples, support-stage boxes, the initial estimate,
and the bias covariance of the agile K1 smoother (used only as the initial estimate's uncertainty for detection)."""
import json, re, time
import numpy as np
import trimesh
from common import kf, kc, mesh_path_for, sample_vertices, STAGE, INIT_FMT, CONST


def stage_boxes(seq):
    """Axis-aligned boxes [(lo, hi)] in the robot world (m, Z up) from the support USDA (Cube: translate/scale/size;
    Cylinder: translate, radius, height, axis Z). Same parser as kfix/run_fix.py."""
    f = STAGE / f"{seq}_support.usda"
    if not f.exists():
        return []
    txt = f.read_text(); boxes = []
    for kind, body in re.findall(r'def (Cube|Cylinder) "\w+"[^{]*\{(.*?)\n    \}', txt, flags=re.S):
        t = np.array([float(v) for v in re.search(r"xformOp:translate = \(([^)]*)\)", body).group(1).split(",")])
        s = re.search(r"xformOp:scale = \(([^)]*)\)", body)
        sc = np.array([float(v) for v in s.group(1).split(",")]) if s else np.ones(3)
        if kind == "Cube":
            size = float(re.search(r"double size = ([\d.eE+-]+)", body).group(1)) if "double size" in body else 2.0
            half = 0.5 * size * sc
        else:
            r = float(re.search(r"double radius = ([\d.eE+-]+)", body).group(1))
            h = float(re.search(r"double height = ([\d.eE+-]+)", body).group(1))
            half = np.array([r, r, h / 2]) * sc
        boxes.append((t - half, t + half))
    return boxes


def agile_params(ep):
    """Frozen filter parameters with the motion-prior process noise q = max(pooled, the episode's own ML value)."""
    base = kf.load_params(); ML = json.load(open(kf.DES / 'd3_ml.json'))
    prm = json.loads(json.dumps(base))
    prm['q_a_cm2_s3'][str(ep)] = max(base['q_a_cm2_s3'][str(ep)], ML[str(ep)]['pos']['q_ml'])
    prm['q_al_rad2_s3'][str(ep)] = max(base['q_al_rad2_s3'][str(ep)], ML[str(ep)]['rot']['q_ml'])
    return prm


def build(ep, path, log=print):
    t00 = time.time()
    prm = agile_params(ep)
    m = {mm['episode_index']: mm for mm in kf.episodes()}[ep]
    e = kf.Episode(m, prm)
    g = kc.load_g1(m['sequence_id'])
    Rs2r, ts2r = np.array(e.fr['Rs2r']), np.array(e.fr['ts2r']); B, b = np.array(e.fr['Bcv2r']), np.array(e.fr['bcv2r'])
    J = (((g['soma_joints'] @ Rs2r.T + ts2r) - b) @ B) * 100.0           # (T,77,3) cm, OpenCV world
    assert J.shape[0] == e.T, (J.shape, e.T)
    mp = mesh_path_for(kc.T2 / 'mesh', e.obj)
    mesh = trimesh.load(mp, force='mesh')
    pts = sample_vertices(mp)                                              # the kit's 500 vertex samples (m)
    V = np.asarray(mesh.vertices)
    Vs = V[np.random.default_rng(0).choice(len(V), min(20000, len(V)), replace=False)]   # for the convex hull (floor)
    n_s = int(np.ceil(mesh.area * 1e4 / CONST['surf_spacing_cm'] ** 2))
    S, fi = trimesh.sample.sample_surface(mesh, n_s, seed=0)               # dense surface samples (m) + face normals
    N = np.asarray(mesh.face_normals)[fi]
    init = np.load(INIT_FMT.format(ep=ep))
    assert init.shape == (int(e.vis.sum()), 7), (init.shape, int(e.vis.sum()))
    t0 = time.time()
    X1, P1, _, _ = e.run(e.base_meas(zv=True))
    tk1 = time.time() - t0
    Pb = np.stack([P1[t][12:18, 12:18] for t in range(e.T)])
    bx = np.array([[lo, hi] for lo, hi in stage_boxes(m['sequence_id'])]).reshape(-1, 2, 3)
    np.savez(path, ep=ep, obj=e.obj, seq=m['sequence_id'], p_m=e.p, q_m=e.rot.as_quat(), vis=e.vis, rest=e.rest,
             up=e.up, e=e.e, J=J, pts=pts, Vs=Vs, S=S, N=N, init=init, Bcv2r=B, bcv2r=b, boxes=bx, Pb=Pb,
             q_a=prm['q_a_cm2_s3'][str(ep)], q_al=prm['q_al_rad2_s3'][str(ep)])
    log(f'prep ep{ep} {e.obj}: T {e.T} visible {int(e.vis.sum())} rest {int(e.rest.sum())} surface samples {n_s} '
        f'boxes {len(bx)} q_a {prm["q_a_cm2_s3"][str(ep)]:.0f} q_al {prm["q_al_rad2_s3"][str(ep)]:.3f} '
        f'K1 {tk1:.0f}s total {time.time()-t00:.0f}s')
