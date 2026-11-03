import os, sys, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, trimesh
ME = Path("/root/autodl-tmp/haoran.zhang"); F0 = ME / "data/human_motion_data_tier3_f0"
sys.path.insert(0, str(ME / "code/fc_r2/src"))
import newton
from flash_chord.scene.support import add_support_surfaces
from flash_chord.assets.registry import support_usda_for_parquet
man = json.loads((F0 / "whole_body/soma/track2_manifest.json").read_text()); seq_of = {v["episode_index"]: k for k, v in man.items()}
def q2m(q):
    w, x, y, z = q.T
    return np.stack([np.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], -1), np.stack([2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], -1),
                     np.stack([2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], -1)], -2)
def bsdf(p, c, h):
    q = np.abs(p - c) - h
    return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0)
for ep in [int(a) for a in sys.argv[1:]]:
    seq = seq_of[ep]; sd = F0 / "whole_body/soma" / f"sequence_id={seq}"
    pqp = sd / "robot_name=g1/data.parquet"
    u = support_usda_for_parquet(str(pqp), robot_name="g1")
    b = newton.ModelBuilder(); bind = add_support_surfaces(b, u)
    n = len(bind.shapes)
    from pxr import Usd
    st = Usd.Stage.Open(str(u)); boxes = []
    for prim in st.Traverse():
        if prim.GetTypeName() == "Cube":
            c = np.array(prim.GetAttribute("xformOp:translate").Get(), float); s = np.array(prim.GetAttribute("xformOp:scale").Get(), float) * float(prim.GetAttribute("size").Get())
            ops = [str(o) for o in prim.GetAttribute("xformOpOrder").Get()]
            boxes.append((c, s / 2, ops))
    up = st.GetMetadata("upAxis"); mpu = st.GetMetadata("metersPerUnit")
    r = pq.read_table(pqp, columns=["object_body_position", "object_body_wxyz"]).to_pylist()[0]
    P = np.asarray(r["object_body_position"])[:, 0]; Rm = q2m(np.asarray(r["object_body_wxyz"])[:, 0])
    om = trimesh.load(str(sd / "object/textured_mesh.obj"), force="mesh", process=False)
    S = np.vstack([trimesh.sample.sample_surface(om, 8000, seed=1)[0], om.convex_hull.vertices])
    T = len(P)
    sdf = np.array([min(bsdf(S @ Rm[t].T + P[t], c, h).min() for c, h, _ in boxes) for t in range(T)])
    top = max(c[2] + h[2] for c, h, _ in boxes)
    bot0 = (S @ Rm[0].T + P[0])[:, 2].min(); botE = (S @ Rm[-1].T + P[-1])[:, 2].min()
    print(json.dumps(dict(ep=ep, usda=u.name, resolved_ok=u.parent == F0 / "whole_body/reconstructed_stage", trainer_shapes=n, cubes=len(boxes),
        xform_ops=sorted({tuple(o) for _, _, o in boxes}), upAxis=up, mpu=mpu, top_z=round(top, 4), obj_bottom_f0=round(bot0, 4), obj_bottom_end=round(botE, 4),
        min_sdf_frame0_cm=round(100 * sdf[0], 2), min_sdf_end_cm=round(100 * sdf[-1], 2), path_min_sdf_cm=round(100 * sdf.min(), 2), argmin=int(sdf.argmin()),
        frames_penetrating_gt1cm=int((sdf < -0.01).sum()), frames_penetrating_gt0_5cm=int((sdf < -0.005).sum()))), flush=True)
