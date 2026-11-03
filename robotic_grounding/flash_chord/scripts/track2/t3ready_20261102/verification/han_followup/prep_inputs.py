"""prep_inputs.py <wd>: build the NAS-like layout V/ for t3_frame0.py (independent of the checker's harness).
agent npz <- tier3_opus_20261021 poses.npy (circular: the agent npz are not on this server);
ref_poses <- pyarrow (not pandas), FIRST ROW ONLY, asserted frame_index == 0. No score computed."""
import json, sys, hashlib
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
WD = Path(sys.argv[1]); V = WD / "V"
HAN = Path("/root/autodl-tmp/han.zhou/tier3/release"); ME = Path("/root/autodl-tmp/haoran.zhang")
DS = ME / "data/video_to_data_challenge/track_2/tier_1_multiview_caption"
(V / "dataset/track_2").mkdir(parents=True, exist_ok=True); (V / "release/tier3_s1_20261020").mkdir(parents=True, exist_ok=True)
for l, t in ((V / "dataset/track_2/tier_1_multiview_caption", DS), (V / "release/tier3_s1_20261020/reconstructed_data", HAN / "tier3_s1_20261020/reconstructed_data")):
    if not l.is_symlink(): l.symlink_to(t)
eps = json.load(open(HAN / "tier3_opus_20261021/episodes.json")); (V / "ref_poses").mkdir(exist_ok=True)
dt = set()
for e in eps:
    i, seq = int(e["episode_index"]), e["sequence_id"]
    run = f"{'CL2' if i <= 10 else 'CL2b' if i <= 21 else 'CL2c'}_ep{i:03d}"
    P = np.load(HAN / "tier3_opus_20261021/reconstructed_data" / seq / "poses.npy"); dt.add(str(P.dtype))
    d = V / "agent_runs" / run / "outputs"; d.mkdir(parents=True, exist_ok=True); np.savez(d / "object_poses_world.npz", T_world_mesh=P)
    pf = pq.ParquetFile(DS / f"data/chunk-000/episode_{i:06d}.parquet")
    tb = pf.read_row_group(0, columns=["frame_index", "observation.object.pose", "observation.object.visible"]).slice(0, 1)
    fi = np.asarray(tb.column("frame_index").to_pylist()); assert fi.tolist() == [0], fi
    pz = np.asarray(tb.column("observation.object.pose").to_pylist(), dtype=np.float64)
    vis = np.asarray(tb.column("observation.object.visible").to_pylist()).reshape(-1)
    np.savez(V / "ref_poses" / f"episode_{i:06d}.npz", frame_index=fi, pose_xyz_wxyz=pz, visible=vis)
print("prepared", len(eps), "episodes; opus poses dtypes", dt, "; pose type", tb.schema.field("observation.object.pose").type)
