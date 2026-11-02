import json,numpy as np,trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
T2DS='/mnt/nas/Group/hanzhou/v2d/dataset/track_2/tier_2_synthetic_noise'
meta={d["episode_index"]:d for d in map(json.loads,open(f"{T2DS}/meta/episodes_metadata.jsonl"))}
for i in range(30):
    m=trimesh.load(f"{T2DS}/{meta[i]['mesh']}",force="scene",process=False).to_geometry()
    np.random.seed(0); p=np.asarray(m.sample(8000)); c=p.mean(0); q=p-c; E=np.linalg.eigh(np.cov(q.T))[1]
    size=np.linalg.norm(q.max(0)-q.min(0)); tr=cKDTree(q); out=[]
    for k in range(3):
        d=[tr.query(Rot.from_rotvec(E[:,k]*np.radians(a)).apply(q))[0].mean() for a in range(30,360,30)]
        out.append(np.mean(d)/size*100)
    k=int(np.argmin(out)); print(i, meta[i]['object'][:18], 'axis-sym score %% of size: '+' '.join('%.2f'%x for x in out), '-> best axis',k)
