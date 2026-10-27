import json,numpy as np,trimesh
T2DS='/mnt/nas/Group/hanzhou/v2d/dataset/track_2/tier_2_synthetic_noise'
meta={d["episode_index"]:d for d in map(json.loads,open(f"{T2DS}/meta/episodes_metadata.jsonl"))}
for i in range(30):
    m=trimesh.load(f"{T2DS}/{meta[i]['mesh']}",force="scene",process=False).to_geometry()
    p=np.asarray(m.sample(20000)); e=np.linalg.eigvalsh(np.cov((p-p.mean(0)).T))
    r=max(e[0]/e[1], e[1]/e[2]); print(i, meta[i]['object'][:18], np.round(np.sqrt(e),3), 'closest-pair ratio %.2f'%r)
