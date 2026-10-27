"""Independent structural and reproducibility checks of final output."""
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--repeat',type=Path);a=p.parse_args()
out=Path('/work/outputs');inp=Path('/work/inputs')
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
hashes=json.loads((out/'input_sha256.json').read_text())
assert all(sha(inp/name)==value for name,value in hashes.items())
with np.load(out/'object_poses_world.npz') as z:
 assert z.files==['T_world_mesh'];T=z['T_world_mesh']
B=np.load(inp/'reconstruction.npz')['T_world_mesh'].astype(np.float64)
cam=json.loads((inp/'camera.json').read_text());W=np.array(cam['T_world_cam'])
C=np.linalg.inv(W);Tc=C@T;Bc=C@B
assert T.shape==(859,4,4) and T.dtype==np.float64 and np.isfinite(T).all()
assert np.array_equal(T[:,3,:],np.tile([0.,0.,0.,1.],(859,1)))
radius=np.linalg.norm(Tc[:,:3,3],axis=1);base_radius=np.linalg.norm(Bc[:,:3,3],axis=1)
assert np.max(np.abs(radius-base_radius))<1e-12
changed=np.any(T!=B,axis=(1,2))
rows=list(csv.DictReader(open(out/'frame_diagnostics.csv')))
assert len(rows)==859 and [int(r['frame']) for r in rows]==list(range(859))
assert np.array_equal(changed,np.array([int(r['changed']) for r in rows],bool))
err=T[:,:3,:3].transpose(0,2,1)@T[:,:3,:3]-np.eye(3)
assert np.max(np.abs(err))<1e-5 and np.max(np.abs(np.linalg.det(T[:,:3,:3])-1))<1e-5
assert np.max(np.abs(err[changed]))<1e-12
result=dict(shape=list(T.shape),dtype=str(T.dtype),finite=True,inputs_unchanged=True,
 changed_frames=int(changed.sum()),unchanged_frames=int((~changed).sum()),
 max_camera_distance_change_m=float(np.max(np.abs(radius-base_radius))),
 max_edited_rotation_orthogonality_error=float(np.max(np.abs(err[changed]))),
 output_sha256=sha(out/'object_poses_world.npz'),solver_sha256=sha(out/'solve.py'))
if a.repeat:
 R=np.load(a.repeat)['T_world_mesh']
 assert np.array_equal(T,R)
 result.update(repeat_bitwise_equal=True,repeat_max_abs_difference=float(np.max(np.abs(T-R))),repeat_sha256=sha(a.repeat))
(out/'validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
