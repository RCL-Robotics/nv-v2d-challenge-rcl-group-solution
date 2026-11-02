import os,time,json,argparse
os.environ.setdefault('OMP_NUM_THREADS','2')
import torch,nvdiffrast.torch as dr,numpy as np,trimesh,h5py
from scipy.spatial.transform import Rotation
pa=argparse.ArgumentParser();pa.add_argument('--candidate',default='/work/outputs/refined_candidate.npz');pa.add_argument('--out',default='/work/outputs/metrics_candidate.json');args=pa.parse_args();p='/work/inputs/';c=json.load(open(p+'camera.json'));K=np.array(c['K']);base=np.linalg.inv(c['T_world_cam'])@np.load(p+'reconstruction.npz')['T_world_mesh'];a=np.load(args.candidate);new=a['T_cam_mesh'] if 'T_cam_mesh' in a else np.linalg.inv(c['T_world_cam'])@a['T_world_mesh'];tracks=np.load('/work/outputs/feature_tracks_full.npz');points=np.array([[-.00386614,-.0140458,-.44275588],[.00168146,.01075304,.99291701]]);uvobs=np.stack([tracks['head'],tracks['tip']],1);m=trimesh.load(p+'object_mesh.glb',force='scene',process=False).to_geometry();v=torch.tensor(np.array(m.vertices),device='cuda',dtype=torch.float32);f=torch.tensor(np.array(m.faces),device='cuda',dtype=torch.int32);ctx=dr.RasterizeCudaContext();mask=h5py.File(p+'masks.h5');rows=[];timer=time.time()
for i in range(len(base)):
 if i%3 and i<980:continue
 oo=mask['object'][i][::2,::2];pp=mask['person'][i][::2,::2];pairs=[]
 for tt in [base[i],new[i]]:
  P=torch.tensor(tt,device='cuda',dtype=torch.float32);xyz=v@P[:3,:3].T+P[:3,3];z=xyz[:,2];u=K[0,0]*xyz[:,0]+K[0,2]*z;vv=K[1,1]*xyz[:,1]+K[1,2]*z;clip=torch.stack([2*u/1536-z,2*vv/1152-z,.99*z-.01,z],-1)[None];rast,_=dr.rasterize(ctx,clip,f,resolution=[576,768]);ma=(rast[0,:,:,3]>0).cpu().numpy();visible=ma&(~pp|oo);iou=(visible&oo).sum()/max(1,(visible|oo).sum());xyzp=points@tt[:3,:3].T+tt[:3,3];uv=xyzp@K.T;uv=uv[:,:2]/uv[:,2:];err=np.linalg.norm(uv-uvobs[i],axis=1);pairs.append(dict(iou=float(iou),head_error=float(err[0]) if np.isfinite(err[0]) else None,tip_error=float(err[1]) if np.isfinite(err[1]) else None))
 rows.append(dict(frame=i,baseline=pairs[0],candidate=pairs[1],rotation_change_deg=float(Rotation.from_matrix(new[i,:3,:3]@base[i,:3,:3].T).magnitude()*180/np.pi),translation_change=float(np.linalg.norm(new[i,:3,3]-base[i,:3,3]))))
json.dump(rows,open(args.out,'w'),indent=2)
for lo,hi in [(0,130),(130,500),(500,660),(660,690),(690,730),(730,790),(790,920),(920,988),(0,988)]:
 rr=[r for r in rows if lo<=r['frame']<hi];print(lo,hi,'IoU',*[round(np.mean([r[k]['iou'] for r in rr]),4) for k in ['baseline','candidate']], 'head px',*[round(np.mean([r[k]['head_error'] for r in rr if r[k]['head_error'] is not None]),2) for k in ['baseline','candidate']],'tip px',*[round(np.mean([r[k]['tip_error'] for r in rr if r[k]['tip_error'] is not None]),2) for k in ['baseline','candidate']])
print('sec',time.time()-timer)
