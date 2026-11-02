import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, torch
from hoilib import *
ep=Episode('/work/inputs'); rd=Renderer(ep.mesh)
res=np.load(sys.argv[1])['T_world_mesh']; a_end,b_start=125,335
Tc=ep.T_cam_world[None]@res
n=ep.plane[:3]; d=ep.plane[3]; nn=np.linalg.norm(n); n=n/nn; d=d/nn
V=np.asarray(ep.mesh.vertices)
# (a) floor contact in static floor phase: lowest vertex (along gravity / plane normal) distance to plane
T=res[0]; Vw=(T[:3,:3]@V.T).T+T[:3,3]
sd=Vw@n+d   # signed distance
print('floor phase: signed plane distance of mesh points: min %.3f max %.3f (n=%s)'%(sd.min(),sd.max(),np.round(n,3)))
# side of the camera: camera centre
cc=ep.T_world_cam[:3,3]; print('camera signed dist %.3f'%(cc@n+d))
# shift along viewing ray needed to bring the lowest point onto the plane
ctr_c=Tc[0,:3,3]; ray=ctr_c/np.linalg.norm(ctr_c); ray_w=ep.T_world_cam[:3,:3]@ray
# moving by s along ray changes plane distance by s*(ray_w.n); choose sign so that the contact distance (the extreme point toward the floor) becomes 0
contact=sd.min() if abs(sd.min())<abs(sd.max()) else sd.max()
# the base is the extreme on the floor side: floor is on the side opposite the camera
base_side = sd.max() if (cc@n+d)<0 else sd.min()
s=-base_side/(ray_w@n)
print('floor cue: base-to-plane distance %.3f m -> along-ray shift %.3f m (dz %.3f m)'%(base_side,s,s*ray[2]))
# (b) silhouette scale: pooled fit with z free
for name,a,b in [('floor',0,a_end+1),('table',b_start,440)]:
    frames=list(range(a,b,4))
    Tm=Tc[frames[0]]
    tg=MaskTargets(ep,frames,np.repeat(Tm[None],len(frames),0),size=192)
    T0c=torch.tensor(Tm[None],device='cuda',dtype=torch.float32)
    w=torch.zeros(1,3,device='cuda',requires_grad=True); t=torch.zeros(1,3,device='cuda',requires_grad=True)
    opt=torch.optim.Adam([w,t],lr=0.005)
    for it in range(200):
        TT=compose(T0c,w,t,1.0,True).expand(len(frames),4,4)
        iou=soft_iou(render_masks(rd,TT,ep.K,tg.crops,tg.size),tg); loss=1-iou.mean()
        opt.zero_grad(); loss.backward(); opt.step()
    Tf=TT[0].detach().cpu().numpy()
    print(name,'silhouette-scale cue: dz %.3f m (iou %.4f)'%(Tf[2,3]-Tm[2,3],iou.mean().item()))
# (c) hand contact during carry: distance from wrists to the cone surface
import trimesh
from scipy.spatial import cKDTree
pts=ep.mesh.sample(20000); kd=cKDTree(pts)
dzs=[]
for t in range(a_end+1,b_start):
    if ep.contact[t].min()<0.9: continue
    Tinv=np.linalg.inv(res[t])
    for j in [41,62]:
        p=Tinv[:3,:3]@ep.kp[t,j]+Tinv[:3,3]
        dist,_=kd.query(p)
        dzs.append(dist)
print('carry: wrist-to-cone-surface distance median %.3f m, p25 %.3f p75 %.3f'%tuple(np.percentile(dzs,[50,25,75])))
# along-ray offset between wrist midpoint and cone centre (camera frame z)
wm=0.5*(ep.kp[:,41]+ep.kp[:,62]); wm_c=(ep.T_cam_world[:3,:3]@wm.T).T+ep.T_cam_world[:3,3]
cl=np.array(list(AXIS_C[:2])+[0.09,1.0])
cz=np.array([(Tc[t]@cl)[2] for t in range(440)])
sel=[t for t in range(a_end+1,b_start) if ep.contact[t].min()>0.9]
print('carry: cone centre z - wrist-mid z: median %.3f m'%np.median(cz[sel]-wm_c[sel,2]))
