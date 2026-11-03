import sys, numpy as np, json, h5py, torch, time
sys.path.insert(0,'/work/outputs')
from render_utils import *; from silfit import *
from scipy.spatial.transform import Rotation as Rot
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam'])
m,tex=load_mesh('/work/inputs/object_mesh.glb'); R=Renderer(m,tex,K)
P=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(np.float64)
Tc=np.linalg.inv(Twc)[None]@P
f=h5py.File('/work/inputs/masks.h5')
a,b=int(sys.argv[1]),int(sys.argv[2]); fr=np.arange(a,b,4)
obj=f['object'][fr[0]:fr[-1]+1:4]; per=f['person'][fr[0]:fr[-1]+1:4]
# robust mean pose
rm=Rot.from_matrix(Tc[a:b,:3,:3]).mean().as_matrix(); tm=np.median(Tc[a:b,:3,3],0)
Tm=np.eye(4); Tm[:3,:3]=rm; Tm[:3,3]=tm
fit=SilFitter(R,K); fit.setup(obj,per,np.repeat(Tm[None],len(fr),0))
t0=time.time()
print('score baseline per-frame', fit.score(Tc[fr]).mean(), 'mean pose', fit.score(np.repeat(Tm[None],len(fr),0)).mean())
Ts=fit.fit(shared=True)
print('shared fit', fit.score(Ts).mean(), time.time()-t0)
d=Ts[0]
print('shared delta rot deg', np.degrees(Rot.from_matrix(d[:3,:3]@rm.T).magnitude()), 'center shift cm', (d[:3,:3]@fit.c.cpu().numpy()+d[:3,3]-(rm@fit.c.cpu().numpy()+tm))*100)
Tsd=fit.fit(shared=True,fix_depth=True)
print('shared fixdepth', fit.score(Tsd).mean())
fit.setup(obj,per,Tc[fr])
Tf=fit.fit(shared=False, w_prior_r=0.0)
print('per-frame free fit', fit.score(Tf).mean())
c=fit.c.cpu().numpy()
cc=np.einsum('bij,j->bi',Tf[:,:3,:3],c)+Tf[:,:3,3]; cb=np.einsum('bij,j->bi',Tc[fr,:3,:3],c)+Tc[fr,:3,3]
print('per-frame centres std cm free-fit', (cc.std(0)*100).round(2), 'baseline', (cb.std(0)*100).round(2))
rr=Rot.from_matrix(Tf[:,:3,:3]); print('rot std deg fit', np.degrees((rr*rr.mean().inv()).magnitude()).mean(), 'base', np.degrees((Rot.from_matrix(Tc[fr,:3,:3])*Rot.from_matrix(rm).inv()).magnitude()).mean())
print('fit mean rot vs shared', np.degrees((rr.mean()*Rot.from_matrix(Ts[0,:3,:3]).inv()).magnitude()), 'centre mean diff cm', (cc.mean(0)-(Ts[0,:3,:3]@c+Ts[0,:3,3]))*100)
np.savez('static_%d.npz'%a, Ts=Ts[0], Tsd=Tsd[0], Tm=Tm, Tf=Tf, fr=fr)
c=fit.c.cpu().numpy()
for nm,T in [('shared',Ts[0]),('shared_fixdepth',Tsd[0])]:
    print(nm,'rot change vs mean deg %.2f'%np.degrees(Rot.from_matrix(T[:3,:3]@rm.T).magnitude()), 'centre shift cm', ((T[:3,:3]@c+T[:3,3])-(rm@c+tm))*100, 'rotvec', np.degrees(Rot.from_matrix(T[:3,:3]@rm.T).as_rotvec()))
