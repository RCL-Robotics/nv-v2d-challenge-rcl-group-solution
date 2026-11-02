import sys, numpy as np, json, h5py, torch
sys.path.insert(0,'/work/outputs')
from render_utils import *; from silfit import *; from traj_utils import *
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam'])
pl=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane'])
m,tex=load_mesh('/work/inputs/object_mesh.glb'); R=Renderer(m,tex,K)
f=h5py.File('/work/inputs/masks.h5')
P=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(np.float64); Tc=np.linalg.inv(Twc)[None]@P
d=np.load('move_fit_0.npz'); Tf=d['Tf']; a,b=201,560
fit=SilFitter(R,K); fit.setup(f['object'][a:b],f['person'][a:b],Tf)
Tfree=fit.fit(shared=False,fix_depth=False,iters=(60,60),sigmas=(3,1.5),lr_r=0.003)
c=R.v.mean(0).cpu().numpy()
zc=lambda T: (np.einsum('fij,j->fi',T[:,:3,:3],c)+T[:,:3,3])[:,2]
sil_ratio=zc(Tfree)/zc(Tf)   # silhouette suggests scaling depth by this
# floor cue: lowest point height in plane coords for camera pose T; find scale s along rays so min height = h
n_c=(Twc[:3,:3].T@pl[:3]); d_c=pl[3]+pl[:3]@Twc[:3,3]   # plane in camera frame: n_c.x + d_c = 0
V=np.asarray(m.vertices)[::10]
def minh(T,s=1.0):
    X=(V@T[:3,:3].T+T[:3,3])  # mesh points
    X=X-T[:3,3]+s*T[:3,3]
    return (X@n_c+d_c).min()
h=np.array([minh(T) for T in Tf])
href=np.median(h)
floor_s=[]
for T,hh in zip(Tf,h):
    ss=np.linspace(0.85,1.15,301); hv=np.array([minh(T,s) for s in ss]); floor_s.append(ss[np.argmin(np.abs(hv-href))])
floor_s=np.array(floor_s)
np.savez('depthcue.npz',sil_ratio=sil_ratio,floor_s=floor_s,h=h,zf=zc(Tf),Tfree=Tfree)
zz=zc(Tf)
print('median sil ratio',np.median(sil_ratio))
sr=sil_ratio/np.median(sil_ratio)
for i in range(0,b-a,12): print(a+i,'z %.2f  h %.1fcm  floor-> %+.1fcm  sil-> %+.1fcm'%(zz[i],h[i]*100,(floor_s[i]-1)*zz[i]*100,(sr[i]-1)*zz[i]*100))
x=(floor_s-1)*zz; y=(sr-1)*zz
print('corr floor vs sil', np.corrcoef(x,y)[0,1], ' smoothed corr', np.corrcoef(np.convolve(x,np.ones(15)/15,'valid'),np.convolve(y,np.ones(15)/15,'valid'))[0,1])
print('std floor %.1f sil %.1f cm'%(x.std()*100,y.std()*100))
