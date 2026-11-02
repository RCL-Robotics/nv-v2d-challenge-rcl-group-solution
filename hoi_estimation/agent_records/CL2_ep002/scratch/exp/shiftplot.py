import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from hoilib import load_inputs
K,Twc,rec,cam=load_inputs('/work/inputs')
res=np.load(sys.argv[1]); a,b,st=[int(x) for x in sys.argv[2:5]]; src=sys.argv[5]
T=rec['T_world_mesh'].astype(np.float64) if src=='base' else np.load(src)['T_world_mesh']
fr=np.arange(a,b,st)
x=T[fr,:3,0]; yaw=np.degrees(np.arctan2(x[:,2],x[:,0]))
# rotation about mesh +y by delta: mesh x -> cos*x - sin*z ... sign check numerically
from scipy.spatial.transform import Rotation as Rot
Ry=Rot.from_euler('y',10,degrees=True).as_matrix()
x2=(T[fr[0],:3,:3]@Ry)[:,0]; dyaw=np.degrees(np.arctan2(x2[2],x2[0]))-yaw[0]
sign=np.sign(((dyaw+180)%360)-180); print('sign',sign)
ny=res.shape[1]; step=360/ny
fig,ax=plt.subplots(1,4,figsize=(20,6))
names=['iou','edge','ncc','negcdist']
for k in range(4):
    z=(res[:,:,k]-res[:,:,k].mean(1,keepdims=True))/(res[:,:,k].std(1,keepdims=True)+1e-9)
    out=np.zeros_like(z)
    for j in range(len(fr)):
        for yi in range(ny):
            absy=(yaw[j]+sign*yi*step)%360
            out[j,int(round(absy/step))%ny]=z[j,yi]
    ax[k].imshow(out,aspect='auto',extent=[0,360,fr[-1],fr[0]]); ax[k].set_title(names[k])
    m=out.mean(0); print(names[k],'abs-yaw best',np.arange(ny)[np.argsort(-m)[:5]]*step)
plt.savefig(sys.argv[6],dpi=50)
