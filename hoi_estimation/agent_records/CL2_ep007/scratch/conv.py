import sys,pickle; sys.path.insert(0,'/work/outputs')
from common import *; from fitcore import *; from hyp import *
K,Kp,Twc,rec=load_inputs('/work/inputs')
fit=Fitter(load_mesh('/work/inputs/object_mesh.glb'),K)
d=dict(np.load('/work/scratch/cache_v1.npz'))
out=pickle.load(open('/work/scratch/keys.pkl','rb'))
for f in [0,600,800]:
    R2,t2,s2,c2=out[f]; b=np.argmin(s2+0.3*c2)
    for it,wc,lr in [(100,0,0.01),(200,0,0.005),(100,0.1,0.01),(200,0.3,0.005)]:
        R,t,s,c=fit_batch(fit,R2[[b]],t2[[b]],[f],d,iters=it,lr_r=lr,lr_t=lr/2.5,w_col=wc)
        print(f,it,wc,lr,'sil %.3f col %.3f'%(s[0],c[0]),'rot change %.1f'%rot_angle_deg(R[0],R2[b]),'z %.3f'%t[0,2])
