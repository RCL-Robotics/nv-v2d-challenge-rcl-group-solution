import sys,time; sys.path.insert(0,'/work/outputs'); sys.path.insert(0,'/work/scratch')
from common import *; from fitcore import *; from hyp import *
K,Kp,Twc,rec=load_inputs('/work/inputs')
mesh=load_mesh('/work/inputs/object_mesh.glb'); fit=Fitter(mesh,K)
d=dict(np.load('/work/scratch/cache_v1.npz'))
pl=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']); up_c=Twc[:3,:3].T@pl[:3]
Rs,tags=hypothesis_set(up_c,n_yaw=36,tilts=(0,1/3,2/3,1))
cm=mesh.bounds.mean(0); mv=np.asarray(mesh.vertices)[::50]
keys=list(range(0,896,int(sys.argv[1]))); out={}
T0=time.time()
for fr in keys:
    t0=init_translation(fit,Rs,d['box'][fr],K,mv,cm)
    R,t,s,c=fit_batch(fit,Rs,t0,[fr]*len(Rs),d,iters=40,opt_rot=False,lr_t=0.01,hw=(96,96))
    o=np.argsort(s); sel=[]
    for k in o:
        if all(rot_angle_deg(R[k],R[j])>25 for j in sel): sel.append(k)
        if len(sel)==8: break
    R2,t2,s2,c2=fit_batch(fit,R[sel],t[sel],[fr]*len(sel),d,iters=100,lr_r=0.01,lr_t=0.004,w_col=0.1)
    out[fr]=(R2,t2,s2,c2)
    b=np.argmin(s2+0.3*c2)
    print(fr,'best sil %.3f col %.3f z %.3f'%(s2[b],c2[b],t2[b,2]),' sils',np.round(s2,3),' cols',np.round(c2,3),flush=True)
print('time',time.time()-T0)
import pickle; pickle.dump(out,open('/work/scratch/keys.pkl','wb'))
