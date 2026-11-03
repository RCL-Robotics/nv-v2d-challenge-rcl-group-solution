import sys; sys.path.insert(0,'/work/outputs'); sys.path.insert(0,'/work/scratch')
from common import *; from fitcore import *; from hyp import *; from vizfit import viz
K,Kp,Twc,rec=load_inputs('/work/inputs')
mesh=load_mesh('/work/inputs/object_mesh.glb'); fit=Fitter(mesh,K)
d=dict(np.load('/work/scratch/cache_v1.npz'))
pl=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']); up_c=Twc[:3,:3].T@pl[:3]
fr=int(sys.argv[1])
Rs,tags=hypothesis_set(up_c,n_yaw=72,tilts=(0,))
cm=mesh.bounds.mean(0); mv=np.asarray(mesh.vertices)[::50]
zs=[2.4,2.6,2.8,3.0,3.2,3.4]
res=[]
for z in zs:
    t0=init_translation(fit,Rs,d['box'][fr],K,mv,cm)
    t0=t0*(z/t0[:,2:3])
    # depth fixed via penalty
    zt=torch.tensor(z,device='cuda')
    R,t,l=fit_batch(fit,Rs,t0,[fr]*len(Rs),d,iters=120,lr_r=0.01,lr_t=0.005,extra_loss=lambda Rm,t: 100*((t[:,2]-zt)**2).sum())
    k=np.argmin(l); res.append((z,l[k],np.degrees(tags[k][1]))); print(z,l[k],np.degrees(tags[k][1]),t[k],flush=True)
    viz(fit,d,[fr],R[[k]],t[[k]],f'/work/scratch/viz/z{z}.jpg',ncol=1)
