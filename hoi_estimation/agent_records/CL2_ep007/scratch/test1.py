import sys; sys.path.insert(0,'/work/outputs'); sys.path.insert(0,'/work/scratch')
from common import *; from fitcore import *; from hyp import *; from vizfit import viz
K,Kp,Twc,rec=load_inputs('/work/inputs')
mesh=load_mesh('/work/inputs/object_mesh.glb'); fit=Fitter(mesh,K)
d=dict(np.load('/work/scratch/cache_v1.npz'))
z=np.load('/work/scratch/test0.npz'); o=z['o'][:8]
R,t,l=fit_batch(fit,z['R'][o],z['t'][o],[0]*8,d,iters=150,lr_r=0.01,lr_t=0.005)
print(l, t)
viz(fit,d,[0]*8,R,t,'/work/scratch/viz/fit0.jpg',ncol=2)
