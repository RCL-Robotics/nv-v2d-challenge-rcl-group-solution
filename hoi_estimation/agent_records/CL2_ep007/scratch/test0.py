import sys; sys.path.insert(0,'/work/outputs')
from common import *; from fitcore import *; from hyp import *
import time
K,Kp,Twc,rec=load_inputs('/work/inputs')
mesh=load_mesh('/work/inputs/object_mesh.glb'); fit=Fitter(mesh,K)
d=dict(np.load('/work/scratch/cache_v1.npz'))
pl=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']); up_c=Twc[:3,:3].T@pl[:3]
print('up_c',up_c)
Rs,tags=hypothesis_set(up_c,n_yaw=72,tilts=(0,1))
cm=mesh.bounds.mean(0); mv=np.asarray(mesh.vertices)[::50]
fr=int(sys.argv[1])
t0=init_translation(fit,Rs,d['box'][fr],K,mv,cm)
t1=time.time()
R,t,l=fit_batch(fit,Rs,t0,[fr]*len(Rs),d,iters=60,opt_rot=False)
print('time',time.time()-t1)
o=np.argsort(l)
for k in o[:10]: print(k,tags[k][0],np.degrees(tags[k][1]),l[k],t[k])
np.savez('/work/scratch/test0.npz',R=R,t=t,l=l,o=o)
