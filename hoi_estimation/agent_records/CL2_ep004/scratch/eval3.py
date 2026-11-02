import sys; sys.path.insert(0,'/work/scratch')
from common import *
from validate import *
import pickle
ep,A=get_ep(); FR,simp=get_FR()
s1,s2=110,706
phases={'static-1':(0,s1),'moving':(s1+1,s2-1),'static-2':(s2,ep.F-1),'all':(0,ep.F-1)}
zt=depth_targets(ep,s1,s2)
out={}; det={}
for nm,path in [('baseline',None)]+[(a,b) for a,b in zip(sys.argv[1::2],sys.argv[2::2])]:
    Tc=ep.Tc_base if path is None else anchor_depth(np.load(path)['Tc'],zt)
    if path is not None: np.savez(path.replace('.npz','_anch.npz'),Tc=Tc,T=ep.Twc[None]@Tc)
    out[nm],det[nm]=evaluate(ep,FR,Tc,phases,simp['vs'])
print(format_table(out))
pickle.dump(det,open('/work/scratch/eval_det.pkl','wb'))
