import numpy as np
d=np.load('/work/scratch/masks_half.npz'); obj=d['obj']; per=d['per']
F=len(obj); k=5
r=[]
for i in range(F-k):
    v=~(per[i]|per[i+k]); a=obj[i]&v; b=obj[i+k]&v
    r.append((a&b).sum()/max(1,(a|b).sum()))
r=np.array(r)
import sys; sys.path.insert(0,'/work/outputs')
from common import *
K,Kp,Twc,rec,gp=load_inputs('/work/inputs')
cp=rec['contact_prob_right_left']
for i in range(0,F-k,5): print(i, '%.3f'%r[i], np.round(cp[i],2), obj[i].sum())
