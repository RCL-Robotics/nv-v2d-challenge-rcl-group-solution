import sys, pickle; sys.path.insert(0,'/work/outputs')
import numpy as np
from stages import *
A=pickle.load(open('/work/scratch/stageA.pkl','rb')); res=A['res']
d=np.load('/work/scratch/masks_small.npz'); s1,s2,i0,i1=detect_static_phases(d['o'],d['p']); print('static',s1,s2)
cands=[r['T'] for r in res]; un=[unary_cost(r['sc']) for r in res]
k=viterbi(cands,un)
names=[f'{s}{j}' for s in ['base','fp','cc'] for j in range(4)]
best=np.array([np.argmin(u) for u in un])
print('frames where viterbi != per-frame best:', (k!=best).sum())
T=np.stack([cands[i][k[i]] for i in range(len(k))])
ep_K,Twc,rec,_=load_inputs('/work/inputs')
np.savez('/work/scratch/dec1.npz',Tc=T,T=Twc[None]@T,k=k)
# axis consistency w.r.t baseline
Tb=np.linalg.inv(Twc)[None]@rec['T_world_mesh'].astype(np.float64)
ang=np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i',T[:,:3,2],Tb[:,:3,2]),-1,1)))
for i in range(0,803,10): print(i,names[k[i]],'u %.3f'%un[i][k[i]],'axis-vs-base %.0f'%ang[i], 'z %.2f zb %.2f'%(T[i,2,3],Tb[i,2,3]))
