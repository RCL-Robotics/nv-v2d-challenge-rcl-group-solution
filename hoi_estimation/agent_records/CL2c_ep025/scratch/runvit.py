import pickle, numpy as np, sys
from vit import *
res=pickle.load(open(sys.argv[1],'rb'))
F=859; by=[[] for _ in range(F)]
for x in res: by[x['f']].append(x)
wc=float(sys.argv[2]); lr=float(sys.argv[3])
cands=[(np.array([x['R'] for x in b]),np.array([x['t'] for x in b])) for b in by]
un=[np.array([x['ls']+wc*x['lc'] for x in b]) for b in by]
path=viterbi(cands,un,lam_r=lr)
sel=[by[i][k] for i,k in enumerate(path)]
R=np.array([s['R'] for s in sel]); t=np.array([s['t'] for s in sel])
steps=np.degrees(np.arccos(np.clip((np.einsum('nij,nij->n',R[1:],R[:-1])-1)/2,-1,1)))
print('big steps', [(i+1,round(s)) for i,s in enumerate(steps) if s>20])
print('srcs', ''.join(s['src'][1:] [0:2]+' ' for s in sel[::10]))
np.savez(sys.argv[4],R=R,t=t,frames=np.arange(F))
