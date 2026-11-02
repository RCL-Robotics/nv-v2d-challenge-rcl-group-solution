import sys, pickle, numpy as np
sys.path.insert(0,'/work/outputs'); sys.path.insert(0,'/work/scratch')
from vit import viterbi
states=pickle.load(open(sys.argv[1],'rb')); F=len(states)
wc=float(sys.argv[2]); lr=float(sys.argv[3])
cands=[(np.array([s['R'] for s in st]),np.array([s['t'] for s in st])) for st in states]
un=[np.array([s['u'] for s in st]) for st in states]
path=viterbi(cands,un,lam_r=lr,lam_t=1.0)
R=np.array([cands[i][0][k] for i,k in enumerate(path)]); t=np.array([cands[i][1][k] for i,k in enumerate(path)])
steps=np.degrees(np.arccos(np.clip((np.einsum('nij,nij->n',R[1:],R[:-1])-1)/2,-1,1)))
print('big steps', [(i+1,round(s)) for i,s in enumerate(steps) if s>25])
np.savez(sys.argv[4],R=R,t=t,frames=np.arange(F))
