import sys; sys.path.insert(0,'/work/scratch')
from common import *
import time, pickle, torch
ep,A=get_ep(); FR,simp=get_FR(); res=A['res']; F=ep.F
s1,s2,_,_=detect_static_phases(ep.mask_small_o,ep.mask_small_p)
zb=smooth_baseline_z(ep)
D2=np.load('/work/scratch/dec2.npz')['Tc']
U=pickle.load(open('/work/scratch/unary2.pkl','rb'))
mid=list(range(s1+1,s2))
t0=time.time()
inits=[np.stack([D2[i-1],D2[i+1]]) for i in mid]
pr=fit_frames(ep,FR,mid,inits); print('prop fit',time.time()-t0)
newc=[r['T'] for r in pr]
dr=depth_scores(ep,FR,mid,newc)
cands=[r['T'] for r in res]; un=list(U['un'])
for j,i in enumerate(mid):
    fh=floor_heights(ep,newc[j],simp['vs'],zb[i])
    u=full_unary(ep,pr[j]['sc'],dr[j],fh)
    cands[i]=np.concatenate([cands[i],newc[j]]); un[i]=np.concatenate([un[i],u])
P1=D2[0]; P2=D2[-1]
cm=[cands[i] for i in mid]; um=[un[i].copy() for i in mid]
um[0]=um[0]+pair_cost(P1[None],cm[0])[0]/10.0; um[-1]=um[-1]+pair_cost(cm[-1],P2[None])[:,0]/10.0
k=viterbi(cm,um)
T=D2.copy()
for j,i in enumerate(mid): T[i]=cm[j][k[j]]
print('changed vs dec2 (axis>20deg):', sum(np.degrees(np.arccos(np.clip(T[i,:3,2]@D2[i,:3,2],-1,1)))>20 for i in mid), 'picked prop cands:', (k>=12).sum())
np.savez('/work/scratch/dec3.npz',Tc=T,T=ep.Twc[None]@T)
pickle.dump(dict(cands=cands,un=un),open('/work/scratch/cands3.pkl','wb'))
t0=time.time()
J=joint_refine(ep,FR,T,[(0,s1),(s2,F-1)]); torch.cuda.synchronize(); print('joint',time.time()-t0)
np.savez('/work/scratch/joint3.npz',Tc=J,T=ep.Twc[None]@J)
