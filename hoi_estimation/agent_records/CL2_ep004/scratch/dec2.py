import sys; sys.path.insert(0,'/work/scratch')
from common import *
import time
ep,A=get_ep(); FR,simp=get_FR(); res=A['res']; F=ep.F
s1,s2,_,_=detect_static_phases(ep.mask_small_o,ep.mask_small_p); print('static',s1,s2)
zb=smooth_baseline_z(ep)
ids=list(range(F)); cands=[r['T'] for r in res]
t0=time.time(); dres=depth_scores(ep,FR,ids,cands); print('depth scores',time.time()-t0)
fh=[floor_heights(ep,cands[i],simp['vs'],zb[i]) for i in ids]
un=[full_unary(ep,res[i]['sc'],dres[i],fh[i]) for i in ids]
import pickle; pickle.dump(dict(dres=dres,fh=fh,un=un),open('/work/scratch/unary2.pkl','wb'))
# static phases: pooled selection: score every candidate pose of the phase against sampled frames of the phase
def static_pose(fr):
    samp=fr[::max(1,len(fr)//12)]
    pool=np.concatenate([cands[i] for i in samp])
    tot=np.zeros(len(pool))
    for i in samp:
        sc=score_frames(ep,FR,[i],[pool])[0]; dr=depth_scores(ep,FR,[i],[pool])[0]
        fhh=floor_heights(ep,pool,simp['vs'],zb[i])
        tot+=full_unary(ep,sc,dr,fhh)
    o=np.argsort(tot); best=pool[o[0]]
    print('static best %.3f next-different-axis:'%(tot[o[0]]/len(samp)), [(round(tot[j]/len(samp),3), int(np.degrees(np.arccos(np.clip(pool[j,:3,2]@best[:3,2],-1,1))))) for j in o[1:40:4]])
    return best
P1=static_pose(list(range(0,s1+1))); P2=static_pose(list(range(s2,F)))
mid=list(range(s1+1,s2))
cm=[cands[i] for i in mid]; um=[un[i].copy() for i in mid]
um[0]=um[0]+pair_cost(P1[None],cm[0])[0]/10.0; um[-1]=um[-1]+pair_cost(cm[-1],P2[None])[:,0]/10.0
k=viterbi(cm,um)
T=np.zeros((F,4,4)); T[:s1+1]=P1; T[s2:]=P2
for j,i in enumerate(mid): T[i]=cm[j][k[j]]
np.savez('/work/scratch/dec2.npz',Tc=T,T=ep.Twc[None]@T)
names=[f'{s}{j}' for s in ['base','fp','cc'] for j in range(4)]
ang=np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i',T[:,:3,2],ep.Tc_base[:,:3,2]),-1,1)))
d1=np.load('/work/scratch/dec1.npz')['Tc']; ang1=np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i',T[:,:3,2],d1[:,:3,2]),-1,1)))
for j,i in enumerate(mid):
    if i%5==0: print(i,names[k[j]],'u %.3f dres %.3f'%(un[i][k[j]],dres[i][k[j]]),'axis-vs-base %.0f vs-dec1 %.0f'%(ang[i],ang1[i]))
