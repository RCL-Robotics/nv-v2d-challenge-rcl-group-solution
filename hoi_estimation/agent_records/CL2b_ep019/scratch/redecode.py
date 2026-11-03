import sys; sys.path.insert(0,'/work/outputs')
from common import *; from decode import *
import decode
s1=np.load('cache/s1.npy',allow_pickle=True).item(); s2=np.load('cache/s2.npy',allow_pickle=True).item()['res']; s3=np.load('cache/s3.npy',allow_pickle=True).item()['ph']
F=886
cands=[s1['T'][i] for i in range(F)]; closs=[s1['L'][i] for i in range(F)]; ph=list(s3)
for r in range(2):
    pr=np.load(f'cache/s4_{r}.npy',allow_pickle=True).item()
    cands=[np.concatenate([cands[i],pr['T'][i]]) for i in range(F)]
    closs=[np.concatenate([closs[i],pr['L'][i]]) for i in range(F)]
    ph=[np.concatenate([ph[i],pr['ph'][i]]) for i in range(F)]
def run(quad,sr,st):
    cs,dc=[],[]
    for i in range(F): cs.append(cands[i]); dc.append(closs[i]*4+ph[i]*50)
    for r in s2:
        a,b=r['phase']
        for i in range(a,b+1): cs[i]=r['T'][None]; dc[i]=np.zeros(1)
    if quad:
        # monkeypatch quadratic transition
        def vit(cands,data_cost,sig_rot,sig_tr):
            acc=[np.asarray(data_cost[0],float)];back=[]
            for f in range(1,len(cands)):
                A,B=cands[f-1],cands[f]
                dr=rot_dist(A[:,:3,:3],B[:,:3,:3]); dcc=np.linalg.norm(centres(A)[:,None]-centres(B)[None],axis=-1)
                tot=acc[-1][:,None]+(dr/sig_rot)**2+(dcc/sig_tr)**2
                bi=tot.argmin(0);back.append(bi);acc.append(tot[bi,np.arange(len(B))]+data_cost[f])
            p=[int(acc[-1].argmin())]
            for f in range(len(cands)-2,-1,-1): p.append(int(back[f][p[-1]]))
            return p[::-1]
        path=vit(cs,dc,sr,st)
    else: path=viterbi(cs,dc,sig_rot=sr,sig_tr=st)
    P=np.array([cs[i][path[i]] for i in range(F)]); D=np.array([dc[i][path[i]] for i in range(F)])
    dR=rot_angle(P[1:,:3,:3],P[:-1,:3,:3])
    print(quad,sr,st,'data cost sum %.0f'%D.sum(),'jumps>15',[(int(i),int(dR[i])) for i in np.where(dR>15)[0]])
    return P
run(False,6,0.04)
for sr in [6,8,10]:
    P=run(True,sr,0.03)
    np.save(f'P1q_{sr}.npy',P)
