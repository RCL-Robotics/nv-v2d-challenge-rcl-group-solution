import sys,pickle; sys.path.insert(0,'/work/outputs')
from common import *
out=pickle.load(open('/work/scratch/keys.pkl','rb'))
fr=sorted(out); N=len(fr)
U=[];C=[]
for f in fr:
    R,t,s,c=out[f]; U.append(20*(s+0.3*c)); C.append((R,t))
def trans(a,b):
    Ra,ta=a; Rb,tb=b
    A=np.array([[rot_angle_deg(Ra[i],Rb[j]) for j in range(len(Rb))] for i in range(len(Ra))])
    return (A/10.0)**2
cost=U[0].copy(); bp=[]
for k in range(1,N):
    T=trans(C[k-1],C[k]); tot=cost[:,None]+T; b=np.argmin(tot,0); bp.append(b); cost=tot[b,np.arange(len(b))]+U[k]
path=[int(np.argmin(cost))]
for b in reversed(bp): path.append(int(b[path[-1]]))
path=path[::-1]
greedy=[int(np.argmin(u)) for u in U]
Rsel=np.array([C[k][0][path[k]] for k in range(N)]); tsel=np.array([C[k][1][path[k]] for k in range(N)])
for k in range(N):
    s=out[fr[k]][2][path[k]]; c=out[fr[k]][3][path[k]]
    dr_=rot_angle_deg(Rsel[k],Rsel[k-1]) if k else 0
    print(fr[k],'vit',path[k],'greedy',greedy[k],'sil %.3f col %.3f'%(s,c),'dR %.1f'%dr_,'t',tsel[k].round(3))
np.savez('/work/scratch/vit.npz',fr=np.array(fr),R=Rsel,t=tsel)
