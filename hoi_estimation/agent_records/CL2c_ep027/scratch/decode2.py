import numpy as np, sys
sys.path.insert(0,'/work/outputs')
d=np.load('/work/scratch/run2/solve_debug.npz'); I=d['IOU']; P=d['PH2']; ps=d['psis']; fa,fb=d['phases']; pb=d['psi_b']
def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi
def viterbi(U, psis, s0, s1, lam, lam1):
    T,nb=U.shape; D=np.degrees(np.abs(wrap(psis[:,None]-psis[None,:]))); trans=lam*D**2/100+lam1*D
    e0=np.degrees(np.abs(wrap(psis-s0))); e1=np.degrees(np.abs(wrap(psis-s1)))
    cost=U[0]+lam*e0**2/100+lam1*e0; bp=np.zeros((T,nb),int)
    for t in range(1,T):
        c=cost[:,None]+trans; bp[t]=c.argmin(0); cost=c.min(0)+U[t]
    cost=cost+lam*e1**2/100+lam1*e1; k=int(cost.argmin()); path=[k]
    for t in range(T-1,0,-1): k=bp[t,k]; path.append(k)
    return np.array(path[::-1])
def unary(iou,ph,wi=10,wp=1):
    vis=np.nanmax(iou,1)>0.3; irel=iou-np.nanmax(iou,1,keepdims=True); irel[~vis]=0
    p=np.nan_to_num(ph,nan=0.); p[~vis]=0; return -wi*irel-wp*p
s0=np.arctan2(np.sin(pb[:fa]).mean(),np.cos(pb[:fa]).mean()); s1=np.arctan2(np.sin(pb[fb:]).mean(),np.cos(pb[fb:]).mean())
res={}
for wi,wp,lam,lam1 in [(10,1,0.05,0),(10,1,0.05,0.1),(10,1,0.05,0.15),(10,1,0.05,0.25),(10,0,0.05,0.15),(10,2,0.05,0.15),(20,1,0.05,0.15),(10,1,0.2,0.15)]:
    path=viterbi(unary(I,P,wi,wp),ps,s0,s1,lam,lam1); res[(wi,wp,lam,lam1)]=ps[path]
ref=res[(10,1,0.05,0.15)]
for k,v in res.items():
    dd=np.degrees(np.abs(wrap(v-ref))); print(k,'>45:',(dd>45).sum(),'>20:',(dd>20).sum())
w=lambda x: int(np.degrees(wrap(x)))
print(' '.join('%d:%d'%(fa+j,w(ref[j])) for j in range(0,len(ref),4)))
