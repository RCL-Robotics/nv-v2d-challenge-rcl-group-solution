import numpy as np, sys
d=np.load('/work/scratch/yawscan.npz'); IoU=d['IoU']; PH=d['PH']; NP=d['NP']; psis=d['psis']; pb=d['psi_b']
CA=(150,452); nb=len(psis)
def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi
def viterbi(U, psi_start, psi_end, lam):
    T=U.shape[0]
    D=np.abs(wrap(psis[:,None]-psis[None,:]))
    trans=lam*np.degrees(D)**2/100.  # cost per (10 deg)^2
    cost=U[0]+lam*np.degrees(np.abs(wrap(psis-psi_start)))**2/100.
    bp=np.zeros((T,nb),int)
    for t in range(1,T):
        c=cost[:,None]+trans
        bp[t]=c.argmin(0); cost=c.min(0)+U[t]
    cost=cost+lam*np.degrees(np.abs(wrap(psis-psi_end)))**2/100.
    k=cost.argmin(); path=[k]
    for t in range(T-1,0,-1): k=bp[t,k]; path.append(k)
    return np.array(path[::-1])
a,b=CA
I=IoU[a:b]; P=np.nan_to_num(PH[a:b],nan=0.)
vis=np.nanmax(IoU[a:b],1)>0.3
Irel=I-np.nanmax(I,1,keepdims=True)  # 0 at best
Irel=np.where(np.isnan(Irel),0,Irel)
Irel[~vis]=0
alpha,beta,lam=float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3])
U=-alpha*Irel*0+alpha*(-Irel)*(-1)  # placeholder
U=alpha*(-Irel) - beta*P
psi_fl=np.arctan2(np.sin(pb[:150]).mean(),np.cos(pb[:150]).mean()); psi_dk=np.arctan2(np.sin(pb[452:]).mean(),np.cos(pb[452:]).mean())
path=viterbi(U,psi_fl,psi_dk,lam)
ps=psis[path]
for t in range(0,b-a,6):
    i=a+t; print(i,'dec %4.0f  base %4.0f  diff %4.0f  iou %.2f (best %.2f) ph %.2f'%(np.degrees(wrap(ps[t])),np.degrees(wrap(pb[i])),np.degrees(wrap(ps[t]-pb[i])),I[t,path[t]],np.nanmax(I[t]),P[t,path[t]]))
np.save('/work/scratch/decoded_psi.npy',ps)
