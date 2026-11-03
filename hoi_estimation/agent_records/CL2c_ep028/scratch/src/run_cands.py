import time,sys
from opt import *
b=load_bank_cache(); prob=Problem(b)
Rf,tf=fp_rescaled()
FL=flips()
def hyps(fr):
    R0=[];t0=[]
    for RR,tt,refl in [(Rb,tb,True),(Rf,tf,False)]:
        for Fm in FL:
            R=RR[fr]@Fm; R0.append(R); t0.append(tt[fr])
            if refl: R0.append(tilt_reflect(R,tt[fr])); t0.append(tt[fr])
    return np.stack(R0,1),np.stack(t0,1)
t0_=time.time()
BS=int(sys.argv[1]) if len(sys.argv)>1 else 32
allR=[];allt=[];allL=[];allc={}
for s in range(0,F,BS):
    fr=np.arange(s,min(F,s+BS))
    R0,t0=hyps(fr)
    R,t,L,ct=optimize(prob,fr,R0,t0,tb[fr,2][:,None])
    allR.append(R);allt.append(t);allL.append(L)
    for k,v in ct.items(): allc.setdefault(k,[]).append(v)
    if s==0: print('batch time',time.time()-t0_, torch.cuda.max_memory_allocated()/1e9,flush=True)
np.savez('../cands.npz',R=np.concatenate(allR),t=np.concatenate(allt),L=np.concatenate(allL),**{k:np.concatenate(v) for k,v in allc.items()})
print('total',time.time()-t0_)
