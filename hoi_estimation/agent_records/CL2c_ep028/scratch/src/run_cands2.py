import time,sys
from opt import *
b=load_bank_cache(); prob=Problem(b)
def Rz(a):
    c,s=np.cos(a),np.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1.]])
ENDFLIP=np.diag([1.,-1,-1])
def hyps(fr):
    R0=[];t0=[]
    for e in [np.eye(3),ENDFLIP]:
        for k in range(8):
            R=Rb[fr]@e@Rz(k*np.pi/4); R0.append(R); t0.append(tb[fr])
            if k%2==0: R0.append(tilt_reflect(R,tb[fr])); t0.append(tb[fr])
    return np.stack(R0,1),np.stack(t0,1)
t0_=time.time()
BS=16
allR=[];allt=[];allL=[];allc={}
for s in range(0,F,BS):
    fr=np.arange(s,min(F,s+BS))
    R0,t0=hyps(fr)
    R,t,L,ct=optimize(prob,fr,R0,t0,tb[fr,2][:,None],steps=80)
    allR.append(R);allt.append(t);allL.append(L)
    for k,v in ct.items(): allc.setdefault(k,[]).append(v)
    if s==0: print('batch time',time.time()-t0_, torch.cuda.max_memory_allocated()/1e9,flush=True)
np.savez('../cands2.npz',R=np.concatenate(allR),t=np.concatenate(allt),L=np.concatenate(allL),**{k:np.concatenate(v) for k,v in allc.items()})
print('total',time.time()-t0_)
