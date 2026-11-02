import sys,time; sys.path.insert(0,'/work/scratch/s')
from track import *
from scipy.ndimage import median_filter
def zref_curve(a,b,Ts,Te):
    zb=median_filter(Tcb[:,2,3],size=15,mode='nearest')
    s0=Ts[2,3]/zb[a-1]; s1=Te[2,3]/zb[b+1]
    ts=np.arange(a,b+1); w=(ts-(a-1))/((b+1)-(a-1))
    return dict(zip(ts,zb[a:b+1]*((1-w)*s0+w*s1)))
OPT=dict(iters=40,blurs=(2.0,1.0),w_tex=0.2,w_z=1.0,w_rot=0.02)
def opt_frame(obs,inits,zr):
    R0=torch.tensor(np.stack([T[:3,:3] for T in inits]),dtype=torch.float32,device=dev)
    t0=torch.tensor(np.stack([T[:3,3] for T in inits]),dtype=torch.float32,device=dev)
    R,tt,_,_,_=optimize(rc,[obs],R0,t0,z_ref=torch.tensor(zr,device=dev),**OPT)
    S=score(rc,[obs],R,tt)
    Ts=[T44(R[k].cpu().numpy(),tt[k].cpu().numpy()) for k in range(len(inits))]
    return Ts,S
def chain(frames,Tstart,zr):
    out={}; prev=Tstart; prev2=None
    for t in frames:
        obs=make_obs(t,prev); inits=[prev]
        if prev2 is not None:
            dR=prev[:3,:3]@prev2[:3,:3].T; inits.append(T44(dR@prev[:3,:3],2*prev[:3,3]-prev2[:3,3]))
        Ts,S=opt_frame(obs,inits,zr[t]); k=int(np.argmin(S[:,0])); Tn=Ts[k]
        if not np.isfinite(Tn).all(): Tn=prev.copy()
        out[t]=(Tn,S[k]); prev2=prev; prev=Tn
    return out
def base_cands(frames,zr):
    out={}
    for t in frames:
        Tb=Tcb[t].copy(); Tb[:3,3]*=zr[t]/Tb[2,3]
        obs=make_obs(t,Tb)
        inits=[T44(Tb[:3,:3]@F,Tb[:3,3]) for F in FLIPS]
        Ts,S=opt_frame(obs,inits,zr[t]); out[t]=list(zip(Ts,S))
    return out
if __name__=='__main__':
    a,b=int(sys.argv[1]),int(sys.argv[2])
    st=np.load('/work/scratch/static_res.npy',allow_pickle=True).item()
    Ts=st[sys.argv[3]]['T']; Te=st[sys.argv[4]]['T']
    zr=zref_curve(a,b,Ts,Te)
    t0=time.time()
    fw=chain(range(a,b+1),Ts,zr); print('fw',time.time()-t0)
    bw=chain(range(b,a-1,-1),Te,zr); print('bw',time.time()-t0)
    bc=base_cands(range(a,b+1),zr); print('bc',time.time()-t0)
    np.save(f'/work/scratch/track2_{a}_{b}.npy',dict(fw=fw,bw=bw,bc=bc,zr=zr),allow_pickle=True)
    for t in range(a,b+1,4):
        bS=np.array([x[1] for x in bc[t]])
        print(t,'fw',fw[t][1].round(2),'bw',bw[t][1].round(2),'bc',bS[:,0].round(2),'fw-bw %.0f'%rdist(fw[t][0],bw[t][0]),'fw-base %.0f'%rdist(fw[t][0],Tcb[t]),'bw-base %.0f'%rdist(bw[t][0],Tcb[t]))
