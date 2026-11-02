import sys,time; sys.path.insert(0,'/work/scratch/s')
from vis import *
c=json.load(open('/work/inputs/camera.json')); K=np.array(c['K']); Twc=np.array(c['T_world_cam']); Kp=np.array(c['K_pipeline'])
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); rc=RC(mesh)
DIAM=np.linalg.norm(mesh.extents)
fr=np.load('/work/scratch/frames_full.npy',mmap_mode='r'); fm=h5py.File('/work/inputs/masks.h5')
rec=np.load('/work/inputs/reconstruction.npz')
Tcb=np.linalg.inv(Twc)[None]@rec['T_world_mesh'].astype(float)
Rfp=rec['T_cam_mesh_foundationpose'][:,:3,:3].astype(float)
def make_obs(t, Tpred):
    om=fm['object'][t]; ys,xs=np.nonzero(om)
    p=K@Tpred[:3,3]; pc=p[:2]/p[2]; dpx=K[0,0]*DIAM/Tpred[2,3]
    if len(xs)>30:
        cx=0.5*(xs.min()+xs.max()); cy=0.5*(ys.min()+ys.max()); ext=max(np.ptp(xs),np.ptp(ys))
        if np.hypot(cx-pc[0],cy-pc[1])>dpx: cx,cy=pc
        size=max(ext*1.6,dpx*1.2)
    else: cx,cy=pc; size=dpx*1.5
    return FrameObs(fr[t],om,fm['person'][t],K,cx,cy,size,out=160)
def ortho(R):
    U,_,Vt=np.linalg.svd(R); D=np.diag([1,1,np.linalg.det(U@Vt)]); return U@D@Vt
def T44(R,t): T=np.eye(4); T[:3,:3]=ortho(R); T[:3,3]=t; return T
def rdist(A,B): return np.degrees(Rot.from_matrix(A[:3,:3].T@B[:3,:3]).magnitude())
def run_chain(frames, Tstart, iters=50):
    out={}; prev=Tstart; prev2=None
    for t in frames:
        obs=make_obs(t,prev)
        inits=[prev]
        if prev2 is not None:
            # constant velocity
            dR=prev[:3,:3]@prev2[:3,:3].T; inits.append(T44(dR@prev[:3,:3],2*prev[:3,3]-prev2[:3,3]))
        R0=torch.tensor(np.stack([T[:3,:3] for T in inits]),dtype=torch.float32,device=dev)
        t0=torch.tensor(np.stack([T[:3,3] for T in inits]),dtype=torch.float32,device=dev)
        R,tt,tot,ls,lt=optimize(rc,[obs],R0,t0,iters=iters,blurs=(2.0,1.0))
        S=score(rc,[obs],R,tt); k=int(np.argmin(S[:,0]))
        Tn=T44(R[k].cpu().numpy(),tt[k].cpu().numpy())
        if not np.isfinite(Tn).all() or Tn[2,3]<0.5 or Tn[2,3]>10: Tn=prev.copy(); print('bad',t)
        out[t]=(Tn,S[k]); prev2=prev; prev=Tn
    return out
if __name__=='__main__':
    a,b=int(sys.argv[1]),int(sys.argv[2])
    st=np.load('/work/scratch/static_res.npy',allow_pickle=True).item()
    Ts=st[sys.argv[3]]['T']; Te=st[sys.argv[4]]['T']
    t0=time.time()
    fw=run_chain(range(a,b+1),Ts); print('fw',time.time()-t0)
    bw=run_chain(range(b,a-1,-1),Te); print('bw',time.time()-t0)
    np.save(f'/work/scratch/track_{a}_{b}.npy',dict(fw=fw,bw=bw),allow_pickle=True)
    for t in range(a,b+1,4):
        print(t,'fw',fw[t][1].round(3),'bw',bw[t][1].round(3),'fw-bw %.1f'%rdist(fw[t][0],bw[t][0]),'fw-base %.1f'%rdist(fw[t][0],Tcb[t]),'bw-base %.1f'%rdist(bw[t][0],Tcb[t]), 'z fw %.2f bw %.2f base %.2f'%(fw[t][0][2,3],bw[t][0][2,3],Tcb[t,2,3]))
