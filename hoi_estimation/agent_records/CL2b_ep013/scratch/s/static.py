import sys,time; sys.path.insert(0,'/work/scratch/s')
from vis import *
c=json.load(open('/work/inputs/camera.json')); K=np.array(c['K']); Twc=np.array(c['T_world_cam'])
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); rc=RC(mesh)
fr=np.load('/work/scratch/frames_full.npy',mmap_mode='r'); f=h5py.File('/work/inputs/masks.h5')
rec=np.load('/work/inputs/reconstruction.npz')
Tcb=np.linalg.inv(Twc)[None]@rec['T_world_mesh'].astype(float)
phases={'A':(0,131),'C':(312,386),'E':(484,701),'G':(817,921)}
st=np.load('/work/scratch/maskstats.npy')
out={}
items=[]
for name,(a,b) in phases.items():
    t0=time.time()
    fs=np.arange(a,b+1)
    # choose frames with smallest person-near-object fraction, spread out
    occ=st[fs,7]; area=st[fs,0]
    good=fs[np.argsort(occ+ (area<0.8*np.median(area))*10)[:max(6,len(fs)//4)]]
    sel=np.sort(good)[np.linspace(0,len(good)-1,6).astype(int)]
    obs=[]
    for t in sel:
        om=f['object'][t]; ys,xs=np.nonzero(om)
        obs.append(FrameObs(fr[t],om,f['person'][t],K,xs.mean(),ys.mean(),max(np.ptp(xs),np.ptp(ys))*1.6,out=192))
    t_c=np.median(Tcb[sel,:3,3],0)
    om=f['object'][sel[0]]; ys,xs=np.nonzero(om); ray=np.linalg.inv(K)@[xs.mean(),ys.mean(),1]; tc=ray/ray[2]*t_c[2]
    extra=np.concatenate([Tcb[sel[0],:3,:3][None]@FLIPS])
    R,t,S=global_search(rc,obs,tc,extra_R=extra)
    print(name,sel,'time %.1f'%(time.time()-t0))
    for k in range(6): print('  ',k,S[k].round(3),'ang base %.1f'%np.degrees(Rot.from_matrix(R[k].T@Tcb[sel[0],:3,:3]).magnitude()),'ang best %.1f'%np.degrees(Rot.from_matrix(R[k].T@R[0]).magnitude()),'t',t[k].round(3),'base z med %.3f'%np.median(Tcb[a:b+1,2,3]))
    T=np.eye(4); T[:3,:3]=R[0]; T[:3,3]=t[0]; out[name]=dict(T=T,cands=(R[:12],t[:12],S[:12]),sel=sel)
    T2=np.eye(4); T2[:3,:3]=R[1]; T2[:3,3]=t[1]
    items.append((sel[0],[Tcb[sel[0]],T,T2],f'{name} base|best|2nd'))
    items.append((sel[-1],[Tcb[sel[-1]],T,T2],f'{name} base|best|2nd'))
np.save('/work/scratch/static_res.npy',out,allow_pickle=True)
vis_poses(rc,fr,f,K,items,'/work/scratch/static_vis.jpg')
