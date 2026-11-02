import sys,time; sys.path.insert(0,'/work/scratch/s')
from rc import *
c=json.load(open('/work/inputs/camera.json')); K=np.array(c['K']); Twc=np.array(c['T_world_cam'])
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
rc=RC(mesh)
fr=np.load('/work/scratch/frames_full.npy',mmap_mode='r'); f=h5py.File('/work/inputs/masks.h5')
rec=np.load('/work/inputs/reconstruction.npz')
res={}
for t in [int(x) for x in sys.argv[1].split(',')]:
    t0=time.time()
    Tc=np.linalg.inv(Twc)@rec['T_world_mesh'][t]
    om=f['object'][t]; ys,xs=np.nonzero(om)
    obs=FrameObs(fr[t],om,f['person'][t],K,xs.mean(),ys.mean(),max(np.ptp(xs),np.ptp(ys))*1.6,out=192)
    # center along ray through mask centroid at baseline depth
    z=Tc[2,3]; ray=np.linalg.inv(K)@[xs.mean(),ys.mean(),1]; tc=ray*z
    R,tt,S=global_search(rc,[obs],tc); tot,ls,lt=S.T
    for k in range(5):
        print(t,k,'loss %.3f sil %.3f tex %.3f'%(tot[k],ls[k],lt[k]),'ang to base %.1f'%np.degrees(Rot.from_matrix(R[k].T@Tc[:3,:3]).magnitude()),'t',tt[k].round(3),'base t',Tc[:3,3].round(3))
    print('time',time.time()-t0)
    T=np.eye(4); T[:3,:3]=R[0]; T[:3,3]=tt[0]; res[t]=Twc@T
np.save('/work/scratch/gs_test.npy',res)
