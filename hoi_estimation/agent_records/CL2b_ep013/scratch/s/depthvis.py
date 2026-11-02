import sys; sys.path.insert(0,'/work/scratch/s')
from vis import *
c=json.load(open('/work/inputs/camera.json')); K=np.array(c['K']); Twc=np.array(c['T_world_cam'])
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); rc=RC(mesh)
fr=np.load('/work/scratch/frames_full.npy',mmap_mode='r'); fm=h5py.File('/work/inputs/masks.h5')
rec=np.load('/work/inputs/reconstruction.npz'); Tcb=np.linalg.inv(Twc)[None]@rec['T_world_mesh'].astype(float)
res=np.load('/work/scratch/static_res.npy',allow_pickle=True).item()
ph={'A':(0,131),'C':(312,386),'E':(484,701),'G':(817,921)}
tiles=[]
for k,(a,b) in ph.items():
    Tc=res[k]['T']; zb=np.median(Tcb[a:b+1,2,3]); t=res[k]['sel'][2]
    om=fm['object'][t]; ys,xs=np.nonzero(om); cx,cy=xs.mean(),ys.mean(); sz=max(np.ptp(xs),np.ptp(ys))*1.3
    M,_=crop_params(cx,cy,sz,480); Kc=torch.tensor(crop_K(K,M),dtype=torch.float32,device=dev)
    im=warp(np.ascontiguousarray(fr[t]),M,480)
    row=[]
    for s,colr in [(1.0,(0,0,255)),(zb/Tc[2,3],(255,0,0))]:
        T=Tc.copy(); T[:3,3]*=s
        o=rc.render(torch.tensor(T[None,:3,:3],dtype=torch.float32,device=dev),torch.tensor(T[None,:3,3],dtype=torch.float32,device=dev),Kc,480,480)
        m=(o['hard'][0].cpu().numpy()>0).astype(np.uint8)
        cnt,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(im,cnt,-1,colr,1)
    omc=warp(om.astype(np.uint8),M,480,cv2.INTER_NEAREST); cnt,_=cv2.findContours(omc,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(im,cnt,-1,(0,255,0),1)
    cv2.putText(im,f'{k} {t} red=sil z blue=base z green=mask',(5,20),0,0.6,(0,255,255),2)
    tiles.append(im)
cv2.imwrite('/work/scratch/depthvis.jpg',np.vstack([np.hstack(tiles[:2]),np.hstack(tiles[2:])]))
