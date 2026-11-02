import sys; sys.path.insert(0,'/work/outputs')
from common import *
K,Kp,Twc=load_camera('/work/inputs'); m=load_mesh('/work/inputs/object_mesh.glb'); r=Renderer(m,K)
f=h5py.File('/work/inputs/masks.h5')
rr=np.load('/work/inputs/reconstruction.npz'); Tc0=np.linalg.inv(Twc)@rr['T_world_mesh'].astype(np.float64)
cxy=np.array([0.005,-0.004,0]); Fz=np.eye(4); Fz[:3,:3]=np.diag([-1,-1,1.]); Fz[:3,3]=2*cxy
frames=[int(x) for x in sys.argv[1].split(',')]; out=sys.argv[2]; sz=int(sys.argv[3]) if len(sys.argv)>3 else 400
rows=[]
for i in frames:
    img=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg'); om=f['object'][i]
    ys,xs=np.nonzero(om); cx,cy=(xs.min()+xs.max())//2,(ys.min()+ys.max())//2; s=int(max(xs.ptp(),ys.ptp())*1.15)+10
    x0,y0=cx-s//2,cy-s//2
    def cr(a): 
        p=cv2.copyMakeBorder(a,s,s,s,s,cv2.BORDER_CONSTANT); return cv2.resize(p[y0+s:y0+2*s,x0+s:x0+2*s],(sz,sz))
    tiles=[cr(img)]
    for T in [Tc0[i],Tc0[i]@Fz]:
        mk,rgb,_,_=r.render(T[None]); rgb=(rgb[0].cpu().numpy()[:,:,::-1]*255).astype(np.uint8).copy()
        tiles.append(cr(rgb))
    cv2.putText(tiles[0],str(i),(5,25),0,0.8,(0,0,255),2); cv2.putText(tiles[1],'base',(5,25),0,0.8,(0,0,255),2); cv2.putText(tiles[2],'flip',(5,25),0,0.8,(0,0,255),2)
    rows.append(np.hstack(tiles))
cv2.imwrite(out,np.vstack(rows))
