import sys, numpy as np, cv2, h5py
from rutil import *
R=Renderer('/work/inputs/object_mesh.glb')
K,Twc,Kp=load_cam(); Tcw=np.linalg.inv(Twc)
r=np.load('/work/inputs/reconstruction.npz')
mf=h5py.File('/work/inputs/masks.h5')
def crop_panel(i, poses, Ks, S=200, scale=2):
    img=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg')
    m=mf['object'][i]
    ys,xs=np.nonzero(m)
    if len(xs)==0:
        # use projection of first pose
        p=Ks[0]@poses[0][:3,3]; cx,cy=p[0]/p[2],p[1]/p[2]
    else: cx,cy=xs.mean(),ys.mean()
    x0=int(cx-S/2); y0=int(cy-S/2)
    x0=max(0,min(1536-S,x0)); y0=max(0,min(1152-S,y0))
    crop=img[y0:y0+S,x0:x0+S].copy()
    out=[crop]
    mc=(m[y0:y0+S,x0:x0+S]*255).astype(np.uint8)
    pc=(mf['person'][i][y0:y0+S,x0:x0+S]*120).astype(np.uint8)
    out.append(np.dstack([mc,pc,np.zeros_like(mc)]))
    for T,KK in zip(poses,Ks):
        c,mm,d=R.render(T,KK,S,S,x0,y0)
        c=(c[0].cpu().numpy()[...,::-1]*255).astype(np.uint8); mm=mm[0].cpu().numpy()
        o=crop.copy(); o[mm]=(0.3*o[mm]+0.7*c[mm]).astype(np.uint8)
        out.append(o)
    pan=np.hstack(out)
    pan=cv2.resize(pan,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST)
    cv2.putText(pan,str(i),(5,25),0,0.8,(0,0,255),2)
    return pan
if __name__=='__main__':
    frames=[int(a) for a in sys.argv[2].split(',')]
    pans=[]
    for i in frames:
        Tb=Tcw@r['T_world_mesh'][i].astype(np.float64)
        pans.append(crop_panel(i,[Tb,r['T_cam_mesh_foundationpose'][i],r['T_cam_mesh_coconet'][i]],[K,Kp,Kp],S=160,scale=1.5))
    cv2.imwrite(sys.argv[1],np.vstack(pans))
