import sys, numpy as np, torch, cv2, json, h5py
sys.path.insert(0,'/work/outputs')
from hoilib import *
fr=np.load('/work/scratch/frames_all.npy',mmap_mode='r')
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam'])
mesh=TexMesh('/work/inputs/object_mesh.glb')
masks=h5py.File('/work/inputs/masks.h5')
def load_traj(p):
    if p=='base': return np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(np.float64)
    return np.load(p)['T_world_mesh']
trajs=[load_traj(p) for p in sys.argv[2].split(',')]
frames=[int(x) for x in sys.argv[3].split(',')]
out=sys.argv[1]; S=int(sys.argv[4]) if len(sys.argv)>4 else 200
rows=[]
for i in frames:
    om=masks['object'][i]
    ys,xs=np.nonzero(om)
    if len(xs): cx,cy=(xs.min()+xs.max())//2,(ys.min()+ys.max())//2
    else:
        Tc=np.linalg.inv(Twc)@trajs[0][i]; p=K@Tc[:3,3]; cx,cy=int(p[0]/p[2]),int(p[1]/p[2])
    x0,y0=int(cx-S//2),int(cy-S//2)
    img=np.zeros((S,S,3),np.uint8)
    xa,ya=max(x0,0),max(y0,0); xb,yb=min(x0+S,1536),min(y0+S,1152)
    img[ya-y0:yb-y0,xa-x0:xb-x0]=fr[i,ya:yb,xa:xb]
    tiles=[img.copy()]
    mm=np.zeros((S,S),np.uint8); mm[ya-y0:yb-y0,xa-x0:xb-x0]=om[ya:yb,xa:xb]
    pm=np.zeros((S,S),np.uint8); pm[ya-y0:yb-y0,xa-x0:xb-x0]=masks['person'][i][ya:yb,xa:xb]
    t0=img.copy(); cs,_=cv2.findContours(mm,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(t0,cs,-1,(0,255,0),1)
    cs,_=cv2.findContours(pm,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(t0,cs,-1,(255,0,255),1)
    tiles=[t0]
    for T in trajs:
        Tc=torch.tensor(np.linalg.inv(Twc)@T[i],dtype=torch.float32,device='cuda')[None]
        rgb,msk,_=render(mesh,Tc,K,S,S,crop=(x0,y0,S,S))
        r=(rgb[0].cpu().numpy()[...,::-1]*255).astype(np.uint8); mk=msk[0].cpu().numpy()
        blend=img.copy(); blend[mk]=(0.5*img[mk]+0.5*r[mk]).astype(np.uint8)
        tiles.append(r); 
        t2=img.copy(); cs,_=cv2.findContours(mk.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(t2,cs,-1,(0,0,255),1); tiles.append(t2)
    row=np.hstack(tiles); cv2.putText(row,str(i),(3,15),0,0.5,(0,255,255),1); rows.append(row)
im=np.vstack(rows)
im=cv2.resize(im,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST) if S<=120 else im
cv2.imwrite(out,im)
