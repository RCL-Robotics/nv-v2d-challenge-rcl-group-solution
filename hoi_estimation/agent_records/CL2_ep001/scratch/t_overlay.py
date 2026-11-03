import sys; sys.path.insert(0,'/work/outputs')
from lib import *
cam,K,Twc,rec=load_inputs('/work/inputs')
fr=np.load('/work/scratch/frames/half.npy',mmap_mode='r')
s=0.5; Ks=K.copy(); Ks[:2]*=s
H,W=fr.shape[1:3]
rd=Renderer('/work/inputs/object_mesh.glb',Ks,H,W)
T=rec['T_world_mesh'].astype(np.float64); Tc=np.linalg.inv(Twc)@T
ids=[int(a) for a in sys.argv[1:]]
ms=load_masks('/work/inputs',idx=ids)
for j,i in enumerate(ids):
    o=rd.render(Tc[i:i+1])
    m=o['mask'][0].cpu().numpy(); c=o['color'][0].cpu().numpy()
    om=ms['object'][j]
    iou=(m&om).sum()/max((m|om).sum(),1)
    im=fr[i].copy().astype(np.float32)
    ov=im.copy(); ov[m]=0.4*im[m]+0.6*c[m][:,::-1]*255
    ys,xs=np.nonzero(om|m); x0,x1,y0,y1=max(xs.min()-60,0),xs.max()+60,max(ys.min()-60,0),ys.max()+60
    cont=cv2.findContours(om.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]
    ov=ov.astype(np.uint8); cv2.drawContours(ov,cont,-1,(0,255,0),1)
    crop=np.concatenate([im[y0:y1,x0:x1].astype(np.uint8),ov[y0:y1,x0:x1]],1)
    crop=cv2.resize(crop,None,fx=2,fy=2)
    cv2.putText(crop,f'{i} iou {iou:.2f}',(5,20),0,0.6,(0,0,255),2)
    cv2.imwrite(f'/work/scratch/viz/ov_{i:04d}.png',crop); print(i,iou)
