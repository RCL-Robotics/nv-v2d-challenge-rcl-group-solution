import sys; sys.path.insert(0,'/work/outputs')
from lib import *
cam,K,Twc,rec=load_inputs('/work/inputs')
key=sys.argv[1]
if key=='base': T=rec['T_world_mesh'].astype(np.float64); Tc=np.linalg.inv(Twc)@T
else: Tc=np.load(key)['T_cam']
cap=cv2.VideoCapture('/work/inputs/video.mp4')
rd=Renderer('/work/inputs/object_mesh.glb',K,1152,1536)
ids=[int(a) for a in sys.argv[3:]]
ms=load_masks('/work/inputs',scale=1,idx=ids)
rows=[]
for j,i in enumerate(ids):
    cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,im=cap.read()
    o=rd.render(Tc[i:i+1]); m=o['mask'][0].cpu().numpy(); c=(o['color'][0].cpu().numpy()[:,:,::-1]*255)
    om=ms['object'][j]
    ys,xs=np.nonzero(om|m); cx,cy=(xs.min()+xs.max())//2,(ys.min()+ys.max())//2
    hs=max(xs.max()-xs.min(),ys.max()-ys.min())//2+30
    x0,y0=max(cx-hs,0),max(cy-hs,0); x1,y1=x0+2*hs,y0+2*hs
    rimg=np.full_like(im,255); rimg[m]=c[m]
    ov=im.copy(); ov[m]=(0.5*im[m]+0.5*c[m]).astype(np.uint8)
    cont=cv2.findContours(om.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]; cv2.drawContours(ov,cont,-1,(0,255,0),1)
    row=np.concatenate([im[y0:y1,x0:x1],rimg[y0:y1,x0:x1],ov[y0:y1,x0:x1]],1)
    row=cv2.resize(row,(900,300)); cv2.putText(row,str(i),(5,20),0,0.7,(0,0,255),2); rows.append(row)
cv2.imwrite(sys.argv[2],np.concatenate(rows,0))
