import sys; sys.path.insert(0,'/work/scratch/src')
from common import *
cam,K,Twc,rec=load_inputs(); M=Mesh()
T=rec['T_world_mesh'].astype(float); Tc=np.linalg.inv(Twc)@T
x0,y0,w,h,s=620,740,240,180,2
frames=list(range(0,748,25))
Tt=torch.tensor(Tc[frames],device=dev)
mask,rgb,_=M.render(Tt,K,x0,y0,w,h,s)
om,pm=load_masks(frames)
tiles=[]
for k,i in enumerate(frames):
    im=read_frame(i)[y0:y0+h,x0:x0+w]; im=cv2.resize(im,None,fx=s,fy=s)
    r=(rgb[k].cpu().numpy()[...,::-1]*255).astype(np.uint8); m=mask[k].cpu().numpy()
    ov=im.copy(); ov[m]=(0.4*im[m]+0.6*r[m]).astype(np.uint8)
    o=cv2.resize(om[k][y0:y0+h,x0:x0+w].astype(np.uint8),None,fx=s,fy=s)
    cnt,_=cv2.findContours(o,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(ov,cnt,-1,(0,0,255),1)
    cv2.putText(ov,str(i),(5,25),0,0.8,(0,255,255),2)
    tiles.append(np.hstack([im,ov]))
while len(tiles)%3: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite('an/base_ov.jpg',np.vstack([np.hstack(tiles[k:k+3]) for k in range(0,len(tiles),3)]))
