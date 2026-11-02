from rend import *
r,cam=load(); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam'])
Tc=np.linalg.inv(Twc)[None]@r['T_world_mesh'].astype(float)
M=Mesh(); MK=Masks(); S=192
idx=[0,100,180,250,380,420,520,600,650,700,750,820]
fr=read_frames(idx)
tiles=[]
for i in idx:
    p=Tc[i,:3,3]; uv=K@p; uv=uv[:2]/uv[2]
    x0=int(np.clip(uv[0]-S/2,0,1536-S)); y0=int(np.clip(uv[1]-S/2,0,1152-S))
    R=torch.tensor(Tc[i:i+1,:3,:3],dtype=torch.float32,device=DEV); t=torch.tensor(Tc[i:i+1,:3,3],dtype=torch.float32,device=DEV)
    a,_=render_sil(M,R,t,K,torch.tensor([[x0,y0]],device=DEV,dtype=torch.float32),S)
    a=a[0].cpu().numpy()>0.5
    o,pm=MK.crop(i,x0,y0,S); valid=~pm|o
    iou=(a&o&valid).sum()/max(((a|o)&valid).sum(),1)
    im=fr[i][y0:y0+S,x0:x0+S].copy()
    c1,_=cv2.findContours(a.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(im,c1,-1,(0,255,0),1)
    c2,_=cv2.findContours(o.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(im,c2,-1,(255,0,0),1)
    c3,_=cv2.findContours(pm.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(im,c3,-1,(0,255,255),1)
    im=cv2.resize(im,(384,384),interpolation=cv2.INTER_NEAREST)
    cv2.putText(im,'%d iou %.2f'%(i,iou),(5,20),0,0.6,(255,255,255),2); tiles.append(im)
cv2.imwrite('viz/base_overlay.jpg',np.vstack([np.hstack(tiles[k:k+6]) for k in range(0,12,6)]))
