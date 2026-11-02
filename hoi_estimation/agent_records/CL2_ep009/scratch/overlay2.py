# overlay final vs mask for chosen frames: usage overlay2.py run.npz out.jpg frames...
from rend import *
import sys as _s
r,cam=load(); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam']); Tcw=np.linalg.inv(Twc)
T1=np.load(_s.argv[1])['T_world_mesh']; out=_s.argv[2]; idx=[int(a) for a in _s.argv[3:]]
Tb=r['T_world_mesh'].astype(float)
M=Mesh(); MK=Masks(); S=160
fr=read_frames(idx); tiles=[]
for i in idx:
    Tc=Tcw@T1[i]; Tcb=Tcw@Tb[i]
    uv=K@Tc[:3,3]; uv=uv[:2]/uv[2]
    x0=int(np.clip(uv[0]-S/2,0,1536-S)); y0=int(np.clip(uv[1]-S/2,0,1152-S))
    im=fr[i][y0:y0+S,x0:x0+S].copy(); im=cv2.resize(im,(S*3,S*3),interpolation=cv2.INTER_CUBIC)
    o,pm=MK.crop(i,x0,y0,S); valid=~pm|o
    txt=[]
    for T,col in [(Tcb,(0,255,255)),(Tc,(0,255,0))]:
        a,_=render_sil(M,torch.tensor(T[None,:3,:3],dtype=torch.float32,device=DEV),torch.tensor(T[None,:3,3],dtype=torch.float32,device=DEV),K,torch.tensor([[x0,y0]],device=DEV,dtype=torch.float32),S)
        a=a[0].cpu().numpy()>0.5
        txt.append((a&o&valid).sum()/max(((a|o)&valid).sum(),1))
        c,_=cv2.findContours(cv2.resize(a.astype(np.uint8),(S*3,S*3),interpolation=cv2.INTER_NEAREST),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(im,c,-1,col,1)
    c,_=cv2.findContours(cv2.resize(o.astype(np.uint8),(S*3,S*3),interpolation=cv2.INTER_NEAREST),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(im,c,-1,(255,0,0),1)
    cv2.putText(im,'%d base %.2f new %.2f'%(i,*txt),(5,20),0,0.6,(255,255,255),2); tiles.append(im)
n=len(tiles); cols=min(n,4)
while len(tiles)%cols: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(out,np.vstack([np.hstack(tiles[k:k+cols]) for k in range(0,len(tiles),cols)]))
