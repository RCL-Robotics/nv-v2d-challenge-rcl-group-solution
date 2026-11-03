import sys; sys.path.insert(0,'/work/scratch/s')
from common import *
K,Twc,Kp=load_cam(); mesh=load_mesh(); R=Renderer(mesh)
rec=np.load(f'{INP}/reconstruction.npz')
idx=[int(x) for x in sys.argv[1].split(',')]
which=sys.argv[2]; out=sys.argv[3]
Ts=np.load(sys.argv[4])['T_world_mesh'] if len(sys.argv)>4 else None
fm=h5py.File(f'{INP}/masks.h5')
c=cv2.VideoCapture(f'{INP}/video.mp4'); frames={}; j=0
while len(frames)<len(idx):
    ok,im=c.read()
    if j in idx: frames[j]=im
    j+=1
tiles=[]
for i in idx:
    if which=='base': T=np.linalg.inv(Twc)@rec['T_world_mesh'][i]; KK=K
    elif which=='fp': T=rec['T_cam_mesh_foundationpose'][i]; KK=Kp
    elif which=='cc': T=rec['T_cam_mesh_coconet'][i]; KK=Kp
    else: T=np.linalg.inv(Twc)@Ts[i]; KK=K
    o=R.render(T,KK,W,H)
    col=(o['color'][0].cpu().numpy()*255).astype(np.uint8); m=o['mask'][0].cpu().numpy()
    im=frames[i].copy()
    om=fm['object'][i]; ys,xs=np.nonzero(om)
    if len(xs)==0: ys,xs=np.nonzero(m)
    cy,cx=(int(ys.mean()),int(xs.mean())) if len(xs) else (576,768)
    blend=im.copy(); blend[m]=(0.35*im[m]+0.65*col[m]).astype(np.uint8)
    cnt,_=cv2.findContours(om.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(blend,cnt,-1,(0,255,0),1)
    s=130
    def cr(a): 
        a=np.pad(a,((s,s),(s,s),(0,0))); return cv2.resize(a[cy:cy+2*s,cx:cx+2*s],(300,300))
    t=np.hstack([cr(im),cr(blend)]); cv2.putText(t,str(i),(5,25),0,0.8,(0,255,0),2); tiles.append(t)
while len(tiles)%3: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(out,np.vstack([np.hstack(tiles[k:k+3]) for k in range(0,len(tiles),3)]))
