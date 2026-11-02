import sys, time; sys.path.insert(0,'/work/outputs')
import numpy as np, torch, cv2, h5py, json
from bat_lib import *; import fitter; from fitter import *
K,Twc,rec,plane=load_inputs('/work/inputs'); Kp=np.array(json.load(open('/work/inputs/camera.json'))['K_pipeline'])
s=np.load('/work/scratch/simp.npz'); FR=FastRenderer(s['vs'],s['fs'],s['cs'])
fp=rec['T_cam_mesh_foundationpose'].astype(np.float64)
f=h5py.File('/work/inputs/masks.h5'); cap=cv2.VideoCapture('/work/inputs/video.mp4')
i=600; cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,img=cap.read()
om=f['object'][i]; pm=f['person'][i]
T0=convert_K(fp[i],Kp,K,0.989)
hint=(K@T0[:3,3]); hint=hint[:2]/hint[2]
win=crop_window(om,hint,0.54*K[0,0]/T0[2,3]); d=prep_frame(img,om,pm,win)
print(win, d['M'].sum(), d['P'].sum(), d['I'].mean())
for it in [0,1,5,20,50,100]:
    T,tot,li,lc,ld=fit_batch(FR,K,T0[None],[win],[d],iters=it) if it>0 else (T0[None],*score_batch(FR,K,T0[None],[win],[d]))
    print(it, tot, 1-li, lc, ld, np.round(T[0,:3,3],3))
Kc=crop_K(K,win)
R=torch.tensor(T0[None,:3,:3],dtype=torch.float32,device='cuda'); t=torch.tensor(T0[None,:3,3],dtype=torch.float32,device='cuda')
S,C,_=FR.render(R,t,Kc[None],RES)
vis=np.hstack([d['M'],d['P'],S[0].cpu().numpy(),d['D']*5]); cv2.imwrite('/work/scratch/dbg.png',(np.clip(vis,0,1)*255).astype(np.uint8))
cv2.imwrite('/work/scratch/dbg_I.png',(d['I'][...,::-1]*255).astype(np.uint8))
