import sys, time; sys.path.insert(0,'/work/outputs')
import numpy as np, torch, cv2, h5py, json
from bat_lib import *; from fitter import *
K,Twc,rec,plane=load_inputs('/work/inputs'); Kp=np.array(json.load(open('/work/inputs/camera.json'))['K_pipeline'])
s=np.load('/work/scratch/simp.npz'); FR=FastRenderer(s['vs'],s['fs'],s['cs'])
fp=rec['T_cam_mesh_foundationpose'].astype(np.float64)
f=h5py.File('/work/inputs/masks.h5'); cap=cv2.VideoCapture('/work/inputs/video.mp4')
i=600; T0=convert_K(fp[i],Kp,K,0.989)
cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,img=cap.read(); om=f['object'][i]; pm=f['person'][i]
hint=(K@T0[:3,3]); hint=hint[:2]/hint[2]
win=crop_window(om,hint,0.54*K[0,0]/T0[2,3]); d=prep_frame(img,om,pm,win)
Kc=crop_K(K,win)[None]
data={k: torch.tensor(np.stack([d[k]]), device='cuda') for k in ["M","P","D","I"]}
R=torch.tensor(T0[None,:3,:3],dtype=torch.float32,device='cuda')
def L(t, comp):
    out=losses(FR,R,t,Kc,data); return out[comp].sum()
for comp in [0,1,2,3]:
    t=torch.tensor(T0[None,:3,3],dtype=torch.float32,device='cuda',requires_grad=True)
    l=L(t,comp); l.backward(); g=t.grad.cpu().numpy()[0]
    fd=[]
    for k in range(3):
        e=torch.zeros(1,3,device='cuda'); e[0,k]=1e-3
        with torch.no_grad(): fd.append(((L(t+e,comp)-L(t-e,comp))/2e-3).item())
    print(comp, 'auto',g,'fd',np.array(fd))
