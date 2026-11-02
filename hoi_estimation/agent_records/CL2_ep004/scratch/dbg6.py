import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, torch, cv2, h5py, json
from bat_lib import *; import fitter; from fitter import *
K,Twc,rec,plane=load_inputs('/work/inputs'); Kp=np.array(json.load(open('/work/inputs/camera.json'))['K_pipeline'])
s=np.load('/work/scratch/simp.npz'); FR=FastRenderer(s['vs'],s['fs'],s['cs'])
fp=rec['T_cam_mesh_foundationpose'].astype(np.float64)
f=h5py.File('/work/inputs/masks.h5'); cap=cv2.VideoCapture('/work/inputs/video.mp4')
i=600; T0=convert_K(fp[i],Kp,K,0.989)
cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,img=cap.read(); om=f['object'][i]; pm=f['person'][i]
hint=(K@T0[:3,3]); hint=hint[:2]/hint[2]
win=crop_window(om,hint,0.54*K[0,0]/T0[2,3]); d=prep_frame(img,om,pm,win)
orig=fitter.losses
log=[]
def wrapped(*a,**k):
    out=orig(*a,**k); log.append([o[0].item() for o in out]); return out
fitter.losses=wrapped
fitter.fit_batch(FR,K,T0[None],[win],[d],iters=30)
for l in log: print(np.round(l,4))
