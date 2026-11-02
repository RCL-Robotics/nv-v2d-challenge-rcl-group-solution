import sys, time; sys.path.insert(0,'/work/outputs')
import numpy as np, torch, cv2, h5py, json
from bat_lib import *; import fitter; from fitter import *
K,Twc,rec,plane=load_inputs('/work/inputs'); Kp=np.array(json.load(open('/work/inputs/camera.json'))['K_pipeline'])
s=np.load('/work/scratch/simp.npz'); FR=FastRenderer(s['vs'],s['fs'],s['cs'])
fp=rec['T_cam_mesh_foundationpose'].astype(np.float64); Tcb=np.linalg.inv(Twc)[None]@rec['T_world_mesh'].astype(np.float64)
f=h5py.File('/work/inputs/masks.h5'); cap=cv2.VideoCapture('/work/inputs/video.mp4')
for i,T0 in [(600,convert_K(fp[i],Kp,K,0.989)) for i in [600]]+[(340,convert_K(fp[340],Kp,K,0.989)),(270,Tcb[270])]:
    cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,img=cap.read()
    om=f['object'][i]; pm=f['person'][i]
    hint=(K@T0[:3,3]); hint=hint[:2]/hint[2]
    win=crop_window(om,hint,0.54*K[0,0]/T0[2,3]); d=prep_frame(img,om,pm,win)
    print(i, score_batch(FR,K,T0[None],[win],[d])[0])
    for lr in [(0.01,0.003,0.003),(0.005,0.002,0.002),(0.02,0.005,0.005)]:
        for it in [50,100,200]:
            torch.cuda.synchronize(); t0=time.time()
            T,tot,li,lc,ld=fit_batch(FR,K,np.repeat(T0[None],12,0),[win]*12,[d]*12,iters=it,lr=lr)
            torch.cuda.synchronize()
            print(' ',lr,it, tot[0], 1-li[0], lc[0], 'time %.2f'%(time.time()-t0))
