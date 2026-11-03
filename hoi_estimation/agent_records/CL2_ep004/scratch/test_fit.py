import sys, time; sys.path.insert(0,'/work/outputs')
import numpy as np, torch, cv2, h5py, json
from bat_lib import *; from fitter import *
K,Twc,rec,plane=load_inputs('/work/inputs'); Kp=np.array(json.load(open('/work/inputs/camera.json'))['K_pipeline'])
s=np.load('/work/scratch/simp.npz'); FR=FastRenderer(s['vs'],s['fs'],s['cs'])
Tcb=np.linalg.inv(Twc)[None]@rec['T_world_mesh'].astype(np.float64)
fp=rec['T_cam_mesh_foundationpose'].astype(np.float64); cc=rec['T_cam_mesh_coconet'].astype(np.float64)
f=h5py.File('/work/inputs/masks.h5'); cap=cv2.VideoCapture('/work/inputs/video.mp4')
frames=[int(x) for x in sys.argv[1:]]
res={}
t0=time.time()
for i in frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,img=cap.read()
    om=f['object'][i]; pm=f['person'][i]
    hint=(K@Tcb[i,:3,3]); hint=hint[:2]/hint[2]
    win=crop_window(om,hint,0.54*K[0,0]/Tcb[i,2,3]); d=prep_frame(img,om,pm,win)
    inits=[]; names=[]
    for nm,T in [('base',Tcb[i]),('fp',convert_K(fp[i],Kp,K,0.989)),('cc',convert_K(cc[i],Kp,K,1.029))]:
        for k,v in enumerate(variants(T)): inits.append(v); names.append(f'{nm}{k}')
    inits=np.stack(inits)
    s0=score_batch(FR,K,inits,[win]*len(inits),[d]*len(inits))
    T,tot,li,lc,ld=fit_batch(FR,K,inits,[win]*len(inits),[d]*len(inits))
    o=np.argsort(tot)
    print(i,' '.join(f'{names[j]}:{tot[j]:.3f}(iou{1-li[j]:.2f},c{lc[j]:.3f},init{s0[0][j]:.2f})' for j in o[:6]))
    res[i]=T[o[0]]
print('time',time.time()-t0)
Tw=np.tile(np.eye(4),(803,1,1)); Tw[:]=rec['T_world_mesh']
for i in frames: Tw[i]=Twc@res[i]
np.savez('/work/scratch/test_fit.npz',T=Tw)
