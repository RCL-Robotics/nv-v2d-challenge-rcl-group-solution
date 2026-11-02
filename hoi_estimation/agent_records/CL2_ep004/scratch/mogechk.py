import sys, pickle; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2, torch
from stages import *; from fitter import *
K,Twc,rec,plane=load_inputs('/work/inputs')
A=pickle.load(open('/work/scratch/stageA.pkl','rb')); wins=A['wins']
datas=pickle.load(open('/work/scratch/datas.pkl','rb'))
raw=np.load('/pipeline/depth/moge2/raw.npy',mmap_mode='r')
s=np.load('/work/scratch/simp.npz'); FR=FastRenderer(s['vs'],s['fs'],s['cs'])
dec=np.load('/work/scratch/dec1.npz')
def moge_crop(i,win):
    x0,y0,side=win; sc=RES/side
    Aff=np.array([[sc,0,-x0*sc],[0,sc,-y0*sc]])
    return cv2.warpAffine(raw[i].astype(np.float32)/1000.,Aff,(RES,RES),flags=cv2.INTER_NEAREST)
def dres(T,i):
    Zm=moge_crop(i,wins[i]); M=datas[i]['M']>0.5
    Me=cv2.erode(M.astype(np.uint8),np.ones((3,3),np.uint8))>0
    with torch.no_grad():
        S,C,Z=FR.render(torch.tensor(T[None,:3,:3],dtype=torch.float32,device='cuda'),torch.tensor(T[None,:3,3],dtype=torch.float32,device='cuda'),crop_K(K,wins[i])[None],RES)
    S=S[0].cpu().numpy(); Z=Z[0].cpu().numpy()
    sel=(S>0.99)&Me&(Zm>0)
    if sel.sum()<15: return np.nan, sel.sum()
    r=Zm[sel]-Z[sel]; r=r-np.median(r)
    return np.median(np.abs(r)), sel.sum()
for i in range(115,700,5):
    T=dec['Tc'][i]; Tm=mirror_tilt(T)
    a,n=dres(T,i); b,_=dres(Tm,i)
    ax=T[:3,2]; r=T[:3,3]/np.linalg.norm(T[:3,3]); tilt=np.degrees(np.arcsin(ax@r))
    print(i,'n %d chosen %.3f mirror %.3f tilt %.0f'%(n,a,b,tilt), '<<' if b<a else '')
