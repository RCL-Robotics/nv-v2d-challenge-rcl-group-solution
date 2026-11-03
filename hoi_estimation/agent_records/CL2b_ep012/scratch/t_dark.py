import sys, time; sys.path.insert(0,'/work/scratch')
from bfit import *
from fit import dark_prob
m,tex=load_mesh(); K,Twc,Kp=load_cam()
R=Renderer(m,tex,CROP,CROP); bf=BatchFitter(R)
d=np.load('crops.npz'); Ks=d['Ks']; tg=d['tg'].astype(np.float32); wt=d['wt'].astype(np.float32)
cap=cv2.VideoCapture(INP+'/video.mp4'); IM=[]
for i in range(700):
    ok,f=cap.read(); x0=int(K[0,2]-Ks[i][0,2]); y0=int(K[1,2]-Ks[i][1,2]); IM.append(crop_img(f,x0,y0))
IM=np.stack(IM); np.save('imcrops.npy',IM)
DP=np.stack([dark_prob(im) for im in IM]).astype(np.float32)
# ROI: dilated SAM mask, minus person
roi=np.stack([cv2.dilate(t.astype(np.uint8),np.ones((21,21))) for t in tg]).astype(np.float32)*wt
Tb=np.load('fit_free.npz')['T']; Ts=np.load('fit_sym.npz')['T']
out={}
for nm,T0 in [('b',Tb),('s',Ts)]:
    L0=[];L1=[];TT=[]
    for s in range(0,700,50):
        sl=slice(s,s+50)
        _,p0,_=bf.fit(T0[sl],Ks[sl],DP[sl],roi[sl],iters=0)
        T,p1,_=bf.fit(T0[sl],Ks[sl],DP[sl],roi[sl],iters=60,lr_r=0.003,lr_t=0.002)
        L0.append(p0);L1.append(p1);TT.append(T)
    out[nm]=(np.concatenate(L0),np.concatenate(L1),np.concatenate(TT))
np.savez('fit_dark.npz',Lb0=out['b'][0],Lb1=out['b'][1],Tb=out['b'][2],Ls0=out['s'][0],Ls1=out['s'][1],Ts=out['s'][2])
for s in range(0,700,25):
    sl=slice(s,s+25)
    print(s,'eval-only base %.4f sym %.4f fracB %.2f | refined base %.4f sym %.4f fracB %.2f'%(out['b'][0][sl].mean(),out['s'][0][sl].mean(),(out['b'][0][sl]<out['s'][0][sl]).mean(),out['b'][1][sl].mean(),out['s'][1][sl].mean(),(out['b'][1][sl]<out['s'][1][sl]).mean()))
