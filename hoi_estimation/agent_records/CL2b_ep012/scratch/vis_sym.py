import sys; sys.path.insert(0,'/work/scratch')
from bfit import *
m,tex=load_mesh(); K,Twc,Kp=load_cam()
R=Renderer(m,tex,CROP,CROP)
d=np.load('crops.npz'); Ks=d['Ks']; tg=d['tg']; wt=d['wt']
Tb=np.load('fit_free.npz')['T']; Ts=np.load('fit_sym.npz')['T']
cap=cv2.VideoCapture(INP+'/video.mp4'); ids=[int(a) for a in sys.argv[1:]]; F={}
for i in range(max(ids)+1):
    ok,f=cap.read()
    if i in ids: F[i]=f
rows=[]
for i in ids:
    a=R.render(Ks[i],np.stack([Tb[i],Ts[i]]),rgb=False)['mask'].cpu().numpy()
    x0=int(K[0,2]-Ks[i][0,2]); y0=int(K[1,2]-Ks[i][1,2])
    img=crop_img(F[i],x0,y0)
    def col(r):
        v=np.zeros((CROP,CROP,3),np.uint8); v[...,1]=tg[i]*255; v[...,2]=r*255; v[~wt[i]]=(80,0,0); return v
    rows.append(np.hstack([img,col(a[0]),col(a[1])]))
cv2.imwrite('vis_sym.png',np.vstack(rows))
