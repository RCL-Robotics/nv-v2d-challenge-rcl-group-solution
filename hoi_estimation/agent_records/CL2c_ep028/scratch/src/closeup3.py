import numpy as np,cv2,sys
from opt import *
from rend import Renderer
d=np.load(sys.argv[4]) if len(sys.argv)>4 else np.load('../cands.npz')
SS=4
Kh=K.copy(); Kh[:2]*=SS
Rn=Renderer(Kh,W=W*SS,H=H*SS)
rows=[]
half=int(sys.argv[5]) if len(sys.argv)>5 else 30
for i in [int(x) for x in sys.argv[2].split(',')]:
    hs=[int(x) for x in sys.argv[3].split(',')]
    im=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg')
    outs=[]
    for k,h in enumerate(hs):
        T=Rt_to_T(d['R'][i,h],d['t'][i,h])
        if k==0:
            p=T[:3,:3]@np.array([0,0,-0.47])+T[:3,3]; u=K@p; u=(u[:2]/u[2]).astype(int); x0,y0=u[0]-half,u[1]-half
            outs.append(cv2.resize(im[y0:y0+2*half,x0:x0+2*half],(360,360),interpolation=cv2.INTER_CUBIC))
        col,m,_=Rn.render(T,crop=(x0*SS,y0*SS,2*half*SS,2*half*SS))
        c=(col[0].cpu().numpy()[...,::-1]*255).astype(np.uint8); mm=m[0].cpu().numpy()
        bg=cv2.resize(im[y0:y0+2*half,x0:x0+2*half],(2*half*SS,2*half*SS),interpolation=cv2.INTER_CUBIC); bg[mm]=c[mm]
        o=cv2.resize(bg,(360,360),interpolation=cv2.INTER_AREA); cv2.putText(o,f'{i} h{h}',(5,20),0,0.6,(0,255,255),2); outs.append(o)
    rows.append(np.hstack(outs))
cv2.imwrite(sys.argv[1],np.vstack(rows))
