import numpy as np,cv2,sys
from opt import *
from rend import Renderer
Rn=Renderer(K)
d=np.load('../cands.npz')
def crop(i,T,half=40,S=320):
    im=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg')
    col,m,_=Rn.render(T); m=m[0].cpu().numpy(); c=(col[0].cpu().numpy()[...,::-1]*255).astype(np.uint8); c[~m]=im[~m]
    hp=T[:3,:3]@np.array([0,0,-0.46])+T[:3,3]; u=K@hp; u=(u[:2]/u[2]).astype(int)
    return im,c,u
rows=[]
for i in [int(x) for x in sys.argv[2].split(',')]:
    hs=[int(x) for x in sys.argv[3].split(',')]
    outs=[]
    for k,h in enumerate(hs):
        im,c,u=crop(i,Rt_to_T(d['R'][i,h],d['t'][i,h]))
        if k==0: u0=u
        half=int(sys.argv[4]) if len(sys.argv)>4 else 40
        x0,y0=u0[0]-half,u0[1]-half
        if k==0: outs.append(cv2.resize(im[y0:y0+2*half,x0:x0+2*half],(320,320),interpolation=cv2.INTER_CUBIC))
        o=cv2.resize(c[y0:y0+2*half,x0:x0+2*half],(320,320),interpolation=cv2.INTER_CUBIC); cv2.putText(o,f'{i} h{h}',(5,20),0,0.6,(0,255,255),2); outs.append(o)
    rows.append(np.hstack(outs))
cv2.imwrite(sys.argv[1],np.vstack(rows))
