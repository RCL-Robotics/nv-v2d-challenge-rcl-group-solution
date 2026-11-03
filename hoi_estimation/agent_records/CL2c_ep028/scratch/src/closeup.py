import numpy as np,cv2,sys
from opt import *
from rend import Renderer
Rn=Renderer(K)
def head_close(i,T,label,half=90,S=360):
    im=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg')
    col,m,_=Rn.render(T); m=m[0].cpu().numpy(); c=(col[0].cpu().numpy()[...,::-1]*255)
    hp=T[:3,:3]@np.array([0,0,-0.48])+T[:3,3]; u=K@hp; u=(u[:2]/u[2]).astype(int)
    x0=int(np.clip(u[0]-half,0,W-2*half)); y0=int(np.clip(u[1]-half,0,H-2*half))
    ov=im.copy(); ov[m]=(0.0*im[m]+1.0*c[m]).astype(np.uint8)
    a=cv2.resize(im[y0:y0+2*half,x0:x0+2*half],(S,S),interpolation=cv2.INTER_CUBIC); b=cv2.resize(ov[y0:y0+2*half,x0:x0+2*half],(S,S),interpolation=cv2.INTER_NEAREST)
    t=np.hstack([a,b]); cv2.putText(t,f'{i} {label}',(5,25),0,0.7,(0,255,255),2); return t
if __name__=='__main__':
    d=np.load('../cands.npz')
    rows=[]
    for i in [int(x) for x in sys.argv[2].split(',')]:
        hs=[int(x) for x in sys.argv[3].split(',')]
        rows.append(np.hstack([head_close(i,Rt_to_T(d['R'][i,h],d['t'][i,h]),f'h{h} L%.3f'%d['L'][i,h]) for h in hs]))
    cv2.imwrite(sys.argv[1],np.vstack(rows))
