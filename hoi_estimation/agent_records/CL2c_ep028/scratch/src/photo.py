import numpy as np,cv2,torch
from opt import *
from rend import Renderer
Rn=None
def head_window(Ts):
    us=[]
    for T in Ts:
        for zz in [-0.50,-0.40]:
            for xx in [-0.22,0.22]:
                p=T[:3,:3]@np.array([xx,0,zz])+T[:3,3]; u=K@p; us.append(u[:2]/u[2])
    us=np.array(us); c=us.mean(0); half=max(24,0.6*np.ptp(us,0).max())
    return int(c[0]-half),int(c[1]-half),int(2*half)
def ncc(a,b,m):
    a=a[m].astype(np.float64); b=b[m].astype(np.float64)
    a=a-a.mean(0); b=b-b.mean(0)
    return float((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()+1e-9))
def photo_scores(i,Ts,im=None):
    global Rn
    if Rn is None: Rn=Renderer(K)
    if im is None: im=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg')
    x0,y0,s=head_window(Ts)
    x0=max(0,min(x0,W-s)); y0=max(0,min(y0,H-s))
    real=cv2.GaussianBlur(im[y0:y0+s,x0:x0+s],(0,0),1.0)
    col,m,_=Rn.render(np.stack(Ts),crop=(x0,y0,s,s))
    col=(col.cpu().numpy()[...,::-1]*255); m=m.cpu().numpy()
    out=[]
    anym=m.any(0)
    # evaluate on union of rendered masks of all hyps dilated -> same pixel set for all
    reg=cv2.dilate(anym.astype(np.uint8),np.ones((5,5)))>0
    for k in range(len(Ts)):
        comp=real.copy(); comp[m[k]]=col[k][m[k]]
        comp=cv2.GaussianBlur(comp.astype(np.uint8),(0,0),1.0)
        out.append(ncc(comp,real,reg))
    return np.array(out)
if __name__=='__main__':
    d=np.load('../cands.npz')
    for i in [5,20,40,60,80,100,200,300,350,400,450,505,515,530,600]:
        Ts=[Rt_to_T(d['R'][i,h],d['t'][i,h]) for h in (0,6)]
        print(i,photo_scores(i,Ts).round(3), d['L'][i,[0,6]].round(3))
