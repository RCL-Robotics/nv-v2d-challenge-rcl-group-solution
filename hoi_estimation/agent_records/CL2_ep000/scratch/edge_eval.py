import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2
from hoilib import *
def edge_dt(img, roi_mask=None):
    g=cv2.GaussianBlur(cv2.cvtColor(img,cv2.COLOR_RGB2GRAY),(5,5),1.2)
    e=cv2.Canny(g,30,80)
    dt=cv2.distanceTransform((e==0).astype(np.uint8),cv2.DIST_L2,5)
    return e,dt
def render_edges(rd,T):
    m,z,_=rd.render(T[None],need_rgb=False); m=m[0].cpu().numpy().astype(np.uint8); z=z[0].cpu().numpy()
    c=cv2.morphologyEx(m,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0
    # internal depth discontinuities
    zz=np.where(m>0,z,100.)
    gx=np.abs(np.diff(zz,axis=1,prepend=zz[:,:1])); gy=np.abs(np.diff(zz,axis=0,prepend=zz[:1]))
    c|=((gx>0.03)|(gy>0.03))&(m>0)
    return c,m
def chamfer(rd,T,dt,e,valid,trunc=10):
    c,m=render_edges(rd,T)
    c&=valid
    d1=np.minimum(dt[c],trunc).mean()
    # reverse: image edges near object -> rendered contour
    dtr=cv2.distanceTransform((~c).astype(np.uint8),cv2.DIST_L2,5)
    near=cv2.dilate(m,np.ones((31,31),np.uint8))>0
    ee=(e>0)&near&valid
    d2=np.minimum(dtr[ee],trunc).mean()
    return d1,d2
