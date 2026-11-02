import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2
from hoi_lib import *
d=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs'); R=Renderer(mesh)
cap=cv2.VideoCapture('/work/inputs/video.mp4')
def getframe(i):
    cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,f=cap.read(); return f[...,::-1]
def sbs(T_wm_list, idx, out):
    rows=[]
    for i in idx:
        im=getframe(i).astype(np.float32)/255; tiles=[]
        rs=[]
        for T_wm in T_wm_list:
            Tc=cam_from_world(d['T_wc'],T_wm[i:i+1]); r=R.render(Tc,d['K'],d['H'],d['W'])
            rs.append((r['rgb'][0].cpu().numpy(),r['mask'][0].cpu().numpy()))
        m=rs[0][1]; ys,xs=np.where(m); y0,y1,x0,x1=max(ys.min()-40,0),ys.max()+40,max(xs.min()-40,0),xs.max()+40
        def prep(a): 
            a=a[y0:y1,x0:x1]; a=np.clip(a/ max(np.percentile(a[a>0],95),1e-3) ,0,1) if a.max()>0 else a
            return cv2.resize((a**0.7*255).astype(np.uint8),(320,320))
        t=prep(im.copy()); cv2.putText(t,str(i),(5,20),0,0.7,(255,255,0),2); tiles.append(t)
        for rgb,mm in rs: tiles.append(prep(rgb))
        rows.append(np.hstack(tiles))
    cv2.imwrite(out,np.vstack(rows)[...,::-1])
if __name__=='__main__':
    T=d['rec']['T_world_mesh'].astype(np.float64)
    idx=[int(x) for x in sys.argv[2].split(',')]
    sbs([T],idx,sys.argv[1])
