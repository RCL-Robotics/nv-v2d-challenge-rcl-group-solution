import numpy as np,cv2,sys
from opt import *
from core import orange_mask
d=np.load('../cands.npz')
rows=[]
for i in [int(x) for x in sys.argv[2].split(',')]:
    hs=[int(x) for x in sys.argv[3].split(',')]
    im=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg'); o=orange_mask(im)
    outs=[]
    for k,h in enumerate(hs):
        R,t=d['R'][i,h],d['t'][i,h]
        Xc=(R@(pts-c_mesh).T).T+t; uv=(K@Xc.T).T; uv=uv[:,:2]/uv[:,2:]
        nr=(R@nrm.T).T; face=(nr*(-Xc)).sum(1)>0
        if k==0: c=uv[orange].mean(0).astype(int); x0,y0=c[0]-40,c[1]-40
        vis=im.copy(); vis[o]=(0,255,0)
        for p,fc,orr in zip(uv,face,orange):
            if orr: cv2.circle(vis,(int(p[0]),int(p[1])),0,(255,0,255) if fc else (255,255,0),-1)
        v=cv2.resize(vis[y0:y0+80,x0:x0+80],(320,320),interpolation=cv2.INTER_NEAREST)
        cv2.putText(v,f'{i} h{h} L%.3f o2i %.2f i2o %.2f'%(d['L'][i,h],d['o2i'][i,h],d['i2o'][i,h]),(3,15),0,0.45,(0,255,255),1); outs.append(v)
    rows.append(np.hstack(outs))
cv2.imwrite(sys.argv[1],np.vstack(rows))
