import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from hoilib import *
import solve
ep=Episode('/work/inputs')
d=np.load('run2_dbg.npz'); seq=list(d['seq']); vis=list(d['vis']); iouf=d['iouf']; tw_f=d['tw_f']; twA=float(d['twA'])
X=solve.HMM_STATES; yaw=solve.body_yaw(ep); dpred=-np.diff(yaw,prepend=yaw[0]); D=X[None,:]-X[:,None]
vk={t:k for k,t in enumerate(vis)}
for sig_iou in [0.01,0.02,0.03]:
  for sig_step in [4,6,10]:
    for prior in [1.0,0.5,0.0]:
      for twBm in [15,20]:
        n=len(seq); un=np.zeros((n,len(X)))
        for s,t in enumerate(seq):
            if t in vk:
                prof=solve.periodic_interp(X,tw_f[vk[t]],iouf[vk[t]]); un[s]=-0.5*((prof-prof.max())/sig_iou)**2
        un[0]+=-0.5*((X-twA)/3)**2; un[-1]+=-0.5*(solve.wrap90(X-twBm)/3)**2
        al=un[0]-un[0].max()
        for s in range(1,n):
            m=al[:,None]-0.5*((D-prior*dpred[seq[s]])/sig_step)**2; mx=m.max(0); al=mx+np.log(np.exp(m-mx).sum(0))+un[s]; al-=al.max()
        p=np.exp(al); p/=p.sum()
        out=[]
        for c in [twBm-180,twBm-90,twBm,twBm+90]:
            sel=np.abs(((X-c)+180)%360-180)<45; out.append('%4d:%.2f'%(c,p[sel].sum()))
        print('sig_iou %.2f step %2d prior %.1f B%2d | '%(sig_iou,sig_step,prior,twBm),' '.join(out))
