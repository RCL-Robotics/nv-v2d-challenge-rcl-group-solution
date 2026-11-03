import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2, torch
from hoilib import *
ep=Episode('/work/inputs'); rd=Renderer(ep.mesh)
fl=list(range(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3])))
offs=np.arange(0,360,15)
S=192
def sat(x):
    mx=x.max(-1)[0]; mn=x.min(-1)[0]; return (mx-mn)/(mx+1e-3)
tot=np.zeros(len(offs)); cnt=0
rows=[]
for i in fl:
    T0=ep.Tc_base[i]
    if ep.obj_mask[i].sum()<500: continue
    ctr=proj(ep.K,(T0@np.array([0,0,0.09,1]))[None,:3])[0]; crop=crop_box(ctr,int(0.25*ep.K[0,0]/T0[2,3]))
    Ts=np.stack([axis_rot(T0,a) for a in offs])
    m,rgb=rd.render(Ts,ep.K,crop,S,color=True)
    im=torch.tensor(cut(ep.frames()[i],crop,S,cv2.INTER_AREA)/255.,device='cuda').float()
    om=torch.tensor(cut(ep.obj_mask[i].astype(np.uint8)*255,crop,S)/255.,device='cuda').float()
    pm=torch.tensor(cut(ep.person_mask[i].astype(np.uint8)*255,crop,S)/255.,device='cuda').float()
    w=(m>0.99).float()*(om>0.99).float()*(pm<0.01).float()
    # erode
    w=-torch.nn.functional.max_pool2d(-w[:,None],5,1,2)[:,0]
    si=sat(im)[None]; sr=sat(rgb)
    # compare also lightness-normalised colour
    d=((si-sr).abs()*w).sum((1,2))/w.sum((1,2)).clamp_min(1)
    d=d.cpu().numpy(); tot+=d; cnt+=1
    rows.append((i,d))
for i,d in rows[::max(1,len(rows)//12)]:
    print(i,' '.join('%d:%.3f'%(o,v) for o,v in zip(offs,d)))
print('MEAN',' '.join('%d:%.4f'%(o,v) for o,v in zip(offs,tot/cnt)))
