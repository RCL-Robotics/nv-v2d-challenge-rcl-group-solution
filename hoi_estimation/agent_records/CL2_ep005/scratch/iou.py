import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, torch
from hoi_lib import *
d=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs'); R=Renderer(mesh)
om=torch.tensor(np.load('objmask_half.npy')).cuda(); pm=torch.tensor(np.load('permask_half.npy')).cuda()
def ious(T_wm, scale=0.5):
    Tc=cam_from_world(d['T_wc'],T_wm); out=[]
    for s in range(0,len(Tc),32):
        r=R.render(Tc[s:s+32],d['K'],d['H'],d['W'],scale=scale,need_rgb=False)['mask']
        o=om[s:s+32]; valid=~(pm[s:s+32]&~o)
        inter=(r&o&valid).sum((1,2)).float(); uni=((r|o)&valid).sum((1,2)).float()
        out.append((inter/uni).cpu().numpy())
    return np.concatenate(out)
if __name__=='__main__':
    T=d['rec']['T_world_mesh'].astype(np.float64)
    i=ious(T); np.save('iou_base.npy',i)
    print('mean',i.mean()); 
    for s in range(0,1003,25): print(s, i[s:s+25].mean().round(3), i[s:s+25].min().round(3), end=' | ')
