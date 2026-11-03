import sys; sys.path.insert(0,'/work/scratch/src')
from fitlib import *
import torch.nn.functional as Fn
cam,K,Twc,rec=load_inputs(); SM=SilMesh(Mesh().tm)
c=np.load('an/cropmasks.npz'); om,pm=c['om'],c['pm']; valid=~(pm&~om)
Tc=np.linalg.inv(Twc)@rec['T_world_mesh'].astype(float)
PH={'S1':(0,135),'lift':(135,230),'pour':(230,500),'down':(500,596),'S2':(596,748)}
def ious(T):
    out=[]
    for c0 in range(0,len(T),128):
        sil,_=render_soft(SM,torch.tensor(T[c0:c0+128],device=dev,dtype=torch.float32),K,X0,Y0,W,H)
        sil=Fn.max_pool2d(sil[:,None],5,1,2)[:,0]
        out.append(soft_iou((sil>0.5).float(),torch.tensor(om[c0:c0+128],device=dev).float(),torch.tensor(valid[c0:c0+128],device=dev).float()).cpu().numpy())
    return np.concatenate(out)
def load(f):
    if f=='base': return Tc
    d=np.load(f); return d['T'] if 'T' in d else np.linalg.inv(Twc)@d['T_world_mesh']
if __name__=='__main__':
    for f in sys.argv[1:]:
        I=ious(load(f)); print('%-28s all %.4f '%(f,I.mean())+' '.join('%s %.4f'%(k,I[a:b].mean()) for k,(a,b) in PH.items()), ' odd %.4f'%I[1::2].mean())
