import sys; sys.path.insert(0,'/work/outputs')
from lib import *
import torch.nn.functional as Fn
m,v,f,uv,tex=load_mesh('/work/inputs/object_mesh.glb')
cam,K,Twc,rec=load_inputs('/work/inputs')
c0,u=body_axis(v)
Tc=np.linalg.inv(Twc)[None]@rec['T_world_mesh'].astype(np.float64)
D=np.load('crops160.npz'); S=160
R=Renderer(v,f,uv,tex)
angs=np.radians(np.arange(0,360,10))
frames=[int(x) for x in sys.argv[1].split(',')]
def iou_shift(rm, om, valid, maxs=6):
    # rm (N,S,S) float; om, valid (S,S) bool tensors; returns best iou over shifts
    best=torch.zeros(rm.shape[0],device=rm.device)-1
    for dy in range(-maxs,maxs+1,2):
        for dx in range(-maxs,maxs+1,2):
            r=torch.roll(rm,(dy,dx),(1,2))
            inter=(r*om*valid).sum((1,2)); uni=((r+om-r*om)*valid).sum((1,2))
            best=torch.maximum(best,inter/uni)
    return best
res=[]
for t in frames:
    Ts=spin(Tc[t],c0,u,angs)
    off=np.repeat(D['off'][t][None],len(Ts),0)
    _,mk,_=R.render(Ts,K,S,S,crop=off)
    om=torch.tensor(D['om'][t],device='cuda').float(); pm=torch.tensor(D['pm'][t],device='cuda').float()
    valid=1-pm*(1-om)
    sc=iou_shift(mk.float(),om,valid).cpu().numpy()
    res.append(sc)
    print(t, ' '.join('%2d'%(100*x) for x in sc), 'best', int(np.degrees(angs[sc.argmax()])))
