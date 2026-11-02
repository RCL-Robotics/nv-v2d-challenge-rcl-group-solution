import sys; sys.path.insert(0,'/work/outputs'); sys.path.insert(0,'/work/scratch/exp')
from common import *
from rectify import *
D=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs'); K=D['K']
Tw=np.load('/work/scratch/opt1.npz')['T_world_mesh']; Tc=np.linalg.inv(D['Twc'])@Tw
R=Renderer(mesh,K,1152,1536)
ax_p=np.array([0.006,0.004,0.0])
def spinT(T,deg):
    S=np.eye(4); S[:3,:3]=axis_angle([0,0,1],np.radians(deg)); S[:3,3]=ax_p-S[:3,:3]@ax_p; return T@S
f=560; P=plane_grid(-0.092)
o=R.render(spinT(Tc[f],40)[None],depth=True)
img=o['rgb'][0].cpu().numpy(); dep=o['depth'][0].cpu().numpy()
rr,w=rectify(img,Tc[f],K,P,depth=dep,valid_mask=o['mask'][0].cpu().numpy().astype(np.float32),normal_sign=1)
np.save('/work/scratch/rect_conv.npy',np.concatenate([rr,w[...,None]],-1))
