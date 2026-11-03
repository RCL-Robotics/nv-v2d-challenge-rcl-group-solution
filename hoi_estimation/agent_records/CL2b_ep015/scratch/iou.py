import sys; sys.path.insert(0,'/work/outputs')
from common import *
K,Kp,Twc=load_camera('/work/inputs')
m=load_mesh('/work/inputs/object_mesh.glb')
r=Renderer(m,K)
f=h5py.File('/work/inputs/masks.h5')
def iou_all(Tw, name):
    Tc=np.linalg.inv(Twc)@Tw
    out=[]
    for i in range(len(Tw)):
        om=torch.from_numpy(f['object'][i]).cuda(); pm=torch.from_numpy(f['person'][i]).cuda()
        mk=r.render(Tc[i][None],want_tex=False)[0][0]
        vis=mk&~pm
        inter=(vis&om).sum().item(); uni=(vis|om).sum().item()
        out.append((inter/max(uni,1), mk.sum().item(), vis.sum().item(), om.sum().item()))
    return np.array(out)
if __name__=='__main__':
    rr=np.load('/work/inputs/reconstruction.npz')
    a=iou_all(rr['T_world_mesh'].astype(np.float64),'base')
    np.save('iou_base.npy',a)
    for i in range(0,715,5): print(i, np.round(a[i],3))
