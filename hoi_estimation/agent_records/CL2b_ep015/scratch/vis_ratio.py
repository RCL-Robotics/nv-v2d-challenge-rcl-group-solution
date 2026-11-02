import sys; sys.path.insert(0,'/work/outputs')
from common import *
K,Kp,Twc=load_camera('/work/inputs'); m=load_mesh('/work/inputs/object_mesh.glb'); r=Renderer(m,K)
rr=np.load('/work/inputs/reconstruction.npz'); Tc0=np.linalg.inv(Twc)@rr['T_world_mesh'].astype(np.float64)
f=h5py.File('/work/inputs/masks.h5')
out=[]
for i in range(715):
    mk=r.render(Tc0[i][None],want_tex=False)[0][0].cpu().numpy()
    om=f['object'][i]; pm=f['person'][i]
    out.append((om.sum(), mk.sum(), (mk&pm&~om).sum()))
out=np.array(out,float); np.save('vis.npy',out)
ratio=out[:,0]/np.maximum(out[:,1],1); occ=out[:,2]/np.maximum(out[:,1],1)
for i in range(160,640,5): print(i,'mask/render %.2f  render-occluded-by-person %.2f'%(ratio[i],occ[i]))
