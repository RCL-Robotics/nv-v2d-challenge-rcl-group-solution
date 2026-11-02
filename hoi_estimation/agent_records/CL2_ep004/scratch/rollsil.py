import sys; sys.path.insert(0,'/work/scratch')
from common import *
from bat_lib import Renderer, W_IMG, H_IMG
from scipy.spatial.transform import Rotation as Ro
import h5py
ep=Episode('/work/inputs'); Rf=Renderer(ep.mesh); f=h5py.File('/work/inputs/masks.h5')
T=ep.Tcw[None]@np.load('/work/scratch/run2/object_poses_world.npz')['T_world_mesh']
for frames in [list(range(0,110,22)),list(range(710,802,18))]:
    i0=frames[0]; angs=np.arange(0,360,15); Ts=[]
    for a in angs:
        Tr=T[i0].copy(); Rr=T[i0,:3,:3]@Ro.from_euler('z',a,degrees=True).as_matrix()
        c=T[i0,:3,:3]@BAT_CENTER+T[i0,:3,3]; Tr[:3,:3]=Rr; Tr[:3,3]=c-Rr@BAT_CENTER; Ts.append(Tr)
    Ts=np.stack(Ts); tot=np.zeros(len(angs)); totb=0
    for i in frames:
        m,_=Rf.render(Ts,ep.K,W_IMG,H_IMG); m=m.cpu().numpy()>0; b=f['object'][i]
        tot+=np.array([(mk&b).sum()/(mk|b).sum() for mk in m])
        mb,_=Rf.render(ep.Tc_base[[i]],ep.K,W_IMG,H_IMG); mb=mb.cpu().numpy()[0]>0; totb+=(mb&b).sum()/(mb|b).sum()
    print(frames[0], 'baseline IoU %.3f'%(totb/len(frames)), ' '.join('%d:%.3f'%(a,v/len(frames)) for a,v in zip(angs,tot)))
