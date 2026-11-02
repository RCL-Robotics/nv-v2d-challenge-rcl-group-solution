import sys; sys.path.insert(0,'/work/outputs')
from common import *
K,Kp,Twc=load_camera('/work/inputs'); m=load_mesh('/work/inputs/object_mesh.glb'); r=Renderer(m,K)
rr=np.load('/work/inputs/reconstruction.npz'); Tc0=np.linalg.inv(Twc)@rr['T_world_mesh'].astype(np.float64)
cxy=np.array([0.005,-0.004,0]); Fz=np.eye(4); Fz[:3,:3]=np.diag([-1,-1,1.]); Fz[:3,3]=2*cxy
f=h5py.File('/work/inputs/masks.h5')
def stats(rgb,mask):
    R_,G,B=[rgb[...,k].astype(float)+1 for k in range(3)]
    sel=mask&(R_>1.5*G)&(R_>60)
    if sel.sum()<5: return (0,0,0)
    return sel.sum(), np.median(G[sel]/R_[sel]), np.median(B[sel]/R_[sel])
for i in [int(x) for x in sys.argv[1].split(',')]:
    img=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg')[:,:,::-1]
    om=f['object'][i]&~f['person'][i]
    s=[stats(img,om)]
    for T in [Tc0[i],Tc0[i]@Fz]:
        mk,rgb,_,_=r.render(T[None]); rgb=(rgb[0].cpu().numpy()*255); mk=mk[0].cpu().numpy()
        s.append(stats(rgb,mk&~f['person'][i]))
    print(i,'img n=%d G/R=%.2f B/R=%.2f'%s[0],'| base n=%d G/R=%.2f B/R=%.2f'%s[1],'| flip n=%d G/R=%.2f B/R=%.2f'%s[2])
