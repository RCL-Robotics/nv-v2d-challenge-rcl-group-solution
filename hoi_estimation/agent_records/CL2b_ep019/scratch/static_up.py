import sys; sys.path.insert(0,'/work/outputs')
from common import *; from fit import *; from photo import *
import h5py
K,Twc,_=load_camera(); Tcw=np.linalg.inv(Twc)
n=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']); nc=Tcw[:3,:3]@n[:3]; up=nc if nc[1]<0 else -nc
P=np.load('cache/P1.npy'); f=h5py.File('/work/inputs/masks.h5'); vid=np.load('video.npy',mmap_mode='r')
R=Renderer(); pts=sample_points(load_mesh()[4],1200)
def align(T,up):
    a=T[:3,2]; v=np.cross(a,up); s=np.linalg.norm(v); ang=np.arctan2(s,a@up)
    Ra=rotvec_to_R(v/s*ang) if s>1e-9 else np.eye(3)
    T2=T.copy(); c=T[:3,:3]@CENTER+T[:3,3]; T2[:3,:3]=Ra@T[:3,:3]; T2[:3,3]=c-T2[:3,:3]@CENTER; return T2
for a,b,ref in [(8,112,60),(768,885,860)]:
    fr=[i for i in np.linspace(a,b,16).round().astype(int) if f['object'][i].sum()>200]
    data={i:FrameData(f['object'][i],f['person'][i],n_pix=800) for i in fr}
    Tu=P[ref]; Ta=align(Tu,up)
    res=[]
    for nm,T0,kw in [('free',Tu,{}),('upright',Ta,dict(rot_axis=[up]*len(fr)))]:
        Tg,L=fit_poses(K,np.array([T0]*len(fr)),fr,data,pts,iters=150,share=[0]*len(fr),**kw)
        ph=np.mean([photo_scores(R,K,vid[i],*[f[k][i] for k in ['object','person']],Tg)[0] for i in fr])
        tilt=np.degrees(np.arccos(Tg[0,:3,2]@up))
        print(a,b,nm,'sil %.3f photo %.4f tilt %.1f z %.3f'%(L['loss'].mean(),ph,tilt,Tg[0,2,3]))
        res.append(Tg[0])
    np.save(f'static_up_{a}.npy',np.array(res))
