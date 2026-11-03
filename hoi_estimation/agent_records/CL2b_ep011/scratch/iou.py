import sys; sys.path.insert(0,'/work/outputs')
from hoi_lib import *
K,Twc,rec=load_inputs('/work/inputs')
cam=json.load(open('/work/inputs/camera.json')); Kp=np.array(cam['K_pipeline'])
R=MeshRenderer('/work/inputs/object_mesh.glb')
F=901
f=h5py.File('/work/inputs/masks.h5'); OM=f['object'][:]; PM=f['person'][:]
srcs={'base':(mesh_to_cam(Twc,rec['T_world_mesh'].astype(np.float64)),K),
      'fp':(rec['T_cam_mesh_foundationpose'].astype(np.float64),Kp),
      'cc':(rec['T_cam_mesh_coconet'].astype(np.float64),Kp)}
res={k:np.zeros(F) for k in srcs}
for i in range(F):
    ys,xs=np.nonzero(OM[i])
    if len(xs)==0:
        for k in res: res[k][i]=np.nan
        continue
    cx,cy=xs.mean(),ys.mean()
    crop=(int(cx)-128,int(cy)-96,256,192)
    x0,y0=crop[0],crop[1]
    om=np.zeros((192,256),bool); pm=np.zeros((192,256),bool)
    sl=(slice(max(y0,0),min(y0+192,1152)),slice(max(x0,0),min(x0+256,1536)))
    dsl=(slice(sl[0].start-y0,sl[0].stop-y0),slice(sl[1].start-x0,sl[1].stop-x0))
    om[dsl]=OM[i][sl]; pm[dsl]=PM[i][sl]
    valid=~(pm&~om)
    for k,(T,KK) in srcs.items():
        _,m,_=R.render(T[i:i+1],KK,crop,(256,192),with_tex=False)
        m=m[0].cpu().numpy()&valid
        res[k][i]=(m&om).sum()/max((m|om).sum(),1)
np.savez('iou_src.npz',**res)
for a in range(0,F,30):
    print(a, ' '.join('%s=%.2f'%(k,np.nanmean(v[a:a+30])) for k,v in res.items()))
print('all', ' '.join('%s=%.3f'%(k,np.nanmean(v)) for k,v in res.items()))
