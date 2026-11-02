import sys, numpy as np, json, h5py, cv2
sys.path.insert(0,'/work/outputs')
from render_utils import *; from edgescore import *
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam'])
m,tex=load_mesh('/work/inputs/object_mesh.glb'); R=Renderer(m,tex,K)
f=h5py.File('/work/inputs/masks.h5')
def scores(Tcam, frames):
    out=[]
    for j,i in enumerate(frames):
        _,_,dp=R.render(Tcam[j],need_rgb=False)
        out.append(edge_score(dp[0].cpu().numpy(), cv2.imread('frames/%04d.jpg'%(i+1)), f['person'][i]))
    return np.array(out)
if __name__=='__main__':
    P=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(np.float64); Tc=np.linalg.inv(Twc)[None]@P
    d=np.load('move_fit_0.npz'); fr=np.arange(201,560,3)
    sb=scores(Tc[fr],fr); ss=scores(d['Tsm'][fr-201],fr); sf=scores(d['Tf'][fr-201],fr)
    for k in range(0,len(fr),4): print(fr[k], '%.2f %.2f %.2f'%(sb[k],ss[k],sf[k]))
    print('mean edge chamfer px: base %.3f smooth %.3f fit %.3f'%(np.nanmean(sb),np.nanmean(ss),np.nanmean(sf)), 'fit better frac', np.nanmean(sf<sb))
