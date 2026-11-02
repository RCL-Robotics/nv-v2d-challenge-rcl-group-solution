# full-res silhouette comparison for spin hypotheses: green=real mask only, red=render only, white=both
import sys; sys.path.insert(0, '/work/outputs')
from common import *
import h5py
poses=sys.argv[1]; f=int(sys.argv[2]); spins=[float(x) for x in sys.argv[3].split(',')]; out=sys.argv[4]
D = load_inputs('/work/inputs'); mesh = load_mesh('/work/inputs')
Tw = D['rec']['T_world_mesh'].astype(np.float64) if poses == 'base' else np.load(poses)['T_world_mesh']
Tc = np.linalg.inv(D['Twc']) @ Tw
R = Renderer(mesh, D['K'], 1152, 1536)
ax_p = np.array([0.006, 0.004, 0.0])
def spinT(T, deg):
    S = np.eye(4); S[:3, :3] = axis_angle([0, 0, 1], np.radians(deg)); S[:3, 3] = ax_p - S[:3, :3] @ ax_p
    return T @ S
h=h5py.File('/work/inputs/masks.h5'); m=h['object'][f]; p=h['person'][f]
o=R.render(np.stack([spinT(Tc[f],s) for s in spins]),rgb=False)['mask'].cpu().numpy()
ys,xs=np.nonzero(m|o.any(0)); y0,y1,x0,x1=ys.min()-10,ys.max()+10,xs.min()-10,xs.max()+10
tiles=[]
for i,s in enumerate(spins):
    a=np.zeros(m.shape+(3,),np.uint8); a[m&~o[i]]=(0,255,0); a[o[i]&~m]=(0,0,255); a[o[i]&m]=(255,255,255); a[p&~m]=(a[p&~m]*0.3+(80,0,0)).astype(np.uint8)
    a=a[y0:y1,x0:x1].copy(); cv2.putText(a,str(s),(5,25),0,0.8,(0,255,255),2); tiles.append(a)
cv2.imwrite(out,np.concatenate(tiles,1))
