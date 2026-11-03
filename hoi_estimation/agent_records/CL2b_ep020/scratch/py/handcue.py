from common import *
from scipy.spatial import cKDTree
kp=r['human_keypoints_world'].astype(np.float64); cp=r['contact_prob_right_left']
vs=v[::20].astype(np.float64); tree=cKDTree(vs)
J=np.load('/work/scratch/joint_free.npz'); Tours=J['T']
def hand_dist(Tc, i, dz=0.0):
    # Tc camera-frame pose; shift along ray by dz (scale)
    T=Tc.copy(); s=(T[2,3]+dz)/T[2,3]; T[:3,3]*=s
    Tw=Twc@T; Tinv=np.linalg.inv(Tw)
    pts=np.r_[kp[i,21:42],kp[i,42:63]]
    pm=(Tinv[:3,:3]@pts.T).T+Tinv[:3,3]
    d,_=tree.query(pm); return d.min()
dzs=np.arange(-0.15,0.1501,0.005)
for i in list(range(130,560,10)):
    if i>552: break
    db=[hand_dist(Tcm[i],i,dz) for dz in dzs]
    k=int(np.argmin(db))
    print(i,'contact %.2f %.2f'%tuple(cp[i]),'hand-surface dist base %.3f, best dz %.3f (dist %.3f)  => z %.3f ; base z %.3f'%(db[30],dzs[k],db[k],Tcm[i,2,3]+dzs[k],Tcm[i,2,3]))
