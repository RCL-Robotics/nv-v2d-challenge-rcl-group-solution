import numpy as np, json
from scipy.spatial.transform import Rotation as R
r=np.load('/work/inputs/reconstruction.npz')
def swing_twist(Rrel):
    q=R.from_matrix(Rrel).as_quat() # x y z w
    tw=np.array([0,0,q[2],q[3]]); tw/=np.linalg.norm(tw)
    ang=2*np.degrees(np.arctan2(tw[2],tw[3]))
    sw=R.from_quat(q)*R.from_quat(tw).inv()
    return (ang+180)%360-180, np.degrees(sw.magnitude())
for key in ['T_world_mesh','T_cam_mesh_foundationpose','T_cam_mesh_coconet']:
    T=r[key].astype(np.float64)
    print(key)
    s=[]
    for i in range(0,440,2):
        tw,sw=swing_twist(T[0,:3,:3].T@T[i,:3,:3])
        s.append('%d:%.0f/%.0f'%(i,tw,sw))
    print(' '.join(s))
