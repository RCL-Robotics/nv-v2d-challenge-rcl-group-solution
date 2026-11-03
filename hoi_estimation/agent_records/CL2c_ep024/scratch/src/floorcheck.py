import sys; sys.path.insert(0,'/work/scratch/src')
from fitlib import *
cam,K,Twc,rec=load_inputs()
Tc=np.linalg.inv(Twc)@rec['T_world_mesh'].astype(float)
pl=json.load(open('/work/inputs/ground_plane.json'))['plane']; n_w=np.array(pl[:3])
n_c=Twc[:3,:3].T@n_w
def tilt(T): return np.degrees(np.arccos(abs(T[:3,:3][:,0]@n_c)))
def yaw_dir(T):  # spout axis direction projected onto floor
    z=T[:3,:3][:,2]; return z
print('baseline mean S1 tilt %.2f  S2 %.2f'%(tilt(mean_pose(Tc[:134])),tilt(mean_pose(Tc[598:]))))
print('baseline per-frame tilt S1 median %.2f S2 %.2f'%(np.median([tilt(T) for T in Tc[:134]]),np.median([tilt(T) for T in Tc[598:]])))
for f in sys.argv[1:]:
    if f.endswith('.npz'):
        T=np.load(f)['T']; print(f,'S1 %.2f S2 %.2f'%(tilt(T[50]),tilt(T[700])))
    else: print(f,'%.2f'%tilt(np.load(f)))
