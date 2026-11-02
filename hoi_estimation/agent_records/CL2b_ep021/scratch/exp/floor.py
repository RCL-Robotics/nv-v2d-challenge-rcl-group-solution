import sys; sys.path.insert(0,'/work/outputs')
from common import *
K,Kp,Twc,rec,gp=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs')
Tw=rec['T_world_mesh'].astype(np.float64)
v=np.asarray(mesh.vertices); pl=np.array(gp['plane'])
print('mesh z range', v[:,2].min(), v[:,2].max())
d=[]
for i in range(len(Tw)):
    vw=v@Tw[i,:3,:3].T+Tw[i,:3,3]; s=vw@pl[:3]+pl[3]
    d.append((s.min(),s.max()))
d=np.array(d)
for i in range(0,678,20): print(i, np.round(d[i],3))
# camera center distance to plane
c=Twc[:3,3]; print('cam height', c@pl[:3]+pl[3])
