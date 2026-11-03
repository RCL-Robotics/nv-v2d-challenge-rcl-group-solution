import sys; sys.path.insert(0,'/work/outputs')
from evidence import *
import pickle, time
t0=time.time()
K,Kp,Twc=load_camera('/work/inputs'); m=load_mesh('/work/inputs/object_mesh.glb'); r=Renderer(m,K)
rr=np.load('/work/inputs/reconstruction.npz'); Tc0=np.linalg.inv(Twc)@rr['T_world_mesh'].astype(np.float64)
frames=read_video('/work/inputs/video.mp4')
ev=Evidence('/work/inputs',frames,r,K,Tc0,S=320)
ev.r=None
pickle.dump(ev,open('/work/scratch/ev320.pkl','wb'),protocol=4)
print(time.time()-t0)
