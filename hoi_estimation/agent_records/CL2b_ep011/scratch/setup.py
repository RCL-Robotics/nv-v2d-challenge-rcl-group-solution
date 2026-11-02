import sys; sys.path.insert(0,'/work/outputs')
from hoi_lib import *
K,Twc,rec=load_inputs('/work/inputs')
import time; t=time.time()
frames=load_video('/work/inputs/video.mp4')
f=h5py.File('/work/inputs/masks.h5'); OM=f['object'][:]; PM=f['person'][:]
fa=h5py.File('/pipeline/depth/moge2/aligned.h5')
ds=[cv2.imdecode(np.frombuffer(fa['aligned/front_stereo_camera_left'][i],np.uint8),cv2.IMREAD_UNCHANGED) for i in range(0,901,10)]
bg=np.median(np.stack(ds),0).astype(np.float32)/1000.
print('load',time.time()-t)
R=MeshRenderer('/work/inputs/object_mesh.glb')
obs=Obs(frames,OM,PM,bg,K)
fit=PoseFitter(R,obs,K)
Tb=mesh_to_cam(Twc,rec['T_world_mesh'].astype(np.float64))
