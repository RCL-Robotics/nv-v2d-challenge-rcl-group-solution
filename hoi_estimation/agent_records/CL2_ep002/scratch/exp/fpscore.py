import sys, os
os.environ['FOUNDATIONPOSE_WEIGHTS_DIR']='/weights/foundationpose/nvlabs_pytorch'
sys.path.insert(0,'/workspace/v2d_foundation_pose/lib/FoundationPose')
sys.path.insert(0,'/work/outputs')
import logging
import numpy as np, cv2, h5py, io, torch
from PIL import Image
from estimater import ScorePredictor, make_mesh_tensors, compute_mesh_diameter
import nvdiffrast.torch as dr
from hoilib import load_inputs, load_mesh
logging.disable(logging.INFO)
K,Twc,rec,cam=load_inputs('/work/inputs')
g,V,F,UV,TEX=load_mesh('/work/inputs')
mc=(V.max(0)+V.min(0))/2
gm=g.copy(); gm.vertices=gm.vertices-mc; gm.visual=gm.visual.copy(); gm.visual.material=gm.visual.material.to_simple() if hasattr(gm.visual.material,"to_simple") else gm.visual.material
mt=make_mesh_tensors(gm); diam=compute_mesh_diameter(model_pts=gm.vertices,n_sample=10000)
scorer=ScorePredictor(); glctx=dr.RasterizeCudaContext()
D=h5py.File('/pipeline/depth/moge2/aligned.h5','r')['aligned/front_stereo_camera_left']
AXC=np.array([0.004,0,0.003])
def wyaw(R): x=R[:,0]; return np.degrees(np.arctan2(x[2],x[0]))
def yawrot(a):
    c,s=np.cos(a),np.sin(a); M=np.eye(4); M[:3,:3]=[[c,0,s],[0,1,0],[-s,0,c]]; M[:3,3]=AXC-M[:3,:3]@AXC; return M
Tw=np.load('/work/scratch/fitA.npz')['T_world_mesh']; Tcw=np.linalg.inv(Twc)
Kp=np.array(cam['K_pipeline'])
Tmc=np.eye(4); Tmc[:3,3]=mc
step=float(sys.argv[2]) if len(sys.argv)>2 else 15
for i in [int(x) for x in sys.argv[1].split(',')]:
    rgb=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg')[...,::-1].copy()
    dep=np.asarray(Image.open(io.BytesIO(D[i].tobytes()))).astype(np.float32)/1000.
    T=Tw[i]; poses=[]; ys=np.arange(0,360,step)
    for ay in ys:
        dl=np.radians(ay-wyaw(T[:3,:3])); Tn=T@yawrot(-dl)
        if abs(((wyaw(Tn[:3,:3])-ay+180)%360)-180)>4: Tn=T@yawrot(dl)
        poses.append(Tcw@Tn@Tmc)
    poses=np.array(poses)
    # rescale depth so depth at object matches pose depth
    om=h5py.File('/work/inputs/masks.h5','r')['object'][i]
    zobj=np.median(dep[om]) if om.sum()>20 else None
    zpose=poses[0][2,3]
    if zobj: dep=dep*(zpose/zobj)
    sc,_=scorer.predict(mesh=gm,rgb=rgb,depth=dep,K=K,ob_in_cams=poses,normal_map=None,mesh_tensors=mt,glctx=glctx,mesh_diameter=diam,get_vis=False)
    sc=sc.float().cpu().numpy()
    o=np.argsort(-sc)
    print(i, 'top', [(int(ys[k]),round(float(sc[k]),2)) for k in o[:6]], 'sc@290-300', [round(float(sc[k]),2) for k in range(len(ys)) if 270<=ys[k]<=315], 'sc@90-135',[round(float(sc[k]),2) for k in range(len(ys)) if 90<=ys[k]<=135])
