import os, sys, numpy as np, json, h5py, torch, trimesh, logging
os.environ['FOUNDATIONPOSE_WEIGHTS_DIR']='/weights/foundationpose/nvlabs_pytorch'
sys.path.insert(0,'/workspace/v2d_cari4d/lib/cari4d')
logging.disable(logging.INFO)
from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
from prep.mhr_depth_h5 import read_metric_depth
from Utils import depth2xyzmap, erode_depth, bilateral_filter_depth
import nvdiffrast.torch as dr
fr=np.load('/work/scratch/frames_all.npy',mmap_mode='r')
cam=json.load(open('/work/inputs/camera.json')); Kp=np.array(cam['K_pipeline'])
r=np.load('/work/inputs/reconstruction.npz'); Tcc=r['T_cam_mesh_coconet'].astype(np.float64)
H5=h5py.File('/work/inputs/masks.h5')
m=trimesh.load('/work/inputs/object_mesh.glb', force='scene', process=False).to_geometry()
m.visual=m.visual.to_color()
est=FoundationPose(model_pts=np.asarray(m.vertices),model_normals=np.asarray(m.vertex_normals),mesh=m,scorer=ScorePredictor(),refiner=PoseRefinePredictor(),debug_dir='/work/scratch/fp',debug=0,glctx=dr.RasterizeCudaContext())
tfc=est.get_tf_to_centered_mesh().cpu().numpy().astype(np.float64)
FY=np.diag([-1,1,-1,1.])
def ang(A,B): return np.degrees(np.arccos(np.clip((np.trace(A[:3,:3].T@B[:3,:3])-1)/2,-1,1)))
frames=[int(x) for x in sys.argv[1].split(',')]
for i in frames:
    rgb=np.ascontiguousarray(fr[i][...,::-1]); depth=read_metric_depth('/pipeline/depth/moge2/aligned.h5','aligned','front_stereo_camera_left','%06d'%i).astype(np.float32)
    depth[(depth<0.001)|(depth>8)]=0
    mo=H5['object'][i].astype(bool)
    best=est.register(K=Kp,rgb=rgb,depth=depth,ob_mask=mo,iteration=5)
    poses=est.poses.cpu().numpy()@tfc; scores=est.scores.cpu().numpy()
    a0=np.array([ang(P,Tcc[i]) for P in poses]); a1=np.array([ang(P,Tcc[i]@FY) for P in poses])
    cls=np.where(a0<a1,0,1); near=np.minimum(a0,a1)<45
    sb=scores[(cls==0)&near].max() if ((cls==0)&near).any() else np.nan
    sf=scores[(cls==1)&near].max() if ((cls==1)&near).any() else np.nan
    # direct comparison: refine base & flip then score
    dd=bilateral_filter_depth(erode_depth(depth,radius=2,device='cuda'),radius=2,device='cuda'); xyz=depth2xyzmap(dd,Kp)
    pc=np.stack([Tcc[i],Tcc[i]@FY])@np.linalg.inv(tfc)
    ref,_=est.refiner.predict(mesh=est.mesh,mesh_tensors=est.mesh_tensors,rgb=rgb,depth=dd,K=Kp,ob_in_cams=pc.astype(np.float32),normal_map=None,xyz_map=xyz,glctx=est.glctx,mesh_diameter=est.diameter,iteration=5,get_vis=False)
    ref=ref.cpu().numpy() if torch.is_tensor(ref) else ref
    sc,_=est.scorer.predict(mesh=est.mesh,rgb=rgb,depth=dd,K=Kp,ob_in_cams=ref,normal_map=None,mesh_tensors=est.mesh_tensors,glctx=est.glctx,mesh_diameter=est.diameter)
    sc=sc.cpu().numpy()
    print(i,'top1: d_base %.0f d_flip %.0f score %.2f | best score base-class %.2f flip-class %.2f | top20 %s | pair score base %.2f flip %.2f'%(a0[0],a1[0],scores[0],sb,sf,''.join(str(c) for c in cls[:20]),sc[0],sc[1]),flush=True)
