import sys; sys.path.insert(0,'/work/scratch/src')
from common import *
from scipy.spatial.transform import Rotation as Rot
X0,Y0,W,H=600,700,320,240

def greenmask_img(im, sam, sth=125):
    hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    g=(hsv[...,0]>25)&(hsv[...,0]<60)&(hsv[...,1]>sth)&(hsv[...,2]>40)
    sd=cv2.dilate(sam.astype(np.uint8),np.ones((9,9),np.uint8))>0
    return g&sd

def mean_pose(Ts):
    Rm=Rot.from_matrix(Ts[:,:3,:3]).mean().as_matrix()
    T=np.eye(4); T[:3,:3]=Rm; T[:3,3]=np.median(Ts[:,:3,3],0); return T

def pose_from(T0,w,t):
    Rd=so3_exp(w+1e-9)
    R=Rd@T0[...,:3,:3]; tt=T0[...,:3,3]+t
    top=torch.cat([R,tt[...,None]],-1)
    bot=torch.zeros(top.shape[:-2]+(1,4),device=dev); bot[...,0,3]=1
    return torch.cat([top,bot],-2)

def soft_iou(sil,tg,valid):
    inter=(sil*tg*valid).sum((-1,-2)); uni=((sil+tg-sil*tg)*valid).sum((-1,-2))
    return inter/uni.clamp(min=1)

class SilMesh:
    def __init__(self, tm, nfaces=20000):
        import open3d as o3d
        om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(tm.vertices), o3d.utility.Vector3iVector(tm.faces))
        om = om.merge_close_vertices(1e-6).simplify_quadric_decimation(nfaces)
        self.v = torch.tensor(np.asarray(om.vertices), dtype=torch.float32, device=dev)
        self.f = torch.tensor(np.asarray(om.triangles), dtype=torch.int32, device=dev)
        self.glctx = dr.RasterizeCudaContext()

def make_targets(om, pm, sth=125):
    F = len(om)
    tg = np.zeros((F, H, W), bool)
    for i in range(F):
        tg[i] = greenmask_img(read_frame(i)[Y0:Y0+H, X0:X0+W], om[i], sth)
    valid = ~(pm & ~om)
    return tg, valid
