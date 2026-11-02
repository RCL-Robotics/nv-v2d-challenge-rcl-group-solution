from traj import *
from render import Renderer
import cv2
out=np.load('test_out.npz')['T_world_mesh']
Rn=Renderer(INP+'/object_mesh.glb')
x0,y0,W,H=(548,600,278,421)
S=1.042
def cam_scaled(Tw_):
    Tc_=Tcw@Tw_; Rc=Tc_[:3,:3]; cmc=Rc@C_M+Tc_[:3,3]
    T=np.eye(4); T[:3,:3]=Rc*S; T[:3,3]=cmc-Rc@C_M*S; return T
def yawed(Tw_,d):
    Ry=R.from_rotvec(NRM*np.radians(d)).as_matrix(); cw=Tw_[:3,:3]@C_M+Tw_[:3,3]
    T=np.eye(4); T[:3,:3]=Ry@Tw_[:3,:3]; T[:3,3]=cw-T[:3,:3]@C_M; return T
def scores(i,Tlist):
    img=cv2.imread(f'frames/{i:04d}.jpg',0)[y0:y0+H,x0:x0+W]
    o=OM[i][y0:y0+H,x0:x0+W]; p=PM[i][y0:y0+H,x0:x0+W]
    inner=cv2.erode(o.astype(np.uint8),np.ones((7,7)))>0
    ei=(cv2.Canny(cv2.GaussianBlur(img,(3,3),0),20,60)>0)&inner&~p
    dti=cv2.distanceTransform((~ei).astype(np.uint8),cv2.DIST_L2,3)
    Ts=np.stack([cam_scaled(T) for T in Tlist])
    rgb,m,z=Rn.render(Ts,K,x0,y0,W,H)
    rgb=rgb.cpu().numpy(); m=m.cpu().numpy(); res=[]
    for j in range(len(Tlist)):
        g=(rgb[j].mean(-1)*255).astype(np.uint8)
        er=(cv2.Canny(cv2.GaussianBlur(g,(3,3),0),30,90)>0)&inner&~p&(cv2.erode(m[j].astype(np.uint8),np.ones((7,7)))>0)
        dtr=cv2.distanceTransform((~er).astype(np.uint8),cv2.DIST_L2,3)
        c=0.5*np.minimum(dti[er],8).mean()+0.5*np.minimum(dtr[ei],8).mean()
        res.append(c)
    return np.array(res)
ds=np.arange(-8,8.1,2)
for a,b in [(0,100),(138,217),(255,337),(379,467)]:
    acc=[]
    for i in range(a,b,8):
        Tl=[yawed(out[i],d) for d in ds]+[Tw[i]]
        acc.append(scores(i,Tl))
    acc=np.array(acc).mean(0)
    print(a,b,'edge chamfer vs yaw offset:',dict(zip(ds.tolist(),acc[:-1].round(3))),'baseline pose: %.3f'%acc[-1])
