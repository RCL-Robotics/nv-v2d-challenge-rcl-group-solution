import sys; sys.path.insert(0,'/work/outputs')
from common import *
K,Kp,Twc=load_camera('/work/inputs'); m=load_mesh('/work/inputs/object_mesh.glb'); r=Renderer(m,K)
rr=np.load('/work/inputs/reconstruction.npz'); Tc0=np.linalg.inv(Twc)@rr['T_world_mesh'].astype(np.float64)
cxy=np.array([0.005,-0.004,0]); Fz=np.eye(4); Fz[:3,:3]=np.diag([-1,-1,1.]); Fz[:3,3]=2*cxy
rows=[]
for i in [int(x) for x in sys.argv[1].split(',')]:
    img=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg')
    mk,_,_,_=r.render(Tc0[i][None]); ys,xs=np.nonzero(mk[0].cpu().numpy())
    x0,y0=xs.min()-15,ys.min()-15; w=xs.max()-xs.min()+30; h=int((ys.max()-ys.min())*0.45)
    tiles=[cv2.resize(img[y0:y0+h,x0:x0+w],(w*4,h*4),interpolation=cv2.INTER_CUBIC)]
    for T in [Tc0[i],Tc0[i]@Fz]:
        # supersampled render
        _,rgb,_,_=r.render(T[None],crop=(x0,y0,w,h),H=h*4,W=w*4); tiles.append((rgb[0].cpu().numpy()[:,:,::-1]*255).astype(np.uint8))
    rows.append(np.hstack(tiles))
    cv2.imwrite(f'hinge_{i}.jpg',rows[-1])
