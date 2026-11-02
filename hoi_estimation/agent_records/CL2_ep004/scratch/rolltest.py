import sys; sys.path.insert(0,'/work/scratch')
from common import *
import torch, cv2
from bat_lib import Renderer, W_IMG, H_IMG
from scipy.spatial.transform import Rotation as Ro
ep,A=get_ep(); R=Renderer(ep.mesh)
T=np.load('/work/scratch/run1/object_poses_world.npz')['T_world_mesh']; Tc=ep.Tcw[None]@T
cap=cv2.VideoCapture('/work/inputs/video.mp4')
for frames in [list(range(0,110,10)), list(range(710,800,9))]:
    angs=np.arange(0,360,15); tot=np.zeros((len(angs),2))
    for i in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,img=cap.read(); img=img[...,::-1].astype(np.float32)/255
        x0,y0,side=ep.wins[i]; crop=(int(x0),int(y0),int(side),int(side))
        Ts=[]
        for a in angs:
            Tr=Tc[i].copy(); Tr[:3,:3]=Tc[i,:3,:3]@Ro.from_euler('z',a,degrees=True).as_matrix(); Ts.append(Tr)
        m,_,rgb=R.render(np.stack(Ts),ep.K,W_IMG,H_IMG,color=True,crop=crop)
        m=m.cpu().numpy(); rgb=rgb.cpu().numpy()
        I=np.zeros((crop[3],crop[2],3),np.float32); ys=slice(max(0,crop[1]),crop[1]+crop[3]); xs=slice(max(0,crop[0]),crop[0]+crop[2])
        I[ys.start-crop[1]:ys.stop-crop[1] if ys.stop<=1152 else 1152-crop[1], xs.start-crop[0]:(xs.stop-crop[0]) if xs.stop<=1536 else 1536-crop[0]]=img[ys,xs]
        om=ep.datas[i]['M']; M=cv2.resize(om,(crop[2],crop[3]))>0.5
        for k in range(len(angs)):
            W=(m[k]>0)&M
            C=rgb[k][W]; X=I[W]
            g=(C*X).sum(0)/((C*C).sum(0)+1e-6)
            tot[k,0]+=np.abs(X-g*C).mean()
            # gradient-structure term: correlation of luminance after removing mean
            cl=C.mean(1); xl=X.mean(1); cl=cl-cl.mean(); xl=xl-xl.mean()
            tot[k,1]+=-(cl*xl).sum()/np.sqrt((cl**2).sum()*(xl**2).sum()+1e-9)
    tot/=len(frames)
    print('frames',frames[0],'-',frames[-1])
    for k,a in enumerate(angs): print('  roll %3d  colL1 %.4f  -corr %.3f'%(a,tot[k,0],tot[k,1]))
