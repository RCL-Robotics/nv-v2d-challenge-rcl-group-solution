import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2, h5py
from hoi_lib import *
v,f,uv,tex=load_mesh('/work/inputs/object_mesh.glb')
K,Twc,Kp,W,H=load_camera('/work/inputs')
R=Renderer(v,f,uv,tex)
def show(Tcm_list, frames, out, labels=None, S=4):
    x0,y0,w,h=690,545,128,112
    Ks=K.copy(); Ks[:2]*=S; Ks[0,2]=(K[0,2]+0.5)*S-0.5; Ks[1,2]=(K[1,2]+0.5)*S-0.5
    rows=[]
    for i in frames:
        im=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg')[y0:y0+h,x0:x0+w]
        im=cv2.resize(im,(w*S,h*S),interpolation=cv2.INTER_CUBIC)
        row=[im]
        for T in Tcm_list:
            rgb,m,d=R.render(T[i],Ks,x0*S,y0*S,w*S,h*S)
            rgb=(rgb[0].cpu().numpy()*255).astype(np.uint8)[...,::-1]; m=m[0].cpu().numpy()
            o=im.copy(); o[m]=rgb[m]; row.append(o)
        row=np.hstack(row); cv2.putText(row,str(i),(5,30),0,1,(0,255,255),2); rows.append(row)
    cv2.imwrite(out,np.vstack(rows))
if __name__=='__main__':
    r=np.load('/work/inputs/reconstruction.npz')
    Tcm=np.linalg.inv(Twc)@r['T_world_mesh'].astype(np.float64)
    show([Tcm],[int(a) for a in sys.argv[2:]],sys.argv[1])
