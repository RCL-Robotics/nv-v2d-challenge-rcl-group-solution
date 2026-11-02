import sys, time; sys.path.insert(0,'/work/outputs')
from hoi_fit import *
from hoi_render import Renderer
S=setup(); pm=S['pm']; dev='cuda'
E=Evidence(S); sr=SilRenderer(decimate(S['mesh']),S['K'],S['Twc'],scale=0.5)
R=Renderer(S['mesh'],1152,1536)
r=np.load('exp4.npz'); th4,a4,b4,dh4=r['th'],r['a'],r['b'],float(r['dh'])
up=torch.tensor(pm.up,dtype=torch.float32,device=dev)
frames=[int(x) for x in sys.argv[1].split(',')]; i0=frames[0]
offs=np.radians(np.arange(-12,13,2.0))
NH=len(offs)
fr=[E.frame(i) for i in frames]
th=torch.tensor(th4[i0]+offs,dtype=torch.float32,device=dev)
a=torch.full((NH,),float(a4[i0]),device=dev,requires_grad=True); b=torch.full((NH,),float(b4[i0]),device=dev,requires_grad=True)
opt=torch.optim.Adam([a,b],lr=0.002)
for it in range(80):
    T=pm.torch_pose(th,a,b,dev).clone(); T[:,:3,3]-=dh4*up
    o=sr.render(T); L=0
    for (M,V,I) in fr: L=L+Evidence.cost(o,M,V,I,blur=max(2*(1-it/40),0))[0]
    opt.zero_grad(); L.sum().backward(); opt.step()
T=pm.torch_pose(th,a,b,dev).clone(); T[:,:3,3]-=dh4*up; T=T.detach().cpu().numpy()
sil=(L/len(fr)).detach().cpu().numpy()
import cv2
Tcw=np.linalg.inv(S['Twc']); x0,y0,w,h=CROP
gr=[cv2.cvtColor(cv2.imread(f'frames/{i+1:04d}.jpg'),cv2.COLOR_BGR2GRAY)[y0:y0+h,x0:x0+w].astype(np.float32) for i in frames]
import h5py
f=h5py.File('/work/inputs/masks.h5','r')
ncc=[]
for k in range(NH):
    rgb,mk,_=R.render(Tcw@T[k],S['K'],crop=CROP)
    g=rgb[0].mean(-1).cpu().numpy()*255; mk=mk[0].cpu().numpy()
    vals=[]
    for j,i in enumerate(frames):
        P=f['person'][i][y0:y0+h,x0:x0+w]; O=f['object'][i][y0:y0+h,x0:x0+w]
        m=mk&~cv2.dilate(P.astype(np.uint8),np.ones((9,9),np.uint8)).astype(bool)&O
        m=cv2.erode(m.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
        A=g[m]; B=gr[j][m]
        # gradient-domain NCC (robust to lighting): compare Sobel magnitudes
        gx=lambda z: np.hypot(cv2.Sobel(z,cv2.CV_32F,1,0,ksize=3),cv2.Sobel(z,cv2.CV_32F,0,1,ksize=3))
        GA=cv2.GaussianBlur(gx(g),(5,5),1.5)[m]; GB=cv2.GaussianBlur(gx(gr[j]),(5,5),1.5)[m]
        n1=np.corrcoef(A,B)[0,1]; n2=np.corrcoef(GA,GB)[0,1]
        vals.append((n1,n2))
    v=np.mean(vals,0); ncc.append(v)
    print(f"yaw offset {np.degrees(offs[k]):5.1f} (abs {np.degrees(th4[i0]+offs[k]):7.1f}) sil {sil[k]:.4f} ncc {v[0]:.3f} gradncc {v[1]:.3f}")
