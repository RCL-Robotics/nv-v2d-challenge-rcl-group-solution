# full-res crop: image with SAM mask contour (green) and render contour (red)
import sys; sys.path.insert(0,'/work/outputs')
from common import *
f=int(sys.argv[1]); out=sys.argv[2]; poses=sys.argv[3] if len(sys.argv)>3 else 'base'
D=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs')
cap=cv2.VideoCapture('/work/inputs/video.mp4'); cap.set(cv2.CAP_PROP_POS_FRAMES,f); ok,im=cap.read()
import h5py
m=h5py.File('/work/inputs/masks.h5')['object'][f]; p=h5py.File('/work/inputs/masks.h5')['person'][f]
Tw=D['rec']['T_world_mesh'].astype(np.float64) if poses=='base' else np.load(poses)['T_world_mesh']
R=Renderer(mesh,D['K'],1152,1536); r=R.render((np.linalg.inv(D['Twc'])@Tw[f])[None],rgb=False)['mask'][0].cpu().numpy()
for mm,c in [(m,(0,255,0)),(r,(0,0,255)),(p,(255,0,0))]:
    cs,_=cv2.findContours(mm.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE); cv2.drawContours(im,cs,-1,c,1)
ys,xs=np.nonzero(m|r); y0,y1,x0,x1=ys.min()-20,ys.max()+20,xs.min()-20,xs.max()+20
cv2.imwrite(out,im[max(y0,0):y1,max(x0,0):x1])
