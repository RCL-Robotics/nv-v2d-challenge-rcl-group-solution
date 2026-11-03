import sys; sys.path.insert(0,'/work/outputs')
from hoi_lib import *
K,Twc,rec=load_inputs('/work/inputs')
R=MeshRenderer('/work/inputs/object_mesh.glb')
Tw=rec['T_world_mesh'].astype(np.float64); Tc=mesh_to_cam(Twc,Tw)
fr=[int(a) for a in sys.argv[2].split(',')]
cap=cv2.VideoCapture('/work/inputs/video.mp4'); imgs={}
i=0
while i<=max(fr):
    ok,f=cap.read()
    if i in fr: imgs[i]=f
    i+=1
src = sys.argv[3] if len(sys.argv)>3 else 'base'
if src!='base': Tc = mesh_to_cam(Twc, np.load(src)['T_world_mesh'])
out=[]
for i in fr:
    P=K@Tc[i,:3,3]; cx,cy=P[:2]/P[2]
    crop=(int(cx)-60,int(cy)-45,120,90)
    rgb,m,d=R.render(Tc[i:i+1],K,crop,(480,360))
    im=cv2.resize(imgs[i][crop[1]:crop[1]+90,crop[0]:crop[0]+120],(480,360),interpolation=cv2.INTER_CUBIC)
    r=(rgb[0].cpu().numpy()[:,:,::-1]*255).astype(np.uint8); mm=m[0].cpu().numpy()
    bl=im.copy(); bl[mm]=(0.5*im[mm]+0.5*r[mm]).astype(np.uint8)
    cnt,_=cv2.findContours(mm.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(bl,cnt,-1,(0,0,255),1)
    full=np.hstack([im,bl,np.where(mm[...,None],r,0)])
    cv2.putText(full,str(i),(5,25),0,0.8,(0,0,255),2)
    out.append(full)
cv2.imwrite(sys.argv[1],np.vstack(out))
