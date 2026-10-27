import numpy as np,cv2,json,h5py
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
root='/work/inputs';cam=json.load(open(root+'/camera.json'));K=np.array(cam['K']);Ki=np.linalg.inv(K);C=np.linalg.inv(cam['T_world_cam']);T=C@np.load(root+'/reconstruction.npz')['T_world_mesh'];h=h5py.File(root+'/masks.h5');cap=cv2.VideoCapture(root+'/video.mp4');ims=[]
for f in range(341):
 ok,b=cap.read();ims.append(cv2.cvtColor(b,cv2.COLOR_BGR2GRAY))
out=[]
for f in list(range(115,190))+list(range(250,340)):
 obj=h['object'][f].astype('uint8');yy,xx=np.where(obj)
 if not len(xx):continue
 obj[:int(np.quantile(yy,.88))]=0
 pts=cv2.goodFeaturesToTrack(ims[f],30,.005,3,mask=cv2.dilate(obj,np.ones((3,3),np.uint8)),blockSize=3)
 if pts is None:continue
 nex,st,err=cv2.calcOpticalFlowPyrLK(ims[f],ims[f+1],pts,None,winSize=(21,21),maxLevel=3)
 back,bst,_=cv2.calcOpticalFlowPyrLK(ims[f+1],ims[f],nex,None,winSize=(21,21),maxLevel=3)
 keep=(st[:,0]>0)&(bst[:,0]>0)&(np.linalg.norm(back[:,0]-pts[:,0],axis=1)<.7)&(err[:,0]<30)
 p0=pts[keep,0];p1=nex[keep,0]
 if len(p0)<3:continue
 n0=T[f,:3,2];n1=T[f+1,:3,2];t0=T[f,:3,3];t1=T[f+1,:3,3];rays=np.c_[p0,np.ones(len(p0))]@Ki.T;dep=np.dot(n0,t0-.093*n0)/(rays@n0);X=rays*dep[:,None]
 keep=(dep>0)&(abs(dep-t0[2])<.5);X=X[keep];p1=p1[keep]
 if len(X)<3:continue
 cr=np.cross(n0,n1);angle=np.arctan2(np.linalg.norm(cr),n0@n1);RR=Rotation.from_rotvec(cr/(np.linalg.norm(cr)+1e-15)*angle).as_matrix();Y=(X-t0)@RR.T
 def res(v):
  R=Rotation.from_rotvec(n1*v[0]).as_matrix();pred=Y@R.T+t1;pp=pred@K.T;pp=pp[:,:2]/pp[:,2:]+v[1:];return (pp-p1).ravel()
 opt=least_squares(res,[0,0,0],loss='soft_l1',f_scale=.5,bounds=([-.5,-20,-20],[.5,20,20]));out.append([f,np.rad2deg(opt.x[0]),len(X),np.mean(np.abs(res(opt.x)))])
np.save('/work/outputs/flow_probe.npy',out)
for a,b in [(117,130),(130,140),(140,150),(150,180),(250,270),(270,280),(280,290),(290,310),(310,330)]:
 x=np.array(out);s=x[(x[:,0]>=a)&(x[:,0]<b)];print(a,b,'degrees',s[:,1].sum(),'frames',len(s),'error',np.mean(s[:,3]))
