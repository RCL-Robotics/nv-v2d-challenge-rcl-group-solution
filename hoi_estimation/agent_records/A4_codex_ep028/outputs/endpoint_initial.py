import numpy as np,json
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation,Slerp
from scipy.ndimage import gaussian_filter1d
p='/work/inputs/';a=np.load(p+'reconstruction.npz');c=json.load(open(p+'camera.json'));K=np.array(c['K']);W=np.array(c['T_world_cam']);T=np.linalg.inv(W)@a['T_world_mesh'];tracks=np.load('/work/outputs/feature_tracks.npz');head=tracks['head'].copy();tip=tracks['tip'].copy();lines=tracks['blade_line'].copy();N=len(T);start=660
# In this interval SAM omits the blade while RGB shows its upper edge.
# These are image-plane edge directions measured by OpenCV LSD and
# checked in the supplied RGB, NOT 3-D annotations. They encode the
# delayed roll (near-vertical through 840, then horizontal by 870).
lines[759:920]=np.nan
for frame,direction in {760:(-.49,.87),770:(-.29,.96),800:(-.12,.99),810:(-.09,1.),820:(-.17,.99),830:(-.23,.97),840:(-.29,.96),850:(-.86,.51),860:(-.97,.23),870:(-1.,.05),900:(-1.,.05),919:(-1.,.05)}.items():
 lines[frame]=np.array(direction)/np.linalg.norm(direction)
for arr in [head,tip]:
 good=np.isfinite(arr[:,0]);
 for k in range(2):arr[start:,k]=np.interp(np.arange(start,N),np.where(good)[0],arr[good,k]);arr[start:,k]=gaussian_filter1d(arr[start:,k],1.2)
# A head colored component momentarily merged with skin is rejected by
# the saturated threshold and interpolated through its occlusion.
head[727:759]=head[727]+np.linspace(0,1,32)[:,None]*(head[759]-head[727]);
zhead=-.44275588;ztip=.992917;dist=np.linalg.norm(T[:,:3,3],axis=1);Rinit=[];centers=[];axes=[];xprev=T[start,:3,0];Q=T.copy()
for i in range(start,N):
 rays=np.c_[np.array([head[i],tip[i]]),np.ones(2)]@np.linalg.inv(K).T
 def fun(z):
  hp,ep=rays*z[:,None];ax=(ep-hp)/(ztip-zhead);t=hp-zhead*ax;return [np.linalg.norm(ep-hp)-(ztip-zhead),np.linalg.norm(t)-dist[i]]
 fits=[]
 for init in [[dist[i]+.4,dist[i]-.5],[dist[i]-.4,dist[i]+.5]]:
  res=least_squares(fun,init,bounds=(.5,6),xtol=1e-11,ftol=1e-11,gtol=1e-11);hp,ep=rays*res.x[:,None];ax=(ep-hp)/(ztip-zhead);t=hp-zhead*ax;fits.append((ax,t,res.cost))
 # The orange handle is closer than the blade before, during and after
 # the turn. Preserve that continuous branch, anchored at frame 660.
 ax,t,_=min(fits,key=lambda x:x[0][2]);ax=ax/np.linalg.norm(ax);axes.append(ax);centers.append(t)
 if np.isfinite(lines[i]).all():
  l=lines[i];uv=head[i];ray=np.linalg.inv(K)@np.r_[uv,1];ray2=np.linalg.inv(K)@np.r_[uv+100*l,1];normal=np.cross(ray,ray2);xx=np.cross(normal,ax);xx/=np.linalg.norm(xx)
  if xx@xprev<0:xx=-xx
 else:
  xx=xprev-ax*(xprev@ax);xx/=np.linalg.norm(xx)
 yy=np.cross(ax,xx);rr=np.stack([xx,yy,ax],axis=1);Rinit.append(rr);xprev=xx;Q[i,:3,:3]=rr;Q[i,:3,3]=t
# Fill missing blade-angle intervals by interpolating the roll relative
# to a parallel-transport frame, so occlusion cannot cause 180deg flips.
axes=np.array(axes);basis=[];xx=T[start,:3,0]
for ax in axes:
 xx=xx-ax*(xx@ax);xx/=np.linalg.norm(xx);basis.append(np.stack([xx,np.cross(ax,xx),ax],axis=1))
basis=np.array(basis);valid=np.where(np.isfinite(lines[start:,0]))[0];theta=np.unwrap([np.arctan2((basis[j].T@Rinit[j])[1,0],(basis[j].T@Rinit[j])[0,0]) for j in valid]);theta=np.interp(np.arange(N-start),valid,theta);theta=gaussian_filter1d(theta,2)
for j,i in enumerate(range(start,N)):
 Q[i,:3,:3]=basis[j]@Rotation.from_rotvec([0,0,theta[j]]).as_matrix()
np.savez('/work/outputs/endpoint_initial.npz',T_cam_mesh=Q,head=head,tip=tip,lines=lines,theta=theta)
for i in [660,690,700,710,720,730,750,760,800,850,900,987]:print(i,'axis',np.round(Q[i,:3,2],2),'x',np.round(Q[i,:3,0],2),'angle',theta[i-start])
