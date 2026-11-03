import cv2,json,numpy as np,h5py
from pathlib import Path
def extract(inputs, start=660):
 p=Path(inputs);obs=json.load(open('/work/outputs/observations.json'));cap=cv2.VideoCapture(str(p/'video.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,start);f=h5py.File(p/'masks.h5');F=len(obs);head=np.full((F,2),np.nan);tip=head.copy();quality=np.zeros((F,2));lines=np.full((F,2),np.nan);lasthead=None;lasttip=None
 for i in range(start,F):
  _,im=cap.read();d=obs[i];cen=np.array(d['center']);ax=np.array(d['axis']);mask=f['object'][i].astype('uint8');hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV);orange=((hsv[:,:,0]>=3)&(hsv[:,:,0]<23)&(hsv[:,:,1]>180)&(hsv[:,:,2]>90)).astype('uint8');n,lab,stats,cent=cv2.connectedComponentsWithStats(orange);entries=[]
  # The head and handle swap their vertical ordering around frame 712,
  # as the actor rotates the squeegee behind her back. The head remains
  # the left endpoint until it is hidden, and the upper endpoint later.
  for k in range(1,n):
   if stats[k,4]<12:continue
   xy=cent[k];side=abs(np.cross(ax,xy-cen));along=(xy-cen)@ax
   # SAM omits the orange head behind the arm in frames 790-890.
   # Admit color features in a narrow extrapolated shaft corridor.
   if side<22 and -550<along<550 and 480<xy[1]<1100 and 400<xy[0]<980:
    entries.append((xy,stats[k,4],side))
  if i<710:
   expected_h=cen+ax*d['bounds'][1];expected_t=cen+ax*d['bounds'][0]
  else:
   if ax[0]<0:expected_h=cen+ax*d['bounds'][1];expected_t=cen+ax*d['bounds'][0]
   else:expected_h=cen+ax*d['bounds'][0];expected_t=cen+ax*d['bounds'][1]
  # Match color components to temporally predicted endpoint positions,
  # allowing brief occlusion without changing head/handle identities.
  for role,expected,prev,arr,qi in [('head',expected_h,lasthead,head,0),('tip',expected_t,lasttip,tip,1)]:
   cand=[]
   for xy,area,side in entries:
    if prev is not None:
     dis=np.linalg.norm(xy-prev)
     if dis>max(85,30+abs(i-(last_h_i if qi==0 else last_t_i))*6):continue
    else:dis=np.linalg.norm(xy-expected)
    if qi==0 and i>=730 and xy[1]>720:continue
    if qi==1 and i>=740 and xy[1]<740:continue
    if qi==0 and i<735:continue
    if qi==0 and i>=790 and area<20:continue
    cand.append((dis+5*side-3*np.log(area),xy,area))
   if cand:
    _,xy,area=min(cand,key=lambda x:x[0]);arr[i]=xy;quality[i,qi]=area
    if qi==0:lasthead=xy;last_h_i=i
    else:lasttip=xy;last_t_i=i
  # Extract a blade direction from mask pixels close to the head.
  # The small dark orange end loop needs the more permissive color
  # threshold already restricted to SAM; the saturated threshold above
  # is only used to recover the head outside SAM near skin.
  cs=np.array([x for x in d['orange'] if x[2]>=20])
  if len(cs):
   if i<710:jj=np.argmin(cs[:,1])
   elif i<740:jj=np.argmax(cs[:,0])
   else:jj=np.argmax(cs[:,1])
   xy=cs[jj,:2]
   if (i<740 and xy[0]>650) or (i>=740 and xy[1]>740):
    tip[i]=xy;quality[i,1]=cs[jj,2];lasttip=xy;last_t_i=i
  if i<=730:
   cs=np.array([x for x in d['orange'] if x[2]>60])
   if len(cs) and cs[:,0].min()<650:
    xy=cs[np.argmin(cs[:,0]),:2];head[i]=xy;quality[i,0]=cs[np.argmin(cs[:,0]),2];lasthead=xy;last_h_i=i
  if 731<=i<=758:head[i]=np.nan;quality[i,0]=0;lasthead=None
  if i>=759 and np.isfinite(head[i]).all():
   near=[(xy,area) for xy,area,side in entries if np.linalg.norm(xy-head[i])<24]
   if near:head[i]=np.average(np.array([x for x,a in near]),axis=0,weights=[a for x,a in near]);quality[i,0]=sum(a for x,a in near);lasthead=head[i]
  if np.isfinite(head[i]).all():
   hp=head[i];yy,xx=np.where(mask);pts=np.c_[xx,yy];delta=pts-hp;axis_dir=(tip[i]-hp) if np.isfinite(tip[i]).all() else ax;axis_dir=axis_dir/np.linalg.norm(axis_dir);long=delta@axis_dir;side=np.abs(np.cross(axis_dir,delta));keep=(np.linalg.norm(delta,axis=1)<85)&(long<12)&(side>12)
   pts=pts[keep]
   if len(pts)>70:
    mid=np.median(pts,axis=0);_,_,vv=np.linalg.svd(pts-mid,full_matrices=False);direction=vv[0];res=np.abs(np.cross(direction,pts-mid));pts=pts[res<np.percentile(res,75)];_,_,vv=np.linalg.svd(pts-pts.mean(0),full_matrices=False);lines[i]=vv[0]
  if i%50==0:print(i,'head',head[i],'tip',tip[i],flush=True)
 return head,tip,quality,lines
if __name__=='__main__':
 h,t,q,l=extract('/work/inputs');np.savez('/work/outputs/feature_tracks.npz',head=h,tip=t,quality=q,blade_line=l)
