exec(open('/work/outputs/inspect_data.py').read().split('frames=')[0])
frames=list(range(220,291,10))+list(range(520,721,10))
cap=cv2.VideoCapture(str(p/'video.mp4'));h=h5py.File(p/'masks.h5');panels=[]
for f in frames:
 cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,im=cap.read(); m=h['object'][f];ys,xs=np.where(m); x=int(np.median(xs));y=int(np.median(ys)); x0=max(0,x-100);y0=max(0,y-100)
 crop=im[y0:y0+200,x0:x0+200].copy()
 for typ,col in [('T_world_mesh',(255,0,255)),('T_cam_mesh_foundationpose',(0,180,0))]:
  t=T[f].copy() if typ=='T_world_mesh' else a[typ][f].copy();t[:3,3]=T[f,:3,3]
  pts=verts@t[:3,:3].T+t[:3,3];uv=pts@K.T;uv=uv[:,:2]/uv[:,2:];hull=cv2.convexHull(np.round(uv-[x0,y0]).astype('int32'));cv2.polylines(crop,[hull],True,col,1)
  ax=np.array([[0,0,0],[0,0,.17]])@t[:3,:3].T+t[:3,3];uv=ax@K.T;uv=np.round(uv[:,:2]/uv[:,2:]-[x0,y0]).astype(int);cv2.arrowedLine(crop,uv[0],uv[1],col,1,tipLength=.15)
 crop=cv2.resize(crop,(300,300));cv2.putText(crop,str(f),(10,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2);panels.append(crop)
while len(panels)%6:panels.append(np.zeros_like(panels[0]))
cv2.imwrite(str(out/'alternatives_montage.jpg'),np.concatenate([np.concatenate(panels[i:i+6],axis=1) for i in range(0,len(panels),6)],axis=0))
