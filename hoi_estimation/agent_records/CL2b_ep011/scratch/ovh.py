# overlay hypotheses: python ovh.py out.jpg frame posesfile(npz with T (N,4,4) cam frame) 
exec(open('setup.py').read().replace("frames=load_video('/work/inputs/video.mp4')","frames=None"))
import sys
out=sys.argv[1]; fr=int(sys.argv[2]); Ts=np.load(sys.argv[3])['T']
cap=cv2.VideoCapture('/work/inputs/video.mp4'); cap.set(cv2.CAP_PROP_POS_FRAMES,fr); ok,img=cap.read()
rows=[]
for T in Ts:
    P=K@T[:3,3]; cx,cy=P[:2]/P[2]
    crop=(int(cx)-60,int(cy)-45,120,90)
    rgb,m,d=R.render(T[None],K,crop,(480,360))
    im=cv2.resize(img[crop[1]:crop[1]+90,crop[0]:crop[0]+120],(480,360),interpolation=cv2.INTER_CUBIC)
    r=(rgb[0].cpu().numpy()[:,:,::-1]*255).astype(np.uint8); mm=m[0].cpu().numpy()
    bl=im.copy(); cnt,_=cv2.findContours(mm.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(bl,cnt,-1,(0,0,255),1)
    rows.append(np.hstack([im,bl,np.where(mm[...,None],r,0)]))
cv2.imwrite(out,np.vstack(rows))
