# python ovcmp.py out.jpg frames result.npz : rows per frame: image | baseline overlay | result overlay
exec(open('setup.py').read().replace("frames=load_video('/work/inputs/video.mp4')","frames=None"))
import sys
fr=[int(x) for x in sys.argv[2].split(',')]
Tr=mesh_to_cam(Twc,np.load(sys.argv[3])['T_world_mesh'])
cap=cv2.VideoCapture('/work/inputs/video.mp4'); imgs={}; i=0
while i<=max(fr):
    ok,f=cap.read()
    if i in fr: imgs[i]=f
    i+=1
rows=[]
def draw(img,T,crop,col):
    rgb,m,d=R.render(T[None],K,crop,(360,270)); mm=m[0].cpu().numpy().astype(np.uint8)
    cnt,_=cv2.findContours(mm,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(img,cnt,-1,col,1)
    r=(rgb[0].cpu().numpy()[:,:,::-1]*255).astype(np.uint8); return r
for i in fr:
    P=K@Tb[i,:3,3]; cx,cy=P[:2]/P[2]; crop=(int(cx)-60,int(cy)-45,120,90)
    im=cv2.resize(imgs[i][crop[1]:crop[1]+90,crop[0]:crop[0]+120],(360,270),interpolation=cv2.INTER_CUBIC)
    a=im.copy(); b=im.copy(); ra=draw(a,Tb[i],crop,(0,0,255)); rb=draw(b,Tr[i],crop,(0,255,0))
    row=np.hstack([im,a,ra,b,rb]); cv2.putText(row,str(i),(5,25),0,0.8,(0,0,255),2); rows.append(row)
cv2.imwrite(sys.argv[1],np.vstack(rows))
