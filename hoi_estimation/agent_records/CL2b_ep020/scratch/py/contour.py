from common import *
def contour_img(i, poses, colors, out, S=6):
    x0,y0,w,h=700,560,110,95
    im=cv2.resize(frame(i)[y0:y0+h,x0:x0+w],(w*S,h*S),interpolation=cv2.INTER_CUBIC)
    Ks=K.copy(); Ks[:2]*=S; Ks[0,2]=(K[0,2]+0.5)*S-0.5; Ks[1,2]=(K[1,2]+0.5)*S-0.5
    mk=cv2.resize(OM[i][y0:y0+h,x0:x0+w].astype(np.uint8),(w*S,h*S),interpolation=cv2.INTER_NEAREST)
    cs,_=cv2.findContours(mk,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(im,cs,-1,(0,255,0),1)
    for T,c in zip(poses,colors):
        _,m,_=Rd.render(T,Ks,x0*S,y0*S,w*S,h*S,need_rgb=False)
        m=m[0].cpu().numpy().astype(np.uint8)
        cs,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(im,cs,-1,c,1)
    cv2.imwrite(out,im)
if __name__=='__main__':
    i=int(sys.argv[1]); T2=Tcm[i].copy(); T2[:3,3]+=[float(x) for x in sys.argv[2:5]]
    contour_img(i,[Tcm[i],T2],[(0,0,255),(255,0,0)],'/work/scratch/contour.jpg')
