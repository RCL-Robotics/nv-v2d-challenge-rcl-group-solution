import sys; sys.path.insert(0,'/work/outputs')
from common import *
def compare(mesh,K,poses,frames,out,scale=1.0,alpha=0.6,size=360):
    """poses: dict name->(F,4,4) cam poses or single 4x4."""
    R=Renderer(mesh,K,W,H,tex_scale=0.5)
    rows=[]
    for i in frames:
        img=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg')
        rs=[];bb=[]
        for n,T in poses.items():
            Tq=T if T.ndim==2 else T[i]
            o=R.render(Tq[None]); m=o['mask'][0].cpu().numpy(); c=(o['color'][0].cpu().numpy()[...,::-1]*255).astype(np.uint8)
            rs.append((n,m,c)); ys,xs=np.nonzero(m); bb.append((xs.min(),xs.max(),ys.min(),ys.max()))
        bb=np.array(bb); x0,x1,y0,y1=bb[:,0].min(),bb[:,1].max(),bb[:,2].min(),bb[:,3].max()
        cx,cy=(x0+x1)//2,(y0+y1)//2; hs=int(max(x1-x0,y1-y0)*0.6)
        def crop(a): a=cv2.copyMakeBorder(a,hs,hs,hs,hs,0); return cv2.resize(a[cy:cy+2*hs,cx:cx+2*hs],(size,size))
        row=[crop(img)]; cv2.putText(row[0],str(i),(5,25),0,0.8,(0,0,255),2)
        for n,m,c in rs:
            ov=img.copy(); ov[m]=((1-alpha)*ov[m]+alpha*c[m]).astype(np.uint8)
            e=cv2.Canny(m.astype(np.uint8)*255,50,100)>0; ov[e]=(0,255,0)
            t=crop(ov); cv2.putText(t,n,(5,25),0,0.8,(0,255,255),2); row.append(t)
        rows.append(np.hstack(row))
    cv2.imwrite(out,np.vstack(rows))
