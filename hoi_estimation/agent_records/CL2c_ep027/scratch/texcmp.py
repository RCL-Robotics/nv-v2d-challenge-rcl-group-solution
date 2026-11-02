import sys; sys.argv=['x','0']
exec(open('/work/scratch/refine_exp.py').read().split("if __name__")[0])
def render_ss(T, crop, Z=4):
    x0,y0,W,H=crop
    Kz=K.copy(); Kz[:2]*=Z; Kz[0,2]+=(Z-1)/2; Kz[1,2]+=(Z-1)/2
    rgb,m,_=render(mesh,torch.tensor(T,dtype=torch.float32,device='cuda'),Kz,H*Z,W*Z,crop=(x0*Z,y0*Z,W*Z,H*Z))
    rgb=torch.nn.functional.avg_pool2d(rgb.permute(0,3,1,2),Z).permute(0,2,3,1).cpu().numpy()
    m=torch.nn.functional.avg_pool2d(m[:,None].float(),Z)[:,0].cpu().numpy()
    return rgb,m
def feats(g, s=1.5):
    gx=cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(g,cv2.CV_32F,0,1,ksize=3)
    a=cv2.GaussianBlur(gx*gx,(0,0),s); b=cv2.GaussianBlur(gy*gy,(0,0),s); c=cv2.GaussianBlur(gx*gy,(0,0),s)
    e=a+b+1e-6
    return np.stack([(a-b)/e, 2*c/e],-1), e
def ncc(a,b,m):
    a=a[m]; b=b[m]; a=a-a.mean(0); b=b-b.mean(0)
    return (a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()+1e-9)
def band(g,m,s1=0.8,s2=3.0):
    mk=m.astype(np.float32)
    def nb(x,s): return cv2.GaussianBlur(x*mk,(0,0),s)/np.maximum(cv2.GaussianBlur(mk,(0,0),s),1e-3)
    return nb(g,s1)-nb(g,s2)
def compare(img, T, crop, obsmask):
    rgb,m=render_ss(T,crop)
    gi=img[...,::-1].mean(-1)/255.
    res=[]
    for k in range(len(T)):
        mm=(m[k]>0.99)&(obsmask>0)
        mm=cv2.erode(mm.astype(np.uint8),np.ones((5,5)))>0
        gr=rgb[k].mean(-1)
        fi,_=feats(gi); fr_,_=feats(gr)
        o=ncc(fi,fr_,mm)
        b=ncc(band(gi,mm),band(gr,mm),mm)
        lo=ncc(band(gi,mm,1.5,6),band(gr,mm,1.5,6),mm)
        res.append((o,b,lo))
    return np.array(res)
