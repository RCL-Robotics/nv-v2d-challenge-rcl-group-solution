# usage: t_cmp.py out.png hyps.npz key1,key2,.. frames...
import sys; sys.path.insert(0,'/work/outputs')
from lib import *
from fit import CropFitter
cam,K,Twc,rec=load_inputs('/work/inputs')
FULL=np.load('/work/scratch/cache/full.npy',mmap_mode='r'); OBJ=np.load('/work/scratch/cache/obj.npy',mmap_mode='r')
out,hp,keys=sys.argv[1],sys.argv[2],sys.argv[3].split(',')
ids=[int(a) for a in sys.argv[4:]]
h=np.load(hp)
fit=CropFitter('/work/inputs/object_mesh.glb',res=64)
rd=Renderer('/work/inputs/object_mesh.glb',K,1152,1536)
def getT(k,i):
    if k=='base0': return np.linalg.inv(Twc)@rec['T_world_mesh'][i].astype(np.float64)
    if k+'/T' in h: return h[k+'/T'][i]
    return fit.to_T(h[k+'/R'][i:i+1],h[k+'/p'][i:i+1])[0]
rows=[]
for i in ids:
    im=FULL[i]; om=OBJ[i]
    ys,xs=np.nonzero(om); cx,cy=(xs.min()+xs.max())//2,(ys.min()+ys.max())//2
    hs=max(xs.max()-xs.min(),ys.max()-ys.min())//2+40
    x0,y0=max(cx-hs,0),max(cy-hs,0); x1,y1=x0+2*hs,y0+2*hs
    cols=[im[y0:y1,x0:x1]]
    cont=cv2.findContours(om.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]
    for k in keys:
        T=getT(k,i); o=rd.render(T[None]); m=o['mask'][0].cpu().numpy(); c=(o['color'][0].cpu().numpy()[:,:,::-1]*255)
        r=np.full_like(im,255); r[m]=c[m]; cv2.drawContours(r,cont,-1,(0,200,0),1)
        cols.append(r[y0:y1,x0:x1])
    row=np.concatenate([cv2.resize(c,(260,260)) for c in cols],1)
    cv2.putText(row,str(i),(5,20),0,0.7,(0,0,255),2); rows.append(row)
cv2.imwrite(out,np.concatenate(rows,0))
