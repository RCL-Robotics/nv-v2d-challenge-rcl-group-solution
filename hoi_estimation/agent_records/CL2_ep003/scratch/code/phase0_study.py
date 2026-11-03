import sys; sys.path.insert(0,'/work/outputs')
from lib import *
exec(open('/work/scratch/code/upright.py').read())
f5=h5py.File('/work/inputs/masks.h5')
S=np.load('static_median.npz'); k=int(sys.argv[1]); a,b=S['phases'][k]; off=S['off'][k]
x0,y0=off
oms=np.stack([f5['object'][i][y0:y0+160,x0:x0+160] for i in range(a,b)])
pms=np.stack([f5['person'][i][y0:y0+160,x0:x0+160] for i in range(a,b)])
ok=~(pms&~oms); frac=(oms&ok).sum(0)/np.maximum(ok.sum(0),1)
md=np.nan_to_num(S['med'][k]).astype(np.uint8)
Tm=Tc[a].copy(); Tm[:3,3]=np.median(Tc[a:b,:3,3],0); Tu=upright(Tm)
hx=Tu[:3,:3]@np.array([1.,0,0]); az0=np.degrees(np.arctan2(hx[2],hx[0]))
targets=np.arange(0,360,10)
Ts=spin(Tu,c0,u,np.radians(targets-az0))
_,mk,_=R.render(Ts,K,160,160,crop=np.repeat(off[None].astype(float),len(Ts),0))
mk=mk.cpu().numpy().astype(float)
om=frac>0.5
# best 2D shift per hyp
best=[]
for j in range(len(Ts)):
    bs=(-1,0,0)
    for dy in range(-4,5):
        for dx in range(-4,5):
            r=np.roll(mk[j],(dy,dx),(0,1)).astype(bool); iou=(r&om).sum()/(r|om).sum()
            if iou>bs[0]: bs=(iou,dx,dy)
    best.append(bs)
for t,bs in zip(targets,best): print('az %4d iou %.3f shift %d %d'%(t,*bs))
sc=4; tiles=[]
for t in [int(x) for x in sys.argv[2].split(',')]:
    j=list(targets).index(t%360); r=np.roll(mk[j],(best[j][2],best[j][1]),(0,1)).astype(np.uint8)
    c=cv2.resize(md,None,fx=sc,fy=sc,interpolation=cv2.INTER_CUBIC)
    cnt,_=cv2.findContours(cv2.resize(r,None,fx=sc,fy=sc,interpolation=cv2.INTER_NEAREST),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(c,cnt,-1,(0,255,0),1)
    cnt,_=cv2.findContours(cv2.resize(om.astype(np.uint8),None,fx=sc,fy=sc,interpolation=cv2.INTER_NEAREST),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(c,cnt,-1,(0,0,255),1)
    c=c[35*sc:125*sc,40*sc:120*sc].copy(); cv2.putText(c,'az%d'%t,(5,25),0,0.8,(255,255,0),2); tiles.append(c)
cv2.imwrite(sys.argv[3],np.hstack(tiles))
