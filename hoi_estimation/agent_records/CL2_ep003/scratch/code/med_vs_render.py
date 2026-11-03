import sys; sys.path.insert(0,'/work/outputs')
from lib import *
exec(open('/work/scratch/code/upright.py').read())
S=np.load('static_median.npz'); k=int(sys.argv[1]); a,b=S['phases'][k]
off=S['off'][k]; md=np.nan_to_num(S['med'][k]).astype(np.uint8)
Tm=Tc[a].copy(); Tm[:3,3]=np.median(Tc[a:b,:3,3],0); Tu=upright(Tm)
hx=Tu[:3,:3]@np.array([1.,0,0]); az0=np.degrees(np.arctan2(hx[2],hx[0]))
targets=np.array([int(x) for x in sys.argv[2].split(',')])
Ts=spin(Tu,c0,u,np.radians(targets-az0))
# align translation in image: shift so rendered bottom-centre matches mask lower body (use SAM median mask)
sc=4; K4=K.copy(); K4[:2]*=sc
o=(off+np.array([40,35]))*sc
rgb,mk,_=R.render(Ts,K4,80*sc,90*sc,crop=np.repeat(o[None],len(Ts),0))
rgb=(rgb.cpu().numpy()[...,::-1]*255).astype(np.uint8); mk=mk.cpu().numpy()
im=cv2.resize(md[35:125,40:120],None,fx=sc,fy=sc,interpolation=cv2.INTER_CUBIC)
tiles=[im]
for j,t in enumerate(targets):
    c=im.copy(); c[mk[j]]=rgb[j][mk[j]]; cv2.putText(c,'az%d'%t,(5,25),0,0.8,(0,0,255),2); tiles.append(c)
while len(tiles)%6: tiles.append(im)
cv2.imwrite(sys.argv[3],np.vstack([np.hstack(tiles[i:i+6]) for i in range(0,len(tiles),6)]))
