import sys; sys.path.insert(0,'/work/outputs')
from lib import *
exec(open('/work/scratch/code/upright.py').read())
t=int(sys.argv[1]); a,b=int(sys.argv[2]),int(sys.argv[3])
Tm=Tc[t].copy(); Tm[:3,3]=np.median(Tc[a:b,:3,3],0); Tu=upright(Tm)
angs=np.radians(np.arange(0,360,30)); Ts=spin(Tu,c0,u,angs)
sc=4; K4=K.copy(); K4[:2]*=sc
off=D['off'][t]*sc+np.array([45,40])*sc
rgb,mk,_=R.render(Ts,K4,70*sc,80*sc,crop=np.repeat(off[None],len(Ts),0))
rgb=(rgb.cpu().numpy()[...,::-1]*255).astype(np.uint8)
im=cv2.resize(D['img'][t][40:120,45:115][...,::-1],None,fx=sc,fy=sc,interpolation=cv2.INTER_CUBIC)
tiles=[im]+[r for r in rgb]
for k,a in enumerate(range(0,360,30)): cv2.putText(tiles[k+1],str(a),(5,25),0,0.8,(0,0,255),2)
tiles.append(im)
cv2.imwrite(sys.argv[4],np.vstack([np.hstack(tiles[:7]),np.hstack(tiles[7:])]))
