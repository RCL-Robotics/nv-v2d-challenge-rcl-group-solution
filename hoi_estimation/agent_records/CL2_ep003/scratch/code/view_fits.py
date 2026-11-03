import sys; sys.path.insert(0,'/work/outputs')
from lib import *
m,v,f,uv,tex=load_mesh('/work/inputs/object_mesh.glb'); cam,K,Twc,rec=load_inputs('/work/inputs')
R=Renderer(v,f,uv,tex); MD=np.load('median_crops.npz')
k=int(sys.argv[1]); Tf=np.load(f'medfit_{k}.npy'); js=[int(x) for x in sys.argv[2].split(',')]
sc=4; K4=K.copy(); K4[:2]*=sc; off=MD['off'][k]*sc
rgb,mk,_=R.render(Tf[js],K4,160*sc,160*sc,crop=np.repeat(off[None],len(js),0))
rgb=(rgb.cpu().numpy()[...,::-1]*255).astype(np.uint8); mk=mk.cpu().numpy()
im=cv2.resize(MD['img'][k][...,::-1],None,fx=sc,fy=sc,interpolation=cv2.INTER_CUBIC)
tiles=[im[30*sc:125*sc,35*sc:120*sc]]
for j in range(len(js)):
    c=im.copy(); c[mk[j]]=rgb[j][mk[j]]
    cnt,_=cv2.findContours(mk[j].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    d=im.copy(); cv2.drawContours(d,cnt,-1,(0,255,0),1)
    tiles.append(c[30*sc:125*sc,35*sc:120*sc]); tiles.append(d[30*sc:125*sc,35*sc:120*sc])
cv2.imwrite(sys.argv[3],np.hstack(tiles))
