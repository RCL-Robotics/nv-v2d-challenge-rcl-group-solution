import sys; sys.path.insert(0,'/work/outputs')
from lib import *
m,v,f,uv,tex=load_mesh('/work/inputs/object_mesh.glb'); cam,K,Twc,rec=load_inputs('/work/inputs'); c0,u=body_axis(v)
R=Renderer(v,f,uv,tex); MD=np.load('median_crops.npz')
k=int(sys.argv[1]); Tf=np.load(f'medfit_{k}.npy')
hx=np.einsum('hij,j->hi',Tf[:,:3,:3],np.array([1.,0,0])); az=np.degrees(np.arctan2(hx[:,2],hx[:,0]))
j=int(np.argmin(np.abs(((az-100)+180)%360-180))); Tb=Tf[j]; azb=az[j]
targets=np.array([int(x) for x in sys.argv[2].split(',')]); Ts=spin(Tb,c0,u,np.radians(targets-azb))
sc=4; K4=K.copy(); K4[:2]*=sc; off=MD['off'][k]*sc
rgb,mk,_=R.render(Ts,K4,160*sc,160*sc,crop=np.repeat(off[None],len(Ts),0))
rgb=(rgb.cpu().numpy()[...,::-1]*255).astype(np.uint8); mk=mk.cpu().numpy()
im=cv2.resize(MD['img'][k][...,::-1],None,fx=sc,fy=sc,interpolation=cv2.INTER_CUBIC)
cr=lambda x:x[35*sc:120*sc,45*sc:115*sc]
tiles=[cr(im)]
for j in range(len(Ts)):
    c=im.copy(); c[mk[j]]=rgb[j][mk[j]]; c=cr(c).copy(); cv2.putText(c,str(targets[j]),(5,25),0,0.8,(0,255,255),2); tiles.append(c)
tiles.append(cr(im))
cv2.imwrite(sys.argv[3],np.hstack(tiles))
