import sys; sys.path.insert(0,'/work/outputs'); sys.path.insert(0,'/work/scratch/code')
from lib import *
from scorelib import scores
m,v,f,uv,tex=load_mesh('/work/inputs/object_mesh.glb'); cam,K,Twc,rec=load_inputs('/work/inputs'); c0,u=body_axis(v)
R=Renderer(v,f,uv,tex); MD=dict(np.load('median_crops.npz')); Fi=Fitter(R,K,MD,c0)
k=int(sys.argv[1]); Tf=np.load(f'medfit_{k}.npy')
hx=np.einsum('hij,j->hi',Tf[:,:3,:3],np.array([1.,0,0])); az=np.degrees(np.arctan2(hx[:,2],hx[:,0]))
j=int(np.argmin(np.abs(((az-100)+180)%360-180))); Tb=Tf[j]; azb=az[j]
targets=np.arange(0,360,5); Ts=spin(Tb,c0,u,np.radians(targets-azb))
s=scores(Fi,Ts,[k])
# edge chamfer on 4x
sc=4; K4=K.copy(); K4[:2]*=sc; off=MD['off'][k]*sc
im=cv2.resize(MD['img'][k],None,fx=sc,fy=sc,interpolation=cv2.INTER_CUBIC)
g=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY); E=cv2.Canny(cv2.GaussianBlur(g,(5,5),0),20,50)
dt=cv2.distanceTransform((E==0).astype(np.uint8),cv2.DIST_L2,3); dt=np.minimum(dt,12)
valid=cv2.resize((~MD['pm'][k]).astype(np.uint8),None,fx=sc,fy=sc,interpolation=cv2.INTER_NEAREST)>0
ch=[]
for a in range(0,len(Ts),24):
    _,mk,_=R.render(Ts[a:a+24],K4,160*sc,160*sc,crop=np.repeat(off[None],len(Ts[a:a+24]),0)); mk=mk.cpu().numpy().astype(np.uint8)
    for q in mk:
        bd=(cv2.morphologyEx(q,cv2.MORPH_GRADIENT,np.ones((3,3)))>0)&valid
        ch.append(dt[bd].mean())
s['cham']=-np.array(ch)/10
print('az  ',' '.join('%4d'%x for x in targets))
for kk in s: print('%-4s'%kk,' '.join('%4d'%(100*x) for x in s[kk]), ' best', targets[np.argmax(s[kk]) if kk not in ('err','cerr') else np.argmin(s[kk])])
