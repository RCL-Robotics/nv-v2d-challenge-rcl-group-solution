import sys, time; sys.path.insert(0,'/work/outputs')
from lib import *
m,v,f,uv,tex=load_mesh('/work/inputs/object_mesh.glb')
cam,K,Twc,rec=load_inputs('/work/inputs'); c0,u=body_axis(v)
Tc=np.linalg.inv(Twc)[None]@rec['T_world_mesh'].astype(np.float64)
D=dict(np.load('crops160.npz')); R=Renderer(v,f,uv,tex); Fi=Fitter(R,K,D,c0)
gp=json.load(open('/work/inputs/ground_plane.json'))['plane']; nw=np.array(gp[:3])
nc=np.linalg.inv(Twc)[:3,:3]@nw; nc/=np.linalg.norm(nc)
if nc[1]>0: nc=-nc   # up in camera (-y)
def upright(T):
    # rotate T minimally so the lid axis points along the table normal, pivot at body centre
    a=T[:3,:3]@u; ax=np.cross(a,nc); s=np.linalg.norm(ax); ang=np.arcsin(min(s,1))
    Rm=rot_about(ax/s,[ang])[0] if s>1e-9 else np.eye(3)
    out=T.copy(); out[:3,:3]=Rm@T[:3,:3]; cc=T[:3,:3]@c0+T[:3,3]; out[:3,3]=cc-out[:3,:3]@c0; return out
a,b=int(sys.argv[1]),int(sys.argv[2])
pf=D['pm'][a:b].sum((1,2)); 
fr=np.arange(a,b); fr=fr[np.argsort([ (D['pm'][i]&~D['om'][i])[40:120,40:120].sum() for i in fr])][:16]
print('frames',sorted(fr))
# phase mean pose: median translation, baseline rotation of medoid
Tm=Tc[fr[0]].copy(); Tm[:3,3]=np.median(Tc[a:b,:3,3],0)
Tu=upright(Tm)
angs=np.radians(np.arange(0,360,20))
for name,T in [('base',Tm),('upright',Tu)]:
    T0=spin(T,c0,u,angs)
    Tf,iou,ncc=Fi.fit(T0,[list(fr)]*len(T0),iters=int(sys.argv[3]))
    tilt=[np.degrees(np.arccos(np.clip((Tq[:3,:3]@u)@nc,-1,1))) for Tq in Tf]
    print(name); print(' iou',' '.join('%2d'%(100*x) for x in iou)); print(' ncc',' '.join('%2d'%(100*x) for x in ncc)); print(' tilt',' '.join('%2d'%x for x in tilt))
    print(' dz cm',' '.join('%2d'%x for x in (Tf[:,2,3]-T0[:,2,3])*100))
