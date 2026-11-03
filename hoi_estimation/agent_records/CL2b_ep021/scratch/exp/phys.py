import sys; sys.path.insert(0,'/work/outputs')
from common import *; from fit import SilFitter
K,Kp,Twc,rec,gp=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs')
v=np.asarray(mesh.vertices); fn=np.array([-0.03238796,-0.00993733,0.99942597])  # mesh foot-plane normal (points down)
pl=np.array(gp['plane']); n=pl[:3]
fz=SilFitter(mesh,K,W,H)
segs=[(0,120),(120,317),(317,375),(375,483),(483,510),(510,582),(582,678)]
def stats(Tw,name):
    out=[]
    for a,b in segs:
        Rw=Tw[a:b,:3,:3]; tilt=np.degrees(np.arccos(np.clip(-(Rw@fn)@n,-1,1)))
        lo=[(v@Tw[i,:3,:3].T+Tw[i,:3,3])@n+pl[3] for i in range(a,b,3)]; lo=np.array([l.min() for l in lo])
        out.append('%5.2f°/%5.1fmm'%(np.median(tilt),1000*np.median(lo)))
    print(name.ljust(8),' '.join(out))
stats(rec['T_world_mesh'].astype(np.float64),'base')
for f in sys.argv[1:]:
    J=np.load(f); stats(world_poses_from_cam(Twc,fz.join(J['R'],J['p'])),f.split('/')[-1][:8])
