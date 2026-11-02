import sys,time; sys.path.insert(0,'/work/outputs')
from common import *; from fitcore import *; from hyp import *; from constrained import *
K,Kp,Twc,rec=load_inputs('/work/inputs')
mesh=load_mesh('/work/inputs/object_mesh.glb'); fit=Fitter(mesh,K)
d=dict(np.load('/work/scratch/cache_v1.npz'))
n_s=np.array([-0.0216,-0.9938,-0.1089]); n_s/=np.linalg.norm(n_s)
V=np.asarray(mesh.vertices); cm=mesh.bounds.mean(0)
frames=[int(x) for x in sys.argv[1].split(',')]
def init_ab(model,psi,fr):
    R,_=model.pose(psi,0,0); x0,y0,x1,y1=d['box'][fr]; ray=np.linalg.inv(K)@[(x0+x1)/2,(y0+y1)/2,1]
    hc=model.s0+(cm@ (model.R0.T@model.n))  # height of centre above table for this orientation
    hc=-((R[0]@cm)@model.n) + 0  # placeholder
    # solve n.(s*ray) + d = h with h = height of centre: centre = R cm + t, t has n.t = s0-d
    h=(R[0]@cm)@model.n + model.s0  # n.(centre)+d = n.R cm + s0
    s=(h-model.d)/(model.n@ray); X=s*ray; t=X-R[0]@cm
    return t@model.e1, t@model.e2
for dd in [0.62,0.655,0.69]:
    model=PlaneModel(V,SOLE_N,n_s,dd)
    for osg in [False,True]:
        res=[]
        for fr in frames:
            psis=np.linspace(0,2*np.pi,36,endpoint=False)
            ab=np.array([init_ab(model,p,fr) for p in psis])
            p,a,b,s,c,_=fit_plane_model(fit,model,[fr]*36,d,psis,ab[:,0],ab[:,1],iters=40,w_col=0.0,hw=(96,96))
            o=np.argsort(s)[:4]
            p,a,b,s,c,sg=fit_plane_model(fit,model,[fr]*4,d,p[o],a[o],b[o],iters=100,w_col=0.1,opt_sigma=osg)
            k=np.argmin(s+0.3*c); R,t=model.pose(p[k],a[k],b[k])
            res.append((fr,s[k],c[k],sg,np.degrees(p[k])%360,t[0,2]))
        r=np.array(res); print('d=%.3f optsig=%d'%(dd,osg),'mean sil %.3f col %.3f'%(r[:,1].mean(),r[:,2].mean()),' sig',r[:,3].round(3),'psi',r[:,4].round(1),'z',r[:,5].round(3),flush=True)
sys.path.insert(0,'/work/scratch'); from vizfit import viz
