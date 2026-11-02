import sys; sys.argv=['x','0']; exec(open('/work/scratch/ctest.py').read().split('for dd in')[0])
sys.path.insert(0,'/work/scratch'); from vizfit import viz
model=PlaneModel(V,SOLE_N,n_s,0.62); fr=0
psis=np.linspace(0,2*np.pi,36,endpoint=False); ab=np.array([init_ab(model,p,fr) for p in psis])
p,a,b,s,c,_=fit_plane_model(fit,model,[fr]*36,d,psis,ab[:,0],ab[:,1],iters=40,w_col=0.0,hw=(96,96))
o=np.argsort(s)[:4]; p,a,b,s,c,sg=fit_plane_model(fit,model,[fr]*4,d,p[o],a[o],b[o],iters=100,w_col=0.1)
k=np.argmin(s+0.3*c); R,t=model.pose(p[k],a[k],b[k])
import pickle; out=pickle.load(open('/work/scratch/keys.pkl','rb'))
R2,t2,s2,c2=out[50]; j=np.argmin(s2+0.3*c2)
viz(fit,d,[0,0],np.array([R[0],R2[j]]),np.array([t[0],t2[j]]),'/work/scratch/viz/c0.jpg',ncol=1)
print('rot diff constrained vs free(50) %.1f'%rot_angle_deg(R[0],R2[j]), t[0], t2[j])
