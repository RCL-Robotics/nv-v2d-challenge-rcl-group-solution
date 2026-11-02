from pano import *
# synthetic: render texture at true pose; unwrap with pose rotated by +d (mesh-frame y rotation)
Rr=Renderer(V,F,UV,TEX,K,1152,1536)
Tc=np.linalg.inv(Twc)@np.load('/work/scratch/fitA.npz')['T_world_mesh']
T=Tc[660]
p=K@T[:3,3]; u,v=p[:2]/p[2]; S=int(K[0,0]*0.4/p[2])+8; x0=int(u-S/2); y0=int(v-S/2)
col,mk,_=Rr.render(T[None],crop=(x0,y0,S,S),scale=1)
gimg=(col[0].cpu().numpy()@np.array([0.299,0.587,0.114])*255).astype(np.float32)
for d in [0,20,60]:
    sm,wt=pano_img(gimg,x0,y0,S,T@yawrot(np.radians(d)))
    sc=corr_shift(sm,wt,sign=1.0)
    print(d, 'best shift deg', np.argsort(-sc)[:4]*360/NPHI, sc.max().round(2))
