from common import *; from render import Renderer; from scorer import *; from param import *
r,cam,pl,m=load(); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam']); Tcw=np.linalg.inv(Twc)
R=Renderer(m,K); msk=h5py.File('/work/inputs/masks.h5'); S=Scorer(R,'/work/scratch/frames',msk)
T1=np.load('/work/scratch/run1.npz')['T_world_mesh']; Tb=r['T_world_mesh'].astype(float)
for i in [300,340,380,420,455,465,470,475,480,490,500,560]:
    T=T1[i]; crop=crop_for(msk['object'][i])
    C=[];P=[]
    for px in np.arange(-8,8.1,2):
        for rz in np.arange(-8,8.1,2):
            for dy in [-0.01,0,0.01]:
                Rl=rodr(np.array([1.,0,0]),np.radians(px))@rodr(np.array([0,0,1.]),np.radians(rz))
                Tn=T.copy(); Tn[:3,:3]=T[:3,:3]@Rl; Tn[:3,3]=T[:3,3]+T[:3,:3]@np.array([0,dy,0])
                C.append((px,rz,dy)); P.append(Tcw@Tn)
    sc=S.score(i,np.stack(P),crop); s=sc[0]+sc[1]; k=s.argmax(); k0=[j for j,c in enumerate(C) if c==(0,0,0)][0]
    print(i,'best tilt (pitch about x, roll about z, dy)',C[k],'score %.3f vs upright %.3f'%(s[k],s[k0]), ' baseline tilt %.1f'%np.degrees(np.arccos(abs(Tb[i,:3,1]@(pl[:3]/np.linalg.norm(pl[:3]))))))
