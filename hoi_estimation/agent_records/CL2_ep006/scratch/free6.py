import sys; sys.path.insert(0,'/work/outputs')
from hoi_fit import *
import roma
dev='cuda'
S=setup(); pm=S['pm']
E=Evidence(S); sr=SilRenderer(decimate(S['mesh']),S['K'],S['Twc'],scale=0.5)
T0=np.load(sys.argv[1]); frames=[int(x) for x in sys.argv[2].split(',')]
Tcw=np.linalg.inv(S['Twc'])
for fi in frames:
    M,V,I=E.frame(fi)
    Tw=torch.tensor(T0[fi],dtype=torch.float32,device=dev)
    w=torch.zeros(3,device=dev,requires_grad=True); t=torch.zeros(3,device=dev,requires_grad=True)
    opt=torch.optim.Adam([{'params':[w],'lr':0.005},{'params':[t],'lr':0.003}])
    c=torch.tensor(S['c'],dtype=torch.float32,device=dev)
    for it in range(200):
        # perturb about the base-contact point in world frame
        R=roma.rotvec_to_rotmat(w)
        T=Tw.clone(); pc=Tw[:3,:3]@c+Tw[:3,3]
        T=torch.eye(4,device=dev).clone(); 
        Rn=R@Tw[:3,:3]; T[:3,:3]=Rn; T[:3,3]=pc+t-Rn@c
        o=sr.render(T[None])
        As=o[...,1]; Af=o[...,0]
        L=((V*M*(1-As)).sum()+(V*(1-M)*As).sum())/(V*M).sum()
        opt.zero_grad(); L.backward(); opt.step()
    Tn=T.detach().cpu().numpy()
    up=S['up']; dn=Tn[:3,:3]@S['nr']
    tilt=np.degrees(np.arccos(np.clip(dn@-up,-1,1)))
    # direction of tilt
    print(fi,'loss',L.item(),'tilt',tilt.round(2),'tilt dir (world)',(dn+up).round(3),'dt cm',(t.detach().cpu().numpy()*100).round(1),'rot deg',np.degrees(np.linalg.norm(w.detach().cpu().numpy())).round(2))
