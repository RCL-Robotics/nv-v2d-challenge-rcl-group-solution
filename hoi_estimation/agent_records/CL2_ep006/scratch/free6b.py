import sys; sys.path.insert(0,'/work/outputs')
from hoi_fit import *
import roma
dev='cuda'
S=setup(); E=Evidence(S); sr=SilRenderer(decimate(S['mesh']),S['K'],S['Twc'],scale=0.5)
T0=S['Tb']; frames=[int(x) for x in sys.argv[1].split(',')]; mode=sys.argv[2]
Tcw=np.linalg.inv(S['Twc'])
out={}
for fi in frames:
    M,V,I=E.frame(fi)
    Tw=torch.tensor(T0[fi],dtype=torch.float32,device=dev)
    w=torch.zeros(3,device=dev,requires_grad=True); t=torch.zeros(3,device=dev,requires_grad=True)
    opt=torch.optim.Adam([{'params':[w],'lr':0.005},{'params':[t],'lr':0.003}])
    c=torch.tensor(S['c'],dtype=torch.float32,device=dev)
    for it in range(300):
        R=roma.rotvec_to_rotmat(w); pc=Tw[:3,:3]@c+Tw[:3,3]
        T=torch.eye(4,device=dev); Rn=R@Tw[:3,:3]; T[:3,:3]=Rn; T[:3,3]=pc+t-Rn@c
        o=sr.render(T[None]); A=o[...,0] if mode=='full' else o[...,1]
        L=((V*M*(1-A)).sum()+(V*(1-M)*A).sum())/(V*M).sum()
        opt.zero_grad(); L.backward(); opt.step()
    Tn=T.detach().cpu().numpy(); out[fi]=Tn
    zc=lambda X:(Tcw@X)[2,3]
    dn=Tn[:3,:3]@S['nr']; tilt=np.degrees(np.arccos(np.clip(dn@-S['up'],-1,1)))
    print(fi,'loss',round(L.item(),4),'tilt',tilt.round(2),'z base',zc(T0[fi]).round(3),'z fit',zc(Tn).round(3),'base height above floor', ((Tn[:3,:3]@S['c']+Tn[:3,3])@S['up']+S['floor_d']).round(3))
Tall=S['Tb'].copy()
for k,v in out.items(): Tall[k]=v
np.save('free6b_T.npy',Tall)
