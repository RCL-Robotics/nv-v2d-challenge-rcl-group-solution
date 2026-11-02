import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, torch
from bat_lib import *
s=np.load('/work/scratch/simp.npz'); FR=FastRenderer(s['vs'],s['fs'],s['cs'])
K=np.array([[300.,0,96],[0,300,96],[0,0,1]])
R=torch.eye(3,device='cuda')[None]; 
Rr=torch.tensor([[[0,0,1.],[0,1,0],[-1,0,0]]],device='cuda')  # bat axis along camera x
W=torch.linspace(0,1,192,device='cuda')[None,:,None].expand(1,192,192)  # weight ramp along y (rows)
for name,RR in [('axis-z',R),('axis-x',Rr)]:
    t=torch.tensor([[0.0,0.0,2.0]],device='cuda',requires_grad=True)
    S,C,Dp=FR.render(RR,t,K[None],192); l=(S*W).sum(); l.backward()
    fd=[]
    for k in range(3):
        e=torch.zeros(1,3,device='cuda'); e[0,k]=1e-3
        with torch.no_grad():
            a=(FR.render(RR,t+e,K[None],192)[0]*W).sum(); b=(FR.render(RR,t-e,K[None],192)[0]*W).sum()
        fd.append(((a-b)/2e-3).item())
    print(name,'area',S.sum().item(),'auto',t.grad.cpu().numpy(),'fd',fd)
