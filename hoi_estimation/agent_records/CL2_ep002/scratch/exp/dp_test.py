import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from hoilib import *
K,Twc,rec,cam=load_inputs('/work/inputs')
T=np.load('/work/scratch/fitA.npz')['T_world_mesh']
d=np.load('/work/scratch/pscoreA.npz'); A=d['A']
Z=(A-np.nanmean(A,1,keepdims=True))/np.nanstd(A,1,keepdims=True)
AXC=np.array([0.004,0,0.003])
def wyaw(R): x=R[...,:,0]; return np.degrees(np.arctan2(x[...,2],x[...,0]))
def yawrot(a):
    c,s=np.cos(a),np.sin(a); M=np.eye(4); M[:3,:3]=[[c,0,s],[0,1,0],[-s,0,c]]; M[:3,3]=AXC-M[:3,:3]@AXC; return M
NB=120
LW=float(sys.argv[5]) if len(sys.argv)>5 else 3.0
a,b=int(sys.argv[1]),int(sys.argv[2]); lo,hi=float(sys.argv[3]),float(sys.argv[4])
lugdir=np.array([np.cos(np.radians(120)),0,np.sin(np.radians(120))])
Tcw=np.linalg.inv(Twc)
data=np.zeros((b-a,NB))
for k,i in enumerate(range(a,b)):
    y0=wyaw(T[i,:3,:3])
    for j in range(NB):
        ay=j*3; dl=np.radians(ay-y0)
        Tn=T[i]@yawrot(-dl)
        if abs(((wyaw(Tn[:3,:3])-ay+180)%360)-180)>4: Tn=T[i]@yawrot(dl)
        Tcn=Tcw@Tn; ray=Tcn[:3,:3]@AXC+Tcn[:3,3]; ray/=np.linalg.norm(ray)
        ld=Tcn[:3,:3]@lugdir; lugc=(ld@ray)**2
        z=Z[i,j] if not np.isnan(Z[i,j]) else 0
        inrange = (lo<=ay<=hi)
        data[k,j]=-z+LW*lugc+(0 if inrange else 100)
# viterbi
maxd=2; cost=data[0].copy(); bp=np.zeros((b-a,NB),int)
for k in range(1,b-a):
    best=np.full(NB,np.inf); arg=np.zeros(NB,int)
    for dd in range(-maxd,maxd+1):
        c=np.roll(cost,dd)+0.3*dd*dd
        m=c<best; best[m]=c[m]; arg[m]=((np.arange(NB)-dd)%NB)[m]
    cost=best+data[k]; bp[k]=arg
j=int(np.argmin(cost)); path=[j]
for k in range(b-a-1,0,-1): j=bp[k,j]; path.append(j)
path=np.array(path[::-1])*3
print('total',cost.min().round(1),'mean texture z on path',np.mean([np.nan_to_num(Z[a+k,path[k]//3]) for k in range(b-a)]).round(3))
print(path[::10])
np.save(f'/work/scratch/dp_{a}_{b}_{int(lo)}.npy',path)
