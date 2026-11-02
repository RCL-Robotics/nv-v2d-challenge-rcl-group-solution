import numpy as np, sys
d=np.load(sys.argv[1]); data=d['data']/float(sys.argv[2])
F,NB=data.shape; yaws=np.arange(NB)*3.0
PH=[(0,124,'s'),(125,213,'c'),(214,436,'s'),(437,846,'c'),(847,979,'s')]
ph=np.zeros(F,int)
for k,(a,b,_) in enumerate(PH): ph[a:b+1]=k
def tl(i):
    md=0 if (PH[ph[i]][2]=='s' and ph[i]==ph[i-1]) else 2
    return [(dd,0.3*dd*dd/float(sys.argv[2])) for dd in range(-md,md+1)]
def lse(x,axis=0):
    m=x.max(axis); return m+np.log(np.exp(x-np.expand_dims(m,axis)).sum(axis))
la=np.zeros((F,NB)); la[0]=-data[0]; la[0]-=lse(la[0])
for i in range(1,F):
    st=np.stack([np.roll(la[i-1],dd)-c for dd,c in tl(i)]); la[i]=lse(st)-data[i]; la[i]-=lse(la[i])
lb=np.zeros((F,NB))
for i in range(F-2,-1,-1):
    st=np.stack([np.roll(lb[i+1]-data[i+1],-dd)-c for dd,c in tl(i+1)]); lb[i]=lse(st); lb[i]-=lse(lb[i])
p=la+lb; p=np.exp(p-p.max(1,keepdims=True)); p/=p.sum(1,keepdims=True)
for i in [60,170,300,600,700,800,900]:
    o=np.argsort(-p[i])[:5]; print(i,[(int(yaws[k]),round(p[i][k],2)) for k in o])
