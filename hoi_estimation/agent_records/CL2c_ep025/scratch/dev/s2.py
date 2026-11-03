import sys, time, pickle, json
sys.path.insert(0,'/work/outputs')
from hoi_lib import *
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K'])
D=pickle.load(open('D.pkl','rb')); F=D.F
R=Renderer('/work/inputs/object_mesh.glb'); RF=Refiner(R,K,D.S)
s1=pickle.load(open(sys.argv[1],'rb'))
by=[[] for _ in range(F)]
for x in s1: by[x[0]].append(x)
NSP=24; angs=np.arange(NSP)*360/NSP
SP=np.array([Rz(a) for a in angs])
# axis hypotheses: dedupe by axis direction
hyps=[]
for i in range(F):
    cs=sorted(by[i],key=lambda x:x[3]+x[4]); keep=[]
    for c in cs:
        if all(np.degrees(np.arccos(np.clip(c[1][:,2]@k[1][:,2],-1,1)))>12 for k in keep): keep.append(c)
    hyps.append(keep[:4])
jobs=[]
for i in range(F):
    for h,c in enumerate(hyps[i]):
        for a in range(NSP): jobs.append((i,h,c[1]@SP[a],c[2]))
tt=lambda a:torch.tensor(np.array(a),dtype=torch.float32,device=dev)
BS=384; L=[]
torch.cuda.synchronize(); t1=time.time()
for s in range(0,len(jobs),BS):
    J=jobs[s:s+BS]
    Lb=RF.evaluate(tt([j[2] for j in J]),tt([j[3] for j in J]),*D.batch([j[0] for j in J]))
    L+=list(Lb.cpu().numpy())
torch.cuda.synchronize(); print('S2 gpu',time.time()-t1, len(jobs))
states=[[] for _ in range(F)]
for j,l in zip(jobs,L): states[j[0]].append(dict(h=j[1],R=j[2],t=j[3],L=l,u=float(RF.unary(*torch.tensor(l)))))
pickle.dump(states,open(sys.argv[2],'wb'))
