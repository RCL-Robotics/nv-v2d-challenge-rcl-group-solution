import time, pickle, sys
from refine import *
from vit import rot_angle_matrix
R=Renderer('/work/inputs/object_mesh.glb'); K,Twc,Kp=load_cam()
D=Data(); RF=Refiner(R,K); F=D.F
resA=pickle.load(open('stageA.pkl','rb'))
byA=[[] for _ in range(F)]
for x in resA: byA[x['f']].append(x)
tt=lambda a:torch.tensor(np.array(a),dtype=torch.float32,device=dev)
WC=1.0; NKEEP=8; NFRESH=3; ITERS=int(sys.argv[2]) if len(sys.argv)>2 else 20
def nms(cs):
    cs=sorted(cs,key=lambda x:x['ls']+WC*x['lc']); out=[]
    for c in cs:
        if all(rot_angle_matrix(c['R'][None],o['R'][None])[0,0]>8 for o in out): out.append(c)
        if len(out)>=NKEEP: break
    return out
def run(order, tag):
    pool=[]; H=None; t1=time.time()
    for n,i in enumerate(order):
        fresh=sorted(byA[i],key=lambda x:x['ls']+WC*x['lc'])[:NFRESH]
        inits=[(c['R'],c['t']) for c in fresh]
        if H is not None: inits+= [(h['R'],h['t']) for h in H]
        d=D.get(i,None)
        B=len(inits)
        img=tt([d[0]]*B);om=tt([d[1]]*B);val=tt([d[2]]*B);x0=tt([d[3]]*B);y0=tt([d[4]]*B)
        Rm,t,ls,lc=RF.optimize(tt([a[0] for a in inits]),tt([a[1] for a in inits]),img,om,val,x0,y0,D.S,iters=ITERS)
        cs=[dict(f=i,src=tag,R=Rm[k].cpu().numpy(),t=t[k].cpu().numpy(),ls=ls[k].item(),lc=lc[k].item()) for k in range(B)]
        pool+=cs; H=nms(cs)
        if n%100==0: print(tag,n,time.time()-t1,flush=True)
    return pool
pf=run(range(F),'fw'); pb=run(range(F-1,-1,-1),'bw')
pickle.dump(pf+pb,open(sys.argv[1],'wb'))
