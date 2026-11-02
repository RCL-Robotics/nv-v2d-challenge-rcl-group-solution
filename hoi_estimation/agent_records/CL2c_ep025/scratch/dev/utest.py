import sys, pickle, json
sys.path.insert(0,'/work/outputs')
from hoi_lib import *
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K'])
D=pickle.load(open('D.pkl','rb'))
RR=Renderer('/work/inputs/object_mesh.glb'); RF=Refiner(RR,K,192)
states=pickle.load(open('s2.pkl','rb'))
tt=lambda a:torch.tensor(np.array(a),dtype=torch.float32,device=dev)
def ev(f,h):
    st=[s for s in states[f] if s['h']==h]; B=len(st); b=D.batch([f]*B)
    L=RF.evaluate(tt([s['R'] for s in st]),tt([s['t'] for s in st]),*b)
    return L.cpu().numpy(), RF.unary(*L.T).cpu().numpy()
for f in [0,30,60,100,800,830,850]:
    L,u=ev(f,0); print(f,'unary argmin',np.argmin(u)*15,'blue argmin',np.argmin(L[:,2])*15,'dark',np.argmin(L[:,3])*15, 'lc',np.argmin(L[:,1])*15, 'u contrast %.3f'%(u.max()-u.min()))
for f in [190,250,300,700]:
    for h in range(len(set(s['h'] for s in states[f]))):
        L,u=ev(f,h); k=np.argmin(u); print(f,'h',h,'ax',np.round(states[f][h*24]['R'][:,2],2),'best u %.3f'%u[k],'comps',L[k].round(3),'spin',k*15)
