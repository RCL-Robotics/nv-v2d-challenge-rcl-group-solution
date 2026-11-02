import pickle, numpy as np, sys, json, torch
sys.path.insert(0,'/work/outputs')
from hoi_lib import *
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K'])
D=pickle.load(open('/work/scratch/cache/data.pkl','rb'))
RF=Refiner(Renderer('/work/inputs/object_mesh.glb'),K,D.S)
Rs,ts,un=pickle.load(open('/work/scratch/cache/s2.pkl','rb'))
tt=lambda a:torch.tensor(np.asarray(a),dtype=torch.float32,device=dev)
for f,h in [(20,0),(100,0),(830,0),(730,0),(720,0),(170,0),(610,0)]:
    sl=slice(h*24,h*24+24)
    L=RF.evaluate(tt(Rs[f][sl]),tt(ts[f][sl]),*D.batch([f]*24)).cpu().numpy()
    u=RF.unary(*torch.tensor(L).T).numpy()
    print(f,'strap argmin',L[:,4].argmin()*15,'range %.2f-%.2f'%(L[:,4].min(),L[:,4].max()),'| unary argmin',u.argmin()*15, ' '.join('%.2f'%x for x in L[::2,4]))
