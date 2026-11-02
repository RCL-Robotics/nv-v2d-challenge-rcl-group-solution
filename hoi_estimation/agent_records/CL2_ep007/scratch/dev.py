import sys,os,pickle,json,time; sys.path.insert(0,'/work/outputs')
import importlib
from common import *; from fitcore import Fitter
import stages; importlib.reload(stages)
inp='/work/inputs'; cache='/work/scratch/cache'
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]",*a,flush=True)
def cached(name,fn):
    p=os.path.join(cache,name+'.pkl')
    if os.path.exists(p): log('cache hit',name); return pickle.load(open(p,'rb'))
    r=fn(); pickle.dump(r,open(p,'wb')); return r
K,Kp,Twc,rec=load_inputs(inp)
plane=np.array(json.load(open(f'{inp}/ground_plane.json'))['plane'])
n_c=Twc[:3,:3].T@plane[:3]; d_c=plane[3]+plane[:3]@Twc[:3,3]
mesh=load_mesh(f'{inp}/object_mesh.glb'); V=np.asarray(mesh.vertices); cm=mesh.bounds.mean(0)
fit=Fitter(mesh,K)
data=pickle.load(open(f'{cache}/crops.pkl','rb'))
info={}
import numpy as _np; data['touch']=_np.load('/work/scratch/touch.npy')
T0=time.time()
sys.path.insert(0,'/work/outputs'); import solve
table=solve.table_geometry(K,n_c,d_c); print(table)
out=stages.run_all(fit,data,K,n_c,d_c,V,cm,896,rec,Twc,table,cached,log,info)
pickle.dump(info,open('/work/scratch/dev_info.pkl','wb'))
log('elapsed',time.time()-T0)
pickle.dump(out,open('/work/scratch/dev_out.pkl','wb'))
