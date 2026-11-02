import sys,json; sys.path.insert(0,'/work/outputs')
from lib import *
m,v,f,uv,tex=load_mesh('/work/inputs/object_mesh.glb'); cam,K,Twc,rec=load_inputs('/work/inputs'); c0,u=body_axis(v)
Tcw=np.linalg.inv(Twc); V1=Tcw[None]@np.load('v1/object_poses_world.npz')['T_world_mesh']
log=json.load(open('v1/object_poses_world_log.json')); MD=np.load('median_crops.npz')
# body-only mesh (drop handle) for width measurement
keep=v[:,0]<0.03; fb=f[keep[f].all(1)]
R=Renderer(v,fb,uv,tex)
sc=8
def edges_width(g,row):
    p=g[row].astype(np.float64); gr=np.gradient(p)
    mid=len(p)//2
    return p,gr
for k,s in enumerate(log['static']):
    a,b=s['phase']; T=V1[a]
    # use phase median crops computed earlier on (manual) phases; offsets from MD
    off=MD['off'][k]; img=MD['img'][k].astype(np.float32)
    g=cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_RGB2GRAY).astype(np.float32)
    g8=cv2.resize(g,None,fx=sc,fy=sc,interpolation=cv2.INTER_CUBIC)
    K8=K.copy(); K8[:2]*=sc
    _,mk,_=R.render(T[None],K8,160*sc,160*sc,crop=(off*sc)[None]); mk=mk[0].cpu().numpy()
    ys,xs=np.nonzero(mk); top=ys.min(); bot=ys.max(); Hh=bot-top
    out=[]
    for fr_ in np.linspace(0.12,0.3,7):   # upper silver band (below lid)
        row=int(top+fr_*Hh); cols=np.nonzero(mk[row])[0]; wr=cols.max()-cols.min()+1
        cx=(cols.max()+cols.min())/2
        gr=np.gradient(cv2.GaussianBlur(g8,(0,0),sc*0.5)[row])
        # left edge: dark->bright => positive gradient near rendered left edge; right: negative
        L0=int(cols.min()); R0=int(cols.max()); win=int(6*sc)
        lseg=gr[L0-win:L0+win]; rseg=gr[R0-win:R0+win]
        le=L0-win+np.argmax(np.abs(lseg)); re=R0-win+np.argmax(np.abs(rseg))
        out.append(((re-le)/wr, le-L0, re-R0))
    o=np.array(out)
    ratio=np.median(o[:,0]); zc=(T[:3,:3]@c0+T[:3,3])[2]
    print('phase',a,b,'obs/render width ratio %.3f -> z %.3f (%+.1fcm vs anchored %.3f) edge offsets L %.1f R %.1f px'%(ratio,zc/ratio,100*(zc/ratio-zc),zc,np.median(o[:,1])/sc,np.median(o[:,2])/sc))
