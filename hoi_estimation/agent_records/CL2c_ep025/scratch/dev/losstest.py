import sys, pickle, json
sys.path.insert(0,'/work/outputs')
from hoi_lib import *
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K'])
D=pickle.load(open('D.pkl','rb'))
RR=Renderer('/work/inputs/object_mesh.glb'); RF=Refiner(RR,K,192)
states=pickle.load(open('s2.pkl','rb'))
tt=lambda a:torch.tensor(np.array(a),dtype=torch.float32,device=dev)
def blur(x,s):
    k=int(3*s)*2+1; g=torch.exp(-(torch.arange(k,device=dev)-k//2)**2/(2*s*s)); g=g/g.sum()
    x=x.permute(0,3,1,2); C=x.shape[1]
    x=Fn.conv2d(x,g.view(1,1,1,k).repeat(C,1,1,1),padding=(0,k//2),groups=C)
    x=Fn.conv2d(x,g.view(1,1,k,1).repeat(C,1,1,1),padding=(k//2,0),groups=C)
    return x.permute(0,2,3,1)
def variants(col,a,img,om,val):
    out={}
    w=(a*val)[...,None]
    for s in [0,1.5,3]:
        cb=blur(col,s) if s else col; ib=blur(img*(1),s) if s else img
        ab=blur(a[...,None],s)[...,0] if s else a
        # compare only where render alpha (blurred) is substantial
        ww=w
        g=((ib*cb*ww).sum((1,2))/((cb*cb*ww).sum((1,2))+1e-6)).clamp(0.7,1.5)
        out['lc_s%.1f'%s]=((cb*g[:,None,None]-ib).abs()*ww).sum((1,2,3))/(ww.sum((1,2,3))+1)
        bl=lambda x:(x[...,2]-x[...,1])
        out['blue_s%.1f'%s]=((bl(cb)-bl(ib)).abs()*ww[...,0]).sum((1,2))/(ww[...,0].sum((1,2))+1)
    return out
frames=[int(a) for a in sys.argv[1].split(',')]
for f in frames:
    st=[s for s in states[f] if s['h']==0]; B=len(st)
    b=D.batch([f]*B)
    with torch.no_grad():
        col,a=RF.render(tt([s['R'] for s in st]),tt([s['t'] for s in st]),b[3],b[4])
        out=variants(col,a,b[0],b[1],b[2])
    for k,v in out.items():
        v=v.cpu().numpy(); print(f,k,'argmin',np.argmin(v)*15,'contrast %.3f'%((v.max()-v.min())/v.mean()), ' '.join('%.3f'%x for x in v[::2]))
