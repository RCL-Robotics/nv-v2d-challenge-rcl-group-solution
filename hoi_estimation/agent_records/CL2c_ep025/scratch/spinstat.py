import sys
from refine import *
R=Renderer('/work/inputs/object_mesh.glb'); K,Twc,Kp=load_cam()
D=Data(); RF=Refiner(R,K)
d=np.load('vitAT.npz')
def Rz(a):
    c,s=np.cos(np.radians(a)),np.sin(np.radians(a)); return np.array([[c,-s,0],[s,c,0],[0,0,1]])
tt=lambda a:torch.tensor(np.array(a),dtype=torch.float32,device=dev)
angs=np.arange(0,360,10)
def feats(x):
    blue=(x[...,2]-x[...,1]).clamp(min=0); dark=(0.45-x.mean(-1)).clamp(min=0)
    return torch.stack([blue,dark],-1)
for ref,frames in [(60,range(0,114,3)),(830,range(785,859,3))]:
    Rs=tt([d['R'][ref]@Rz(a) for a in angs]); ts=tt([d['t'][ref]]*len(angs)); B=len(angs)
    out={k:[] for k in ['ls','lc','lf','blur_lc']}
    for f in frames:
        img,om,val,x0,y0,n=D.get(f,None)
        I=tt([img]*B);O=tt([om]*B);V=tt([val]*B)
        with torch.no_grad():
            ls,lc,_=RF.losses(Rs,ts,I,O,V,tt([x0]*B),tt([y0]*B),D.S)
            col,a=RF.render(Rs,ts,tt([x0]*B),tt([y0]*B),D.S)
            w=(a*V)[...,None]
            lf=((feats(col)-feats(I)).abs()*w).sum((1,2,3))/(w.sum((1,2,3))+1)
        for k,v in [('ls',ls),('lc',lc),('lf',lf)]: out[k].append(v.cpu().numpy())
    for k in ['ls','lc','lf']:
        A=np.array(out[k]); am=angs[A.argmin(1)]
        print(ref,k,'argmins',np.bincount(am//10,minlength=36).nonzero()[0]*10, np.bincount(am//10)[np.bincount(am//10)>0], 'mean curve min at',angs[A.mean(0).argmin()], 'contrast', (A.mean(0).max()-A.mean(0).min()).round(4))
        print('   mean curve', ' '.join('%d:%.3f'%(a_,v) for a_,v in zip(angs[::3],A.mean(0)[::3])))
