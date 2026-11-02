import sys
from refine import *
R=Renderer('/work/inputs/object_mesh.glb'); K,Twc,Kp=load_cam()
D=Data(); RF=Refiner(R,K)
d=np.load(sys.argv[1])
def Rz(a):
    c,s=np.cos(np.radians(a)),np.sin(np.radians(a)); return np.array([[c,-s,0],[s,c,0],[0,0,1]])
def feats(x):  # x (...,3) rgb in [0,1]
    blue=(x[...,2]-x[...,1]).clamp(min=0)
    dark=(0.45-x.mean(-1)).clamp(min=0)
    return torch.stack([blue,dark],-1)
tt=lambda a:torch.tensor(np.array(a),dtype=torch.float32,device=dev)
for f in [int(a) for a in sys.argv[2].split(',')]:
    img,om,val,x0,y0,n=D.get(f,None)
    angs=np.arange(0,360,20)
    Rs=tt([d['R'][f]@Rz(a) for a in angs]); ts=tt([d['t'][f]]*len(angs)); B=len(angs)
    I=tt([img]*B);O=tt([om]*B);V=tt([val]*B)
    with torch.no_grad():
        ls,lc,a=RF.losses(Rs,ts,I,O,V,tt([x0]*B),tt([y0]*B),D.S)
        col,a=RF.render(Rs,ts,tt([x0]*B),tt([y0]*B),D.S)
        fr=feats(col); fi=feats(I)
        w=(a*V)[...,None]
        lf=((fr-fi).abs()*w).sum((1,2,3))/(w.sum((1,2,3))+1)
        imb=(fi[0,...,0]*O[0]).sum()/O[0].sum()
    print(f,'img blue %.3f'%imb.item())
    print('  lc', ' '.join('%d:%.3f'%(a_,l) for a_,l in zip(angs,lc.cpu().numpy())))
    print('  lf', ' '.join('%d:%.4f'%(a_,l) for a_,l in zip(angs,lf.cpu().numpy())))
