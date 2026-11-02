import sys, time; sys.path.insert(0,'/work/outputs')
from common import *; from fit import SilFitter; from stages import *
K,Kp,Twc,rec,gp=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs')
s=0.5; Ks=K.copy(); Ks[:2]*=s; w,h=int(W*s),int(H*s)
d=np.load('/work/scratch/masks_half.npz'); obj=d['obj']; per=d['per']; valid=~(per&~obj)
Tb=cam_poses_from_world(Twc,rec['T_world_mesh'].astype(np.float64))
fz=SilFitter(mesh,Ks,w,h); Rb,pb=fz.split(Tb)
a,b=483,510; fr=np.arange(a,b)[::2]
Rm=chordal_mean(Rb[a:b]); pm=np.median(pb[a:b],0)
out={}
for name,R0 in [('base',Rm),('z90',Rm@rot_about_mesh_axis('z',90))]:
    Rg,pg,l=fz.fit(R0[None],pm[None],np.zeros(len(fr),int),obj[fr],valid[fr],iters=150)
    out[name]=fz.join(Rg,pg)[0]
out['raw']=fz.join(Rm[None],pm[None])[0]
R=Renderer(mesh,K,W,H,tex_scale=0.5)
tiles=[]
for i in [486,495,505]:
    img=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg')
    row=[img[350:1050,250:850]]
    for n in ['raw','base','z90']:
        o=R.render(out[n][None]); m=o['mask'][0].cpu().numpy(); c=(o['color'][0].cpu().numpy()[...,::-1]*255).astype(np.uint8)
        ov=img.copy(); ov[m]=(0.4*ov[m]+0.6*c[m]).astype(np.uint8); row.append(ov[350:1050,250:850]); cv2.putText(row[-1],n,(5,25),0,0.8,(0,0,255),2)
    tiles.append(cv2.resize(np.hstack(row),None,fx=0.5,fy=0.5))
cv2.imwrite('/work/scratch/s3.jpg',np.vstack(tiles))
