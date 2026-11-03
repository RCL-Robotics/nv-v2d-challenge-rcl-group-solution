import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2, h5py, torch, torch.nn.functional as Fn
from hoilib import *
K,Twc,rec,cam=load_inputs('/work/inputs')
g,V,F,UV,TEX=load_mesh('/work/inputs')
R=Renderer(V,F,UV,TEX,K,1152,1536)
Tcw=np.linalg.inv(Twc)
m=h5py.File('/work/inputs/masks.h5','r')
AX=np.array([0.004,0,0.003])
def yawrot(a):
    c,s=np.cos(a),np.sin(a); M=np.eye(4); M[:3,:3]=[[c,0,s],[0,1,0],[-s,0,c]]
    M[:3,3]=AX-M[:3,:3]@AX; return M
def grads(x):  # x (B,1,H,W)
    kx=torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=torch.float32,device=x.device)[None,None]/8
    gx=Fn.conv2d(x,kx,padding=1); gy=Fn.conv2d(x,kx.transpose(2,3),padding=1); return gx,gy
frames=list(range(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3])))
src=sys.argv[4] if len(sys.argv)>4 else 'base'
yaws=np.radians(np.arange(0,360,5))
S=96; SC=4
res=[]
for i in frames:
    T0=Tcw@rec['T_world_mesh'][i].astype(np.float64)
    if src!='base': T0=Tcw@np.load(src)['T_world_mesh'][i]
    p=K@T0[:3,3]; u,v=p[:2]/p[2]; x0=int(u-S/2); y0=int(v-S/2)
    img=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg')[y0:y0+S,x0:x0+S]
    om=m['object'][i][y0:y0+S,x0:x0+S]; pm=m['person'][i][y0:y0+S,x0:x0+S]
    Ts=np.stack([T0@yawrot(a) for a in yaws])
    col,mk,_=R.render(Ts,crop=(x0,y0,S,S),scale=SC)
    col=col.permute(0,3,1,2); mk=mk[:,None].float()
    col=Fn.avg_pool2d(col,2); mk=Fn.avg_pool2d(mk,2)   # 2x res
    real=torch.tensor(cv2.resize(img,(2*S,2*S),interpolation=cv2.INTER_CUBIC)[...,::-1].copy(),device='cuda').float().permute(2,0,1)[None]/255
    omt=torch.tensor(cv2.resize(om.astype(np.uint8),(2*S,2*S),interpolation=cv2.INTER_NEAREST),device='cuda').float()[None,None]
    pmt=torch.tensor(cv2.resize(pm.astype(np.uint8),(2*S,2*S),interpolation=cv2.INTER_NEAREST),device='cuda').float()[None,None]
    valid=1-pmt
    inter=(mk*omt*valid).sum((1,2,3)); uni=((mk+omt-mk*omt)*valid).sum((1,2,3)); iou=inter/uni
    gr=real.mean(1,keepdim=True); gc=col.mean(1,keepdim=True)
    gxr,gyr=grads(gr); gxc,gyc=grads(gc)
    # polarity-invariant orientation agreement weighted; region = dilated object
    reg=(Fn.max_pool2d(mk,7,1,3)>0).float()*valid
    dot=(gxr*gxc+gyr*gyc); nr=(gxr**2+gyr**2); nc=(gxc**2+gyc**2)
    ec=((dot**2)*reg).sum((1,2,3))/torch.sqrt(((nr**2)*reg).sum((1,2,3))*((nc**2)*reg).sum((1,2,3))+1e-12)
    # signed ncc of gray within render mask
    w=mk*valid; mr=(gr*w).sum((1,2,3))/w.sum((1,2,3)); mc=(gc*w).sum((1,2,3))/w.sum((1,2,3))
    a=(gr-mr.view(-1,1,1,1)); b=(gc-mc.view(-1,1,1,1))
    ncc=(a*b*w).sum((1,2,3))/torch.sqrt((a*a*w).sum((1,2,3))*(b*b*w).sum((1,2,3))+1e-12)
    # color: distance in rgb inside mask
    cd=(((real-col)**2).sum(1,keepdim=True)*w).sum((1,2,3))/w.sum((1,2,3))
    res.append(torch.stack([iou,ec,ncc,cd],1).cpu().numpy())
res=np.array(res)  # F,Y,4
np.save(f'/work/scratch/yawscore_{frames[0]}_{frames[-1]}.npy',res)
names=['iou','edge','ncc','negcdist']
res[:,:,3]*=-1
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,4,figsize=(20,6))
for k in range(4):
    z=(res[:,:,k]-res[:,:,k].mean(1,keepdims=True))/(res[:,:,k].std(1,keepdims=True)+1e-9)
    ax[k].imshow(z,aspect='auto',extent=[0,360,frames[-1],frames[0]]); ax[k].set_title(names[k])
    print(names[k],'mean-z argmax',np.degrees(yaws[np.argmax(z.mean(0))]), 'top3', np.degrees(yaws[np.argsort(-z.mean(0))[:6]]))
plt.savefig(sys.argv[5] if len(sys.argv)>5 else '/work/scratch/vis/yawscore.png',dpi=50)
