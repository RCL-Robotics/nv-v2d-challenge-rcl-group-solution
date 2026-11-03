import sys; sys.path.insert(0,'/work/scratch/src')
from common import *
cam,K,Twc,rec=load_inputs(); M=Mesh()
X0,Y0,W,H=600,700,320,240
d=np.load('an/cropmasks.npz'); om,pm=d['om'],d['pm']
Tc=np.linalg.inv(Twc)@rec['T_world_mesh'].astype(float)
def greenmask(i):
    im=read_frame(i)[Y0:Y0+H,X0:X0+W]; hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    g=(hsv[...,0]>25)&(hsv[...,0]<60)&(hsv[...,1]>90)&(hsv[...,2]>50)
    sd=cv2.dilate(om[i].astype(np.uint8),np.ones((9,9),np.uint8))>0
    return g&sd
fr=[0,50,200,300,475,540,700]
tgt=torch.tensor(np.stack([greenmask(i) for i in fr]),device=dev).float()
tgt2=torch.tensor(om[fr],device=dev).float()
valid=torch.tensor(~(pm[fr]&~om[fr]),device=dev).float()
T0=torch.tensor(Tc[fr],device=dev,dtype=torch.float32)
for name,tg in [('green',tgt),('sam',tgt2)]:
    w=torch.zeros(len(fr),3,device=dev,requires_grad=True); t=torch.zeros(len(fr),3,device=dev,requires_grad=True)
    opt=torch.optim.Adam([w,t],lr=0.01)
    for it in range(150):
        Rd=so3_exp(w+1e-9); T=T0.clone()
        T=torch.cat([torch.cat([Rd@T0[:,:3,:3], (T0[:,:3,3]+t*0.1)[...,None]],2), T0[:,3:]],1)
        sil,_=render_soft(M,T,K,X0,Y0,W,H)
        inter=(sil*tg*valid).sum((1,2)); uni=((sil+tg-sil*tg)*valid).sum((1,2))
        iou=inter/uni; loss=(1-iou).sum()
        if it==0: print(name,'init',iou.detach().cpu().numpy().round(3))
        opt.zero_grad(); loss.backward(); opt.step()
    print(name,'final',iou.detach().cpu().numpy().round(3))
    print(' rot deg',(w.norm(dim=1)*57.3).detach().cpu().numpy().round(1),'t cm',(t*10).detach().cpu().numpy().round(1).tolist())
