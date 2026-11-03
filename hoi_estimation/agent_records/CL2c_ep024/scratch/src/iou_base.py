import sys; sys.path.insert(0,'/work/scratch/src')
from common import *
cam,K,Twc,rec=load_inputs(); M=Mesh()
X0,Y0,W,H,S=600,700,320,240,2
def crop_masks():
    om,pm=load_masks()
    om=om[:,Y0:Y0+H,X0:X0+W]; pm=pm[:,Y0:Y0+H,X0:X0+W]
    return om,pm
if __name__=='__main__':
    om,pm=crop_masks(); np.savez_compressed('an/cropmasks.npz',om=om,pm=pm)
    out={}
    for name,Tc in [('base',np.linalg.inv(Twc)@rec['T_world_mesh'].astype(float))]:
        ious=[]
        for b in range(0,748,64):
            m,_,_=M.render(torch.tensor(Tc[b:b+64],device=dev),K,X0,Y0,W,H,1,need_tex=False)
            m=m.cpu().numpy()
            for k in range(m.shape[0]):
                o=om[b+k]; p=pm[b+k]&~o; r=m[k]
                inter=(r&o).sum(); uni=((r|o)&~p).sum()
                ious.append(inter/max(uni,1))
        out[name]=np.array(ious)
        I=out[name]
        for a in range(0,748,25): print(a, np.round(I[a:a+25].mean(),3), np.round(I[a:a+25].min(),3))
        print(name,'mean',I.mean())
    np.save('an/iou_base.npy',out['base'])
