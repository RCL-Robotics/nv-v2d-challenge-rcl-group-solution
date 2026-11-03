import sys; sys.path.insert(0,'/work/outputs')
from common import *
D=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs')
M=np.load('/work/scratch/mobj_half.npy',mmap_mode='r'); P=np.load('/work/scratch/mper_half.npy',mmap_mode='r')
def ious(Tw):
    Tc=np.linalg.inv(D['Twc'])@Tw
    K=D['K'].copy(); K[:2]*=0.5
    R=Renderer(mesh,K,576,768)
    out=np.zeros(len(Tw))
    for s in range(0,len(Tw),32):
        o=R.render(Tc[s:s+32],rgb=False)['mask'].cpu().numpy()
        for i in range(o.shape[0]):
            f=s+i; valid=~(P[f]&~M[f])
            r=o[i]&valid; m=M[f]
            out[f]=(r&m).sum()/max((r|m).sum(),1)
    return out
if __name__=='__main__':
    Tw=D['rec']['T_world_mesh'].astype(np.float64) if len(sys.argv)<2 else np.load(sys.argv[1])['T_world_mesh']
    iou=ious(Tw)
    np.save('/work/scratch/iou_last.npy',iou)
    np.set_printoptions(precision=2,linewidth=200)
    print('mean',iou.mean(),'median',np.median(iou))
    print(np.array([iou[i:i+25].mean() for i in range(0,993,25)]))
    # motion
    R=Tw[:,:3,:3]; dr=rot_angle_deg(R[:-1],R[1:]); dt=np.linalg.norm(np.diff(Tw[:,:3,3],axis=0),axis=1)*100
    print('rot step deg per 25-block max:',np.array([dr[i:i+25].max() for i in range(0,992,25)]))
    print('trans step cm per 25-block max:',np.array([dt[i:i+25].max() for i in range(0,992,25)]))
