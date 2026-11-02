import numpy as np,torch
from scipy.spatial.transform import Rotation
from refine import losses

@torch.no_grad()
def search(renderer,Rs,ts,d,rot_reg=.012,shift_reg=.004,rounds=3):
    # Forward-only rasterization avoids the nondeterministic atomic reductions
    # used in differentiable rendering. Search arithmetic and ordering are fixed.
    Rs=np.asarray(Rs,dtype=np.float64).copy();ts=np.asarray(ts,dtype=np.float64).copy();R0=Rs.copy();t0=ts.copy();N=len(Rs);K=renderer.K
    uv0=t0[:,:2]/t0[:,2:]*[K[0,0],K[1,1]]+K[:2,2];uv=uv0.copy();rad=np.linalg.norm(t0,axis=-1)
    def trans(u):
        ray=np.concatenate([(u-K[:2,2])/[K[0,0],K[1,1]],np.ones((len(u),1))],axis=1);return ray/np.linalg.norm(ray,axis=1,keepdims=True)*np.repeat(rad,len(u)//N)[:,None]
    def evaluate(R,t,u,dd):
        l,st,_=losses(renderer,R,t,dd,texture=True);l=l.cpu().numpy();nn=len(R)//N
        theta=np.arccos(np.clip((np.einsum('nij,nij->n',R,np.repeat(R0,nn,axis=0))-1)/2,-1,1));shift=u-np.repeat(uv0,nn,axis=0)
        return l+rot_reg*theta**2+shift_reg*np.sum((shift/4)**2,axis=1),st.cpu().numpy()
    best,stats=evaluate(Rs,ts,uv,d);dd={k:v.repeat_interleave(3,0) for k,v in d.items()}
    axes=np.eye(3)
    for angle,pix in [(12,3),(6,1.5),(3,.75),(1.5,.375),(.75,.1875)]:
        for cycle in range(rounds):
            changed=0
            for dim in range(5):
                rc=np.repeat(Rs[:,None],3,axis=1);uc=np.repeat(uv[:,None],3,axis=1)
                if dim<3:
                    deltas=Rotation.from_rotvec(np.array([0,-1,1])[:,None]*axes[dim]*np.deg2rad(angle)).as_matrix();rc=deltas[None]@rc
                else:uc[:,:,dim-3]+=np.array([0,-pix,pix])
                uc=np.clip(uc,uv0[:,None]-16,uv0[:,None]+16);tc=trans(uc.reshape(-1,2));rc=rc.reshape(-1,3,3);uc=uc.reshape(-1,2)
                cost,st=evaluate(rc,tc,uc,dd);cost=cost.reshape(N,3);ch=np.argmin(cost,axis=1);ind=np.arange(N)*3+ch
                Rs=rc[ind];ts=tc[ind];uv=uc[ind];best=cost[np.arange(N),ch];stats=st[ind];changed+=np.count_nonzero(ch)
            if changed==0:break
    return Rs,ts,best,stats
