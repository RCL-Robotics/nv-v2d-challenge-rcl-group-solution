import sys; sys.path.insert(0,'/work/scratch/s')
from rc import *
def vis_poses(rc, frames_full, fmask, K, items, out_path, size_mult=2.2, out=300):
    """items: list of (frame, [T_cam_mesh,...], label)"""
    tiles=[]
    for (t,Ts,lab) in items:
        om=fmask['object'][t]; ys,xs=np.nonzero(om)
        if len(xs)==0:
            p=K@Ts[0][:3,3]; cx,cy=p[0]/p[2],p[1]/p[2]; sz=200
        else: cx,cy=xs.mean(),ys.mean(); sz=max(np.ptp(xs),np.ptp(ys))*size_mult
        M,_=crop_params(cx,cy,sz,out); Kc=torch.tensor(crop_K(K,M),dtype=torch.float32,device=dev)
        im=warp(np.ascontiguousarray(frames_full[t]),M,out)
        row=[im.copy()]
        omc=warp(om.astype(np.uint8),M,out,cv2.INTER_NEAREST)
        cnt,_=cv2.findContours(omc,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        for T in Ts:
            o=rc.render(torch.tensor(T[None,:3,:3],dtype=torch.float32,device=dev),torch.tensor(T[None,:3,3],dtype=torch.float32,device=dev),Kc,out,out)
            col=(o['color'][0].cpu().numpy()*255).astype(np.uint8); m=o['hard'][0].cpu().numpy()>0
            b=im.copy(); b[m]=col[m]
            cv2.drawContours(b,cnt,-1,(0,255,0),1)
            row.append(b)
        r=np.hstack(row); cv2.putText(r,f'{t} {lab}',(5,20),0,0.6,(0,255,255),2); tiles.append(r)
    wmax=max(x.shape[1] for x in tiles)
    tiles=[np.pad(x,((0,0),(0,wmax-x.shape[1]),(0,0))) for x in tiles]
    cv2.imwrite(out_path,np.vstack(tiles))
