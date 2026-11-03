"""Image-evidence gates and temporal selection of refinement proposals."""
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d,median_filter

def angle(A,B):
    return np.arccos(np.clip((np.einsum('...ij,...ij->...',A,B)-1)/2,-1,1))

def choose(cand,K):
    R=cand['R'];t=cand['t'];l=cand['scores'];stats=cand['stats'];F,N=l.shape
    ang=angle(R,R[:,0,None]);axis=np.arccos(np.clip(np.einsum('fni,fi->fn',R[:,:,:,2],R[:,0,:,2]),-1,1))
    uv=t[...,:2]/t[...,2:]*np.array([K[0,0],K[1,1]])+K[:2,2];shift=np.linalg.norm(uv-uv[:,0,None],axis=-1)
    # Cost for departing from the calibrated reconstruction, especially for a
    # poorly constrained roll around the bottle axis. Pixel evidence must pay it.
    unary=l+.028*np.minimum(ang**2,3)+.006*(shift/3)**2
    # Only serious silhouette or color mismatches justify trying a replacement.
    gain=l[:,0,None]-l
    valid=(gain>.075)&(stats[:,:,0]>.67)&(stats[:,:,0]>stats[:,0,None,0]-.015)
    # Cap flip demands a very strong improvement; apparently symmetric body
    # silhouettes do not identify which end is the cap.
    valid&=(axis<np.deg2rad(80))|((gain>.22)&(stats[:,:,0]>.78))
    valid[:,0]=True
    unary=np.where(valid,unary,1e4)
    dp=unary[0].copy();back=np.zeros((F,N),dtype=int)
    for i in range(1,F):
        rot=angle(R[i-1,:,None],R[i,None]);ax=np.arccos(np.clip(R[i-1,:,:,2]@R[i,:,:,2].T,-1,1))
        # Smooth orientation without assuming constant angular velocity.
        temporal=.018*(rot/.25)**2+.05*(ax/.20)**2
        du=uv[i,None]-uv[i-1,:,None];baseline_du=uv[i,0]-uv[i-1,0]
        temporal+=.003*np.minimum(np.sum((du-baseline_du)**2,axis=-1)/9,20)
        cost=dp[:,None]+temporal;back[i]=np.argmin(cost,axis=0);dp=unary[i]+np.min(cost,axis=0)
    chosen=np.zeros(F,dtype=int);chosen[-1]=np.argmin(dp)
    for i in range(F-1,0,-1):chosen[i-1]=back[i,chosen[i]]
    return chosen,dict(gain=gain,angle=ang,axis=axis,shift=shift,unary=unary)

if __name__=='__main__':
 import json
 c=np.load('/work/outputs/candidates.npz');K=np.array(json.load(open('/work/inputs/camera.json'))['K']);ch,meta=choose(c,K);ii=np.arange(len(ch));np.savez_compressed('/work/outputs/selected.npz',chosen=ch,R=c['R'][ii,ch],t=c['t'][ii,ch],**{k:v[ii,ch] for k,v in meta.items()});
 sel=ch>0;print('changed',sel.sum());print('iou',np.mean(c['stats'][:,0,0]),np.mean(c['stats'][ii,ch,0]));print('selected angles',np.percentile(meta['angle'][ii,ch][sel]*180/np.pi,[0,25,50,75,90,100]));
 changes=np.where(np.diff(np.r_[False,sel,False]))[0].reshape(-1,2);print('intervals',[(a,b-1) for a,b in changes]);
 for f in range(0,len(ch),25):print(f,ch[f],round(c['scores'][f,0],3),round(c['scores'][f,ch[f]],3),round(np.degrees(meta['axis'][f,ch[f]]),1),round(meta['shift'][f,ch[f]],1))
