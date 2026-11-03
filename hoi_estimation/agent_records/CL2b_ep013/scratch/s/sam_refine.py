import os, numpy as np, torch, cv2, h5py, sys, time, json
from sam2.sam2_image_predictor import SAM2ImagePredictor
import hydra
from hydra import compose
from omegaconf import OmegaConf
from hydra.utils import instantiate
def build():
    d='/weights/sam2'
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(config_dir=d, job_name="sam2"):
        cfg=compose(config_name='sam2.1_hiera_l.yaml')
    OmegaConf.resolve(cfg); model=instantiate(cfg.model,_recursive_=True)
    model.load_state_dict(torch.load(f'{d}/sam2.1_hiera_large.pt',map_location='cpu')['model'],strict=True)
    return SAM2ImagePredictor(model.cuda().eval())
def refine(pred, im, om, z, f=820.):
    ys,xs=np.nonzero(om)
    if len(xs)<20: return om, -1
    cx,cy=np.median(xs),np.median(ys); s=max(np.ptp(xs),np.ptp(ys),20)
    with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):
        pred.set_image(im[...,::-1].copy())
        box=np.array([cx-s*1.2,cy-s*1.2,cx+s*1.2,cy+s*1.2])
        masks,scores,_=pred.predict(point_coords=np.array([[cx,cy]]),point_labels=np.array([1]),box=box[None],multimask_output=True)
    k2=(f/z)**2; amin,amax=0.05*k2,0.22*k2
    best=-1; bm=om
    for k in np.argsort(-scores):
        m=masks[k]>0; a=m.sum()
        if a<amin or a>amax: continue
        if (m&om).sum()<0.7*om.sum(): continue
        # compactness: mask area vs its bbox area
        yy,xx=np.nonzero(m); 
        if a/((np.ptp(xx)+1)*(np.ptp(yy)+1))<0.35: continue
        best=k; bm=m|om; break
    return bm, best
if __name__=='__main__':
    pred=build()
    fr=np.load('/work/scratch/frames_full.npy',mmap_mode='r'); f=h5py.File('/work/inputs/masks.h5')
    rec=np.load('/work/inputs/reconstruction.npz'); c=json.load(open('/work/inputs/camera.json'))
    Tcb=np.linalg.inv(np.array(c['T_world_cam']))[None]@rec['T_world_mesh'].astype(float)
    frames=list(range(132,312))+list(range(702,817))
    out={}; t0=time.time()
    for t in frames:
        m,k=refine(pred,np.ascontiguousarray(fr[t]),f['object'][t],Tcb[t,2,3]); out[t]=(m,k)
    print(time.time()-t0)
    np.save('/work/scratch/sam_ref.npy',{t:np.packbits(v[0]) for t,v in out.items()},allow_pickle=True)
    print(''.join('x' if out[t][1]<0 else str(out[t][1]) for t in frames))
