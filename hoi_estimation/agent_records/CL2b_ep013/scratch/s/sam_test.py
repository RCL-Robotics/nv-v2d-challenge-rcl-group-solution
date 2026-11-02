import os, numpy as np, torch, cv2, h5py, sys, time
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2
import hydra
from hydra import compose
from omegaconf import OmegaConf
from hydra.utils import instantiate
d='/weights/sam2'
hydra.core.global_hydra.GlobalHydra.instance().clear()
with hydra.initialize_config_dir(config_dir=d, job_name="sam2"):
    cfg=compose(config_name='sam2.1_hiera_l.yaml')
OmegaConf.resolve(cfg); model=instantiate(cfg.model,_recursive_=True)
model.load_state_dict(torch.load(f'{d}/sam2.1_hiera_large.pt',map_location='cpu')['model'],strict=True)
model=model.cuda().eval(); pred=SAM2ImagePredictor(model)
fr=np.load('/work/scratch/frames_full.npy',mmap_mode='r'); f=h5py.File('/work/inputs/masks.h5')
idx=[int(x) for x in sys.argv[1].split(',')]
tiles=[]; t0=time.time()
for t in idx:
    im=np.ascontiguousarray(fr[t]); om=f['object'][t]; ys,xs=np.nonzero(om)
    cx,cy=np.median(xs),np.median(ys); s=max(np.ptp(xs),np.ptp(ys))
    with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):
        pred.set_image(im[...,::-1].copy())
        box=np.array([cx-s*1.2,cy-s*1.2,cx+s*1.2,cy+s*1.2])
        masks,scores,_=pred.predict(point_coords=np.array([[cx,cy]]),point_labels=np.array([1]),box=box[None],multimask_output=True)
    k=int(np.argmax(scores)); m=masks[k]>0
    x0,y0=int(cx-200),int(cy-200); crop=im[max(y0,0):y0+400,max(x0,0):x0+400].copy()
    mm=m[max(y0,0):y0+400,max(x0,0):x0+400]
    crop[mm]=(0.5*crop[mm]+[0,0,127]).astype(np.uint8)
    cnt,_=cv2.findContours(om[max(y0,0):y0+400,max(x0,0):x0+400].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(crop,cnt,-1,(0,255,0),1)
    cv2.putText(crop,f'{t} {scores.round(2)}',(5,25),0,0.6,(0,255,255),2); tiles.append(cv2.resize(crop,(300,300)))
print(time.time()-t0, torch.cuda.max_memory_allocated()/1e9)
while len(tiles)%4: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(sys.argv[2],np.vstack([np.hstack(tiles[k:k+4]) for k in range(0,len(tiles),4)]))
