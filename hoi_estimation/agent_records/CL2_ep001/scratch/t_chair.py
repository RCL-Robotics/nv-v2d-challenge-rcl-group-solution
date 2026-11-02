import sys; sys.path.insert(0,'/work/outputs')
from lib import *
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import glob
cfg='/weights/sam2/sam2.1_hiera_l.yaml'
model=build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml','/weights/sam2/sam2.1_hiera_large.pt',device='cuda')
pred=SAM2ImagePredictor(model)
cap=cv2.VideoCapture('/work/inputs/video.mp4')
ms=load_masks('/work/inputs',scale=1,idx=[0,700])
out=[]
for j,i in enumerate([0,700]):
    cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,im=cap.read()
    pred.set_image(im[:,:,::-1].copy())
    m,s,_=pred.predict(box=np.array([625,650,835,1005]),multimask_output=False)
    m=m[0]>0; m&=~ms['object'][j]
    print(i,s,m.sum()); out.append(m)
    v=im.copy(); v[m]=(0.5*v[m]+[0,0,127]).astype(np.uint8); cv2.imwrite(f'/work/scratch/viz/chair{i}.png',v[600:1050,550:900])
print('iou',(out[0]&out[1]).sum()/(out[0]|out[1]).sum())
ch=out[0]|out[1]
np.save('/work/scratch/cache/chair.npy',ch)
