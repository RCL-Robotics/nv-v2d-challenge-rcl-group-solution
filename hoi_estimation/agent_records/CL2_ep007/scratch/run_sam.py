import numpy as np, torch, sys, time
from v2d.sam2.lib.sam2_utils import build_sam2_video_predictor_low_mem
pred = build_sam2_video_predictor_low_mem(config_file="configs/sam2.1/sam2.1_hiera_l.yaml", ckpt_path="/weights/sam2/sam2.1_hiera_large.pt")
F=896; M=np.zeros((F,400,480),bool); t=time.time()
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    st = pred.init_state("/work/scratch/crops", offload_video_to_cpu=True, offload_state_to_cpu=True)
    pred.add_new_points_or_box(st, frame_idx=0, obj_id=1, box=[160,228,242,282])
    for rev in (False,):
        for fi, ids, logits in pred.propagate_in_video(st, reverse=rev):
            M[fi]=(logits[0,0]>0).cpu().numpy()
print('time',time.time()-t, 'maxmem', torch.cuda.max_memory_allocated()/1e9)
np.savez_compressed('/work/scratch/sam/iron_v1.npz',m=M)
