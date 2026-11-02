import sys,torch,inspect
sys.path.insert(0,"/workspace/v2d_cari4d/lib/cari4d")
from tools import render_mhr_wild_inference as m
print(m.MHR_PARAM_DIMS)
b=torch.load("/mnt/nas/Group/hanzhou/v2d/experiments/DB_JC2s1_shared_f820_pp/episode_000000/coconet.pth",map_location="cpu",weights_only=False)
print({k:(tuple(v.shape) if hasattr(v,"shape") else type(v).__name__) for k,v in b["pr"].items()})
from lib_mhr.mhr_layer import MHRLayer
print([n for n in dir(MHRLayer) if not n.startswith("__")])
from soma.pose_inversion import PoseInversion; print("PI ok")
