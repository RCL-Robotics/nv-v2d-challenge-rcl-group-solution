import sys,torch,inspect
sys.path.insert(0,"/workspace/v2d_cari4d/lib/cari4d")
from lib_mhr.mhr_layer import MHRLayer
src=inspect.getsource(MHRLayer.mhr_forward); print(src[:3500])
