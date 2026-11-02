import sys, time, pickle; sys.path.insert(0,'/work/outputs')
import numpy as np, torch
from stages import *
t0=time.time()
ep=Episode('/work/inputs'); ep.load_frames(); print('load',time.time()-t0)
FR=ep.fast_renderer()
ids=list(range(ep.F)); inits=[ep.init_hypotheses(i) for i in ids]
init_sc=score_frames(ep,FR,ids,inits)
t1=time.time(); res=fit_frames(ep,FR,ids,inits); torch.cuda.synchronize(); print('fit',time.time()-t1)
pickle.dump(dict(res=res,init_sc=init_sc,wins=ep.wins,mask_area=ep.mask_area,mask_cent=ep.mask_cent),open('/work/scratch/stageA.pkl','wb'))
np.savez_compressed('/work/scratch/masks_small.npz',o=ep.mask_small_o,p=ep.mask_small_p)
pickle.dump(ep.datas,open('/work/scratch/datas.pkl','wb'))
