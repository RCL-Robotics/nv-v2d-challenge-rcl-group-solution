import time,pickle; t=time.time()
from core import *
fb=FrameBank('/work/inputs'); print('feat',time.time()-t,flush=True)
np.save('../fb_dtm.npy',fb.dtm); np.save('../fb_dto.npy',fb.dto); np.save('../fb_occ.npy',fb.occ)
pickle.dump(dict(mpts=fb.mpts,opts=fb.opts,area=fb.area,oarea=fb.oarea),open('../fb_pts.pkl','wb'))
print(time.time()-t)
