# build cached frame data (incl. MoGe crops) for fast experiments
import sys, pickle; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2
from stages import *
A=pickle.load(open('/work/scratch/stageA.pkl','rb')); wins=A['wins']
datas=pickle.load(open('/work/scratch/datas.pkl','rb'))
raw=np.load(MOGE_RAW,mmap_mode='r')
for i,(d,win) in enumerate(zip(datas,wins)):
    x0,y0,side=win; sc=RES/side; Aff=np.array([[sc,0,-x0*sc],[0,sc,-y0*sc]])
    d['Zm']=cv2.warpAffine(raw[i].astype(np.float32)/1000.,Aff,(RES,RES),flags=cv2.INTER_NEAREST)
pickle.dump(datas,open('/work/scratch/datas.pkl','wb'))
