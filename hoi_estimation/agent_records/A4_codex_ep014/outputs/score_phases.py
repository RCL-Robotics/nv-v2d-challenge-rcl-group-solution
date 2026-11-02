from inspect_baseline import *
from scipy.spatial.transform import Rotation
with h5py.File(p+'masks.h5') as h:
 for i in [0,50,200,250,350,400,500,550,600,625,650,750,800,850,900,992]:
  gt=h['object'][i,::2,::2];person=h['person'][i,::2,::2];valid=~person|gt;scores=[]
  for k in range(5):
   t=T[i].copy();t[:3,:3]=t[:3,:3]@Rotation.from_euler('z',72*k,degrees=True).as_matrix();m=render(t);scores.append(round(((m&gt)&valid).sum()/((m|gt)&valid).sum(),3))
  print(i,scores)
