import numpy as np, h5py
f=h5py.File('/work/inputs/masks.h5')
O=f['object']; P=f['person']
prev=None; ref0=O[0]; refE=O[700]
for i in list(range(140,180,2))+list(range(590,640,2)):
    o=O[i]; p=P[i]
    vis=~p
    def iou(a,b): 
        a=a&vis; b=b&vis; return (a&b).sum()/max((a|b).sum(),1)
    ys,xs=np.nonzero(o)
    print(i,'area',o.sum(),'iou w/ f0 %.3f w/ f700 %.3f'%(iou(o,ref0),iou(o,refE)),'bottom y %d cx %.1f'%(ys.max(),xs.mean()))
