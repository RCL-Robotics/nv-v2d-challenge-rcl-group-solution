import numpy as np
d=np.load('/work/scratch/masks_small.npz'); o=d['o']; p=d['p']; F=len(o)
def iou_ref(ref):
    r=[]
    for i in range(F):
        valid=~p[i]  # ignore pixels occluded by person
        a=o[i]&valid; b=ref&valid
        r.append((a&b).sum()/max((a|b).sum(),1))
    return np.array(r)
i0=iou_ref(o[0]); i1=iou_ref(o[-1])
for i in range(0,F,5): print(i,'%.2f %.2f'%(i0[i],i1[i]), int(o[i].sum()), int((o[i]&p[i]).sum()))
np.savez('/work/scratch/phase_sig.npz',i0=i0,i1=i1)
