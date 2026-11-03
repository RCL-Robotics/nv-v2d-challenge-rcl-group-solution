import numpy as np
d=np.load('/work/scratch/masks_half.npz'); obj=d['obj']; per=d['per']
F=len(obj)
def mscore(k):
    m=np.zeros(F)
    for i in range(F-k):
        v=~(per[i]|per[i+k]); a=obj[i]&v; b=obj[i+k]&v
        m[i]=1-(a&b).sum()/max(1,(a|b).sum())
    m[F-k:]=m[F-k-1]
    return m
m3=mscore(3)
np.save('/work/scratch/m3.npy',m3)
thr=0.04
# frame i static if all pair scores touching it (i-3..i) below thr
st=np.array([m3[max(0,i-3):i+1].max()<thr for i in range(F)])
segs=[];i=0
while i<F:
    j=i
    while j<F and st[j]==st[i]: j+=1
    segs.append((i,j,st[i])); i=j
print([s for s in segs])
for i in range(F):
    if 110<i<130 or 305<i<380 or 470<i<515 or 570<i<595: print(i,'%.3f'%m3[i],end='; ')
