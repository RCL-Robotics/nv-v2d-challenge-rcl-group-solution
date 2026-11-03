import sys, pickle; sys.path.insert(0,'/work/outputs')
import numpy as np
A=pickle.load(open('/work/scratch/stageA.pkl','rb')); res=A['res']
names=[f'{s}{k}' for s in ['base','fp','cc'] for k in range(4)]
F=len(res)
best=[]; 
for i in range(F):
    sc=res[i]['sc']; c=sc[:,1]+2*sc[:,3]+1.0*sc[:,2]
    o=np.argsort(c); best.append(o[0])
    # flip margin: best non-flip-equivalent candidate (axis opposite)
    T=res[i]['T']; a0=T[o[0],:3,2]
    opp=[j for j in o if T[j,:3,2]@a0<0]
    m=c[opp[0]]-c[o[0]] if opp else 9
    if i%10==0: print(i,names[o[0]],'c %.3f iou %.2f col %.3f'%(c[o[0]],1-sc[o[0],1],sc[o[0],2]),'flipmargin %.3f'%m, 'area',int(A['mask_area'][i]))
best=np.array(best); print(np.bincount(best,minlength=12))
