import numpy as np,sys; sys.path.insert(0,'/work/outputs')
from lib import rot_angle
h=np.load('/work/scratch/hyp.npz'); ks=['base','fp','base_fz','fp_fz','base_fy']
S=np.stack([h[f'{k}/iou']+0.3*h[f'{k}/ncc']-2*h[f'{k}/dt'] for k in ks],1)
R=np.stack([h[f'{k}/R'] for k in ks],1)
best=S.argmax(1)
for i in range(0,747,8):
    print(i, ' '.join('%s:%.2f/%.2f'%(k[:7],h[f'{k}/iou'][i],h[f'{k}/ncc'][i]) for k in ks), 'best',ks[best[i]], 'ang(base,fp)=%.0f ang(best,base)=%.0f'%(rot_angle(R[i,0],R[i,1]),rot_angle(R[i,best[i]],R[i,0])))
