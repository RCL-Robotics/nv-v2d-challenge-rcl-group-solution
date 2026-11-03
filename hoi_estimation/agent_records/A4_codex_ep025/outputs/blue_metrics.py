import numpy as np,torch

@torch.no_grad()
def compute_blue(c,renderer):
 R=c['R'];t=c['t'];F,N=R.shape[:2];score=np.zeros((F,N))
 for j in range(0,F,4):
  n=min(4,F-j);m,at,col=renderer.render(R[j:j+n].reshape(-1,3,3),t[j:j+n].reshape(-1,3),np.repeat(c['boxes'][j:j+n],N,axis=0),texture=True,antialias=True)
  br=torch.sigmoid((col[...,2]-1.16*col[...,1]-.015)*30)*torch.sigmoid((col[...,2]-1.4*col[...,0])*30)*m
  ob=torch.as_tensor(np.repeat(c['blue'][j:j+n],N,axis=0),device='cuda',dtype=torch.float32);v=torch.as_tensor(np.repeat(c['valid'][j:j+n],N,axis=0),device='cuda',dtype=torch.float32)
  val=(br*ob).sum((1,2))/(br*v+ob-br*ob).sum((1,2)).clamp(min=20);score[j:j+n]=val.cpu().numpy().reshape(n,N)
 return score
