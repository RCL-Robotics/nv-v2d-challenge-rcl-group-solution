import sys; sys.path.insert(0,'/work/scratch')
from bfit import *
from fit import dark_prob
m,tex=load_mesh(); K,Twc,Kp=load_cam()
R=Renderer(m,tex,CROP,CROP); bf=BatchFitter(R)
d=np.load('crops.npz'); Ks=d['Ks']; tg=d['tg'].astype(np.float32); wt=d['wt'].astype(np.float32)
IM=np.load('imcrops.npy'); DP=np.stack([dark_prob(im) for im in IM]).astype(np.float32)
roi=np.stack([cv2.dilate(t.astype(np.uint8),np.ones((21,21))) for t in tg]).astype(np.float32)*wt
def evaluate(T):
    Ld=[];Ls=[];IoU=[]
    for s in range(0,700,50):
        sl=slice(s,s+50)
        _,pd,mm=bf.fit(T[sl],Ks[sl],DP[sl],roi[sl],iters=0)
        _,ps,_=bf.fit(T[sl],Ks[sl],tg[sl],wt[sl],iters=0)
        Ld.append(pd);Ls.append(ps)
        a=(mm>0.5)&(wt[sl]>0); b=(tg[sl]>0)&(wt[sl]>0)
        IoU.append((a&b).sum((1,2))/np.maximum((a|b).sum((1,2)),1))
    return np.concatenate(Ld),np.concatenate(Ls),np.concatenate(IoU)
if __name__=='__main__':
    r=np.load(INP+'/reconstruction.npz'); Tw=r['T_world_mesh'].astype(np.float64)
    Tc=np.linalg.inv(Twc)[None]@Tw
    res={'base':evaluate(Tc),'fitsam':evaluate(np.load('fit_free.npz')['T']),'fitdark':evaluate(np.load('fit_dark.npz')['Tb'])}
    np.savez('evalq.npz',**{k:np.stack(v) for k,v in res.items()})
    for s in range(0,700,50):
        sl=slice(s,s+50)
        print(s,'  '.join('%s dark %.4f sam %.4f iou %.3f'%(k,v[0][sl].mean(),v[1][sl].mean(),v[2][sl].mean()) for k,v in res.items()))
