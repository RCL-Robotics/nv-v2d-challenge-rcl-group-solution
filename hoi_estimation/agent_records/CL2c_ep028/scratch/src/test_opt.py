import time,cv2
from opt import *
from viz import panel
b=load_bank_cache(); prob=Problem(b)
fr=np.array([0,160,400,520,720,800,840,920,960])
t0=time.time()
Rs=[];ts=[];Ls=[];names=[]
Rf,tf=fp_rescaled()
for src,(RR,tt) in [('b',(Rb,tb)),('fp',(Rf,tf))]:
  for fi,Fm in enumerate(flips()):
    for tr in [0,1]:
        R0=RR[fr]@Fm
        if tr: R0=tilt_reflect(R0,tt[fr])
        R,t,L,ct=optimize(prob,fr,R0,tt[fr],tb[fr,2])
        Rs.append(R);ts.append(t);Ls.append(L);names.append(f'{src}f{fi}r{tr}')
print(time.time()-t0)
Ls=np.array(Ls); np.set_printoptions(precision=2,suppress=True,linewidth=200)
for n,l in zip(names,Ls): print(n,l)
best=Ls.argmin(0)
rows=[]
for k,i in enumerate(fr):
    j=best[k]; print(i,names[j],Ls[j,k], 'angle to base', rot_angle_deg(Rs[j][k],Rb[i]))
    rows.append(panel(i,Rt_to_T(Rs[j][k],ts[j][k]),names[j],S=300))
cv2.imwrite('../viz/opt.jpg',np.vstack(rows))
