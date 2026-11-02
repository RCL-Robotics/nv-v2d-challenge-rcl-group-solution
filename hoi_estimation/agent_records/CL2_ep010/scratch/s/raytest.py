import sys
exec(open('/work/scratch/s/evaluate.py').read().split("phases=[")[0])
A=np.load('runs/blk_nofloor.npz')['T_world_mesh'].copy()
Vs=mesh.vertices[::5]
def hmin(Tw): return ((Vs@Tw[:3,:3].T+Tw[:3,3])@plane[:3]+plane[3]).min()
B=A.copy()
for i in range(261,597):
    Tc=Tcw@A[i]; p=Tc[:3,:3]@c+Tc[:3,3]
    # find scale s so that hmin = max(hmin, -0.001) (only lift penetrating)
    if hmin(A[i])>=-0.001: continue
    lo,hi=0.9,1.0
    for _ in range(30):
        s=(lo+hi)/2; T=Tc.copy(); T[:3,3]=p*s-T[:3,:3]@c
        if hmin(Twc@T)<-0.001: hi=s
        else: lo=s
    T=Tc.copy(); T[:3,3]=p*lo-T[:3,:3]@c; B[i]=Twc@T
I0=ious(A); I1=ious(B)
for a in range(440,520,10): print(a, I0[a:a+10].mean().round(3), I1[a:a+10].mean().round(3))
print('M', I0[261:597].mean(), I1[261:597].mean())
