from pano import *
T=np.load('/work/scratch/fitA.npz')['T_world_mesh']
def wyaw(Tw): x=Tw[:3,0]; return np.degrees(np.arctan2(x[2],x[0]))
AX=AXC
lug=[AX+np.array([0.114*np.cos(np.radians(a)),0.075,0.114*np.sin(np.radians(a))]) for a in (-60,120)]
for i in [int(x) for x in sys.argv[1].split(',')]:
    Tc=np.linalg.inv(Twc)@T[i]; y0=wyaw(T[i])
    res=[]
    for ay in range(0,360,15):
        d=np.radians(-(ay-y0))  # approx: rotation about mesh y changes world yaw by -angle? check sign
        Tn=Tc@yawrot(d)
        if abs(((wyaw(Twc@Tn)-ay+180)%360)-180)>5: Tn=Tc@yawrot(-d)
        c=Tn[:3,:3]@AX+Tn[:3,3]; uc=K[0,0]*c[0]/c[2]
        fr=[]
        for L in lug:
            p=Tn[:3,:3]@L+Tn[:3,3]; fr.append((K[0,0]*p[0]/p[2]-uc)/(K[0,0]*0.114/c[2]))
        res.append(f'{ay}:{fr[0]:+.1f}/{fr[1]:+.1f}')
    print(i,' '.join(res))
