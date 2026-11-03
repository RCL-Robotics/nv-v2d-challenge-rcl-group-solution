from common import *
import json
pl=json.load(open(INP+'/ground_plane.json'))['plane']; PN=np.array(pl[:3]); PD=pl[3]
_V=load_mesh().vertices
LEGS=np.array([[-0.1949,-0.2123,0.4867],[0.2395,0.2168,0.5149],[0.2347,-0.2154,0.5149],[-0.1983,0.2206,0.4809]])
_c=LEGS.mean(0); LEGN=np.linalg.svd(LEGS-_c)[2][2]; LEGN=-LEGN if LEGN[2]>0 else LEGN  # 'up' direction in mesh coords (mesh +z is down)
def floor_info(Tcam):
    Tw_=Twc@Tcam
    L=(Tw_[:3,:3]@LEGS.T).T+Tw_[:3,3]; s=L@PN+PD
    up=Tw_[:3,:3].T@PN; tilt=np.degrees(np.arccos(np.clip(abs(up@LEGN),0,1)))
    return s, tilt
