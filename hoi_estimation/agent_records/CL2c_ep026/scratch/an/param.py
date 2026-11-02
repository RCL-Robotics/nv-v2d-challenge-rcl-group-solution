import numpy as np
def rodr(axis,ang):
    a=axis/np.linalg.norm(axis); K=np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
    return np.eye(3)+np.sin(ang)*K+(1-np.cos(ang))*K@K
class FloorParam:
    """poses = base pose rotated by yaw about floor normal through object origin and translated in floor plane (+ optional height)"""
    def __init__(self, n):
        self.n=n/np.linalg.norm(n)
        e1=np.cross(self.n,[0,0,1.]); self.e1=e1/np.linalg.norm(e1); self.e2=np.cross(self.n,self.e1)
    def apply(self, Tw, yaw, a, b, h=0.0):
        T=Tw.copy(); Rz=rodr(self.n,yaw)
        T[:3,:3]=Rz@Tw[:3,:3]; T[:3,3]=Tw[:3,3]+a*self.e1+b*self.e2+h*self.n
        return T
    def yaw_of(self, Tw):
        z=Tw[:3,2]; return np.arctan2(z@self.e2, z@self.e1)
