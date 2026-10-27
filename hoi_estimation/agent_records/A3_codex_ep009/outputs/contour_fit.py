"""Occlusion-aware contour translation fitting for diagnostic experiments."""
import cv2,numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize

def edge_points(mask):
    edge=(mask>0).astype(np.uint8)
    edge-=cv2.erode(edge,np.ones((3,3),np.uint8))
    ys,xs=np.where(edge)
    return np.stack([xs,ys],1).astype(np.float32),cv2.distanceTransform(1-edge,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)

def sample(a,p):
    return cv2.remap(a,p[:,0,None].astype(np.float32),p[:,1,None].astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=10)[:,0]

def prepare(obs,person,rgb=None):
    ys,xs=np.where(obs);hull=np.zeros_like(obs,dtype=np.uint8)
    if len(xs):cv2.fillConvexPoly(hull,cv2.convexHull(np.stack([xs,ys],1)),1)
    # Person SAM masks sometimes include the held bowl and the torso behind it.
    # Within the crop, skin coloured person pixels identify hand occlusion more reliably.
    if rgb is not None:
        r,g,b=np.moveaxis(rgb.astype(np.float32),-1,0)
        skin=(r>45)&(g>30)&(b>15)&(r>1.08*g)&(r<1.9*g)&(g>1.05*b)&(r-b>10)
        human=(person.astype(bool)&skin&~obs.astype(bool)).astype(np.uint8)
    else:
        human=(person.astype(bool)&~obs.astype(bool)).astype(np.uint8)
    valid=1-cv2.dilate(human,np.ones((5,5),np.uint8))
    po,dt=edge_points(hull)
    po=po[sample(valid.astype(np.float32),po)>.5]
    return hull,valid.astype(np.float32),po,dt

def fit_contour(render,prepared,limit=6):
    target,valid,po,dto=prepared
    pr,dtr=edge_points(render>.5)
    if len(po)<10 or len(pr)<10:return np.zeros(2),np.array([10.,10.])
    def score(v):
        pts=pr+np.array(v,np.float32)
        weights=sample(valid,pts)
        d1=np.minimum(sample(dto,pts),8)
        d2=np.minimum(sample(dtr,po-np.array(v,np.float32)),8)
        if len(d2)<10 or weights.sum()<10:return 10.
        # Mean distances in pixels. Clipping limits influence of SAM contour errors.
        return .5*((d1*weights).sum()/max(weights.sum(),1)+d2.mean())
    # Bounded 2D grid followed by local subpixel search avoids Powell's flat starts.
    grid=np.array([[x,y] for y in [-4,-2,0,2,4] for x in [-4,-2,0,2,4]])
    costs=np.array([score(v) for v in grid]);initial=grid[np.argmin(costs)].astype(float)
    fit=minimize(score,initial,method='Powell',bounds=[(-limit,limit),(-limit,limit)],options={'xtol':.03,'ftol':1e-5,'maxiter':15})
    v=fit.x if fit.fun<min(costs) else initial
    return v,np.array([score([0,0]),score(v)])
