import sys; sys.path.insert(0,'/work/scratch/s')
import numpy as np
from scipy.spatial.transform import Rotation as Rot
def rot_ang(A,B): return np.degrees(Rot.from_matrix(A[:3,:3].T@B[:3,:3]).magnitude())
def viterbi(cands, unary, Tstart, Tend, w_u=4.0, s_r=10.0, s_t=0.05):
    """cands: list over frames of list of 4x4; unary: list of arrays"""
    n=len(cands)
    def pair(A,B): return rot_ang(A,B)/s_r+np.linalg.norm(A[:3,3]-B[:3,3])/s_t
    cost=[np.array([pair(Tstart,c) for c in cands[0]])+w_u*unary[0]]; back=[]
    for t in range(1,n):
        P=np.array([[pair(a,b) for a in cands[t-1]] for b in cands[t]])  # (cur,prev)
        tot=P+cost[-1][None]; bi=tot.argmin(1); back.append(bi)
        cost.append(tot[np.arange(len(bi)),bi]+w_u*unary[t])
    last=cost[-1]+np.array([pair(c,Tend) for c in cands[-1]])
    k=int(last.argmin()); path=[k]
    for t in range(n-2,-1,-1): k=int(back[t][k]); path.append(k)
    return path[::-1]
