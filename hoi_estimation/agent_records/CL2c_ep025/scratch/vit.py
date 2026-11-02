import numpy as np
def rot_angle_matrix(Ra, Rb):
    # Ra (n,3,3), Rb (m,3,3) -> (n,m) degrees
    tr=np.einsum('aij,bij->ab',Ra,Rb)
    return np.degrees(np.arccos(np.clip((tr-1)/2,-1,1)))
def viterbi(cands, unary, lam_r=0.004, lam_t=1.0, static=None, cap=None):
    """cands: list per frame of (R (n,3,3), t (n,3)); unary list of (n,) arrays."""
    F=len(cands); back=[]; acc=unary[0].copy()
    for i in range(1,F):
        Ra,ta=cands[i-1]; Rb,tb=cands[i]
        A=rot_angle_matrix(Ra,Rb); 
        Td=np.linalg.norm(ta[:,None]-tb[None],axis=-1)
        lr=lam_r*(static[i] if static is not None else 1)
        C=lr*A+lam_t*Td
        tot=acc[:,None]+C
        b=np.argmin(tot,0); back.append(b)
        acc=tot[b,np.arange(len(b))]+unary[i]
    path=[int(np.argmin(acc))]
    for b in back[::-1]: path.append(int(b[path[-1]]))
    return path[::-1]
