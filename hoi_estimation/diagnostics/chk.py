import os, json,numpy as np,torch,sys,glob
import soma
N=os.environ["V2D_ROOT"]; FLIP=np.diag([1.,-1,-1])
od=glob.glob(f"{N}/release/tier3_s1_test/reconstructed_data/*")[0]
z=np.load(f"{od}/soma_params.npz",allow_pickle=True); print({k:z[k].shape for k in z.files})
sl=soma.SOMALayer(identity_model_type="mhr",device="cuda")
def dec(pose,ident,scale,transl,extra={}):
    with torch.no_grad():
        return sl(torch.tensor(pose).float().view(len(pose),-1,3).cuda(),torch.tensor(ident).float().cuda(),scale_params=torch.tensor(scale).float().cuda(),transl=torch.tensor(transl).float().cuda()).vertices.cpu().numpy()@FLIP
r0=np.load(f"{N}/ref_human_f0/episode_000000.npz"); print({k:r0[k].shape for k in r0.files})
g0=dec(r0["pose"][None],r0["identity"][None],r0["scale"][None],r0["transl"][None])[0]
v0=dec(z["poses"][:1].reshape(1,-1),z["identity_coeffs"][:1],z["scale_params"][:1],z["transl"][:1])[0]
print("frame0 ours vs GT human: centroid diff cm",np.round((v0.mean(0)-g0.mean(0))*100,1),"height ours %.3f GT %.3f"%(np.ptp(v0[:,1]),np.ptp(g0[:,1])))
gp=json.load(open(f"{od}/ground_plane.json"))["plane"]; n=np.array(gp[:3]);d=gp[3]
print("lowest vertex to plane (cm): ours",round(float(min(v0@n+d))*100,1),"GT",round(float(min(g0@n+d))*100,1))
