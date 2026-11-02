exec(open('/work/scratch/common_exp.py').read())
h=np.load('/work/scratch/hyp2.npz'); hid=list(h['ids'])
fit=CropFitter('/work/inputs/object_mesh.glb',res=192)
ks=['b0','b90','b180','b270','f0','f90','f180','f270']
allS={k:[] for k in ks}
for c0 in range(0,len(hid),140):
    ids=hid[c0:c0+140]; fd=make_fd(fit,ids,Tb[ids]); j=list(range(c0,c0+len(ids)))
    for k in ks:
        Rm=torch.tensor(np.nan_to_num(h[k+'/R'][j]),dtype=torch.float32,device='cuda'); p=torch.tensor(h[k+'/p'][j],dtype=torch.float32,device='cuda')
        with torch.no_grad():
            o=fit.render(Rm,p,fd.K); L=losses(fit,Rm,p,fd,w_col=0)
            dr_,dv=app_score(o['color'],o['hard'],fd.img,fd.M,fd.V)
        allS[k].append(np.stack([L['iou'].cpu().numpy(),L['dt'].cpu().numpy(),dr_.cpu().numpy(),dv.cpu().numpy()],1))
np.savez('/work/scratch/hyp2_scores.npz',ids=hid,**{k:np.concatenate(v) for k,v in allS.items()})
