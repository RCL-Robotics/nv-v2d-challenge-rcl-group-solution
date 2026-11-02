exec(open('/work/scratch/common_exp.py').read())
h=np.load('/work/scratch/hyp2.npz'); hid=list(h['ids'])
fit=CropFitter('/work/inputs/object_mesh.glb',res=192)
ids=[int(a) for a in sys.argv[1].split(',')]
ks=['b0','b90','b180','b270','f0','f180']
fd=make_fd(fit,ids,Tb[ids])
res={}
for k in ks:
    j=[hid.index(i) for i in ids]
    Rm=torch.tensor(np.nan_to_num(h[k+'/R'][j]),dtype=torch.float32,device='cuda'); p=torch.tensor(h[k+'/p'][j],dtype=torch.float32,device='cuda')
    with torch.no_grad():
        o=fit.render(Rm,p,fd.K); L=losses(fit,Rm,p,fd,w_col=1)
        dr_,dv=app_score(o['color'],o['hard'],fd.img,fd.M,fd.V)
    res[k]=(L['iou'].cpu().numpy(),dr_.cpu().numpy(),dv.cpu().numpy())
for jj,i in enumerate(ids):
    print(i,' '.join('%s: %.2f %.3f %.3f |'%(k,res[k][0][jj],res[k][1][jj],res[k][2][jj]) for k in ks))
