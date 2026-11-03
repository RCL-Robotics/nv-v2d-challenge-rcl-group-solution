exec(open('/work/scratch/common_exp.py').read())
h=np.load('/work/scratch/hyp.npz'); ks=['base','fp','base_fz','fp_fz','base_fy']
fit=CropFitter('/work/inputs/object_mesh.glb',res=192)
ids=list(range(0,747,8))
fd=make_fd(fit,ids,Tb[ids])
for k in ks:
    Rm=torch.tensor(np.nan_to_num(h[k+'/R'][ids]),dtype=torch.float32,device='cuda'); p=torch.tensor(h[k+'/p'][ids],dtype=torch.float32,device='cuda')
    with torch.no_grad():
        o=fit.render(Rm,p,fd.K); w=o['hard']*fd.M*fd.V
        d,ws=chroma_score(o['color'],fd.img,w)
    h_=dict(np.load('/work/scratch/hyp.npz')) if False else None
    globals()['C_'+k]=d.cpu().numpy()
for j,i in enumerate(ids):
    print(i,' '.join('%s:%.3f'%(k[:7],globals()['C_'+k][j]) for k in ks))
