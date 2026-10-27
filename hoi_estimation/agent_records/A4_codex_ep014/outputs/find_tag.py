from inspect_baseline import *
uv=torch.tensor(mesh.visual.uv,dtype=torch.float32,device='cuda')[None].contiguous();tex=torch.tensor(np.array(mesh.visual.material.baseColorTexture)[::-1].copy()/255,dtype=torch.float32,device='cuda')[None].contiguous()
tri=tri[(v[tri.long(),2].max(dim=1).values<.02)].contiguous()
xyz=v*torch.tensor([1,-1,-1],device='cuda')+torch.tensor([0,0,1.2],device='cuda');x,y,z=xyz.unbind(-1);clip=torch.stack((2*1100/1024*x,2*1100/1024*y,1.002002*z-.020020,z),-1)[None].contiguous();rast,_=dr.rasterize(ctx,clip,tri,(1024,1024));tc,_=dr.interpolate(uv,rast,tri);color=dr.texture(tex,tc,filter_mode='linear')[0].cpu().numpy();pos,_=dr.interpolate(v[None],rast,tri);pos=pos[0].cpu().numpy();sel=(color.min(2)>.38)&((color.max(2)-color.min(2))<.25)&(pos[:,:,2]>-.12)&(pos[:,:,2]<-.06)&(np.linalg.norm(pos[:,:,:2],axis=-1)>.1)&(np.linalg.norm(pos[:,:,:2],axis=-1)<.3)
n,lab,stats,centers=cv2.connectedComponentsWithStats(sel.astype(np.uint8));
for j in np.argsort(stats[1:,4])[::-1][:15]+1:
 print('area',stats[j,4],'position',pos[lab==j].mean(0),'range',np.ptp(pos[lab==j],axis=0))
Image.fromarray((color*255).astype('uint8')).save('/work/scratch/mesh_under.jpg')
