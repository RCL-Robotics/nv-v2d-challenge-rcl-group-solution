exec(open('/work/scratch/dev.py').read().split('info={}')[0])
import stages, solve
o=pickle.load(open('/work/scratch/dev_out.pkl','rb'))
data['touch']=np.load('/work/scratch/touch.npy')
table=solve.table_geometry(K,n_c,d_c); table['rim_px']=solve.TABLE_RIM_PX
M=pickle.load(open('/work/scratch/cache/sam2.pkl','rb'))
cues=stages.depth_cues(o['T_fit'],o['sole'],o['X_fit'],o['s_fit'],data,K,Twc,rec,M,solve.CROP,table,896,log)
print(cues)
T=o['T_fit']; c=np.einsum('fij,j->fi',T[:,:3,:3],cm)+T[:,:3,3]; print('fit-space median z',np.median(c[:,2]))
for k,v in cues.items():
    if isinstance(v,float): print(k,'k=%.4f'%v,'median depth %.3f'%(np.median(c[:,2])*v),'shift vs geometry %.1f cm'%(100*np.median(c[:,2])*(v-cues['table_geometry'])))
