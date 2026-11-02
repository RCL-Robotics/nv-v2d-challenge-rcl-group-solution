exec(open('setup.py').read())
a,b=0,900
idx=np.arange(a,b+1)
t=time.time()
Ts,Lb,L0,sh=shift_search(fit,Tb[idx],idx)
print('time',time.time()-t)
np.savez('shift.npz',T=Ts,Lb=Lb,L0=L0,sh=sh)
for s in range(0,len(idx),10): print(idx[s], 'L0 %.2f Lb %.2f'%(L0[s:s+10].mean(),Lb[s:s+10].mean()), np.round(np.median(sh[s:s+10],0),1), np.round(np.abs(sh[s:s+10]).max(0),1))
