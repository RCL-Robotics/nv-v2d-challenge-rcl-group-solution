from inspect_baseline import *
ids=[275,282,288,295,302,590,600,604,605,608,612,615,620,630,650]
c=cv2.VideoCapture(p+'video.mp4');out=Image.new('RGB',(5*440,3*480),'white');d=ImageDraw.Draw(out)
with h5py.File(p+'masks.h5') as h:
 for j,i in enumerate(ids):
  c.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=c.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);gt=h['object'][i];ys,xs=np.where(gt);x=int((xs.min()+xs.max())/2);y=int((ys.min()+ys.max())/2);side=420
  raw=im[max(0,y-side//2):min(1152,y+side//2),max(0,x-side//2):min(1536,x+side//2)];out.paste(Image.fromarray(raw).resize((220,440)),(j%5*440,j//5*480))
  m=cv2.resize(render(T[i]).astype('uint8'),(1536,1152),interpolation=cv2.INTER_NEAREST).astype(bool);im[m]=(.6*im[m]+.4*np.array([255,30,30])).astype('uint8');crop=im[max(0,y-side//2):min(1152,y+side//2),max(0,x-side//2):min(1536,x+side//2)];out.paste(Image.fromarray(crop).resize((220,440)),(j%5*440+220,j//5*480));d.text((j%5*440,j//5*480+440),str(i),fill='black')
out.save('/work/scratch/transitions.jpg')
