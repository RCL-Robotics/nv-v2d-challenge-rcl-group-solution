from diagnose import *
root='/work/inputs';cam=json.load(open(root+'/camera.json'));K=np.array(cam['K']);T=np.linalg.inv(cam['T_world_cam'])@np.load(root+'/reconstruction.npz')['T_world_mesh'];h=h5py.File(root+'/masks.h5');ren=Renderer();cap=cv2.VideoCapture(root+'/video.mp4');out=Image.new('RGB',(6*300,500));d=ImageDraw.Draw(out)
for j,f in enumerate([0,130,150,270,300,400]):
 cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,im=cap.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);col,mask=ren.render(T[f],K,1152,1536);obj=h['object'][f];ys,xs=np.where(obj);cx=int((xs.min()+xs.max())/2);cy=int((ys.min()+ys.max())/2)
 cv2.drawContours(im,cv2.findContours(obj.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(0,255,0),1);cv2.drawContours(im,cv2.findContours(mask[0].astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(255,0,0),1)
 crop=Image.fromarray(im[cy-115:cy+115,cx-65:cx+65]).resize((280,460));out.paste(crop,(j*300,30));d.text((j*300+5,5),f'{f} green observed / red model',fill='white')
out.save('/work/outputs/edge_review.jpg')
