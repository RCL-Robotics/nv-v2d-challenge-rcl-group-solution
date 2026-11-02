def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    started=time.perf_counter()
    np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    args.out.parent.mkdir(parents=True,exist_ok=True)
    camera=json.loads((args.inputs/'camera.json').read_text())
    K=np.array(camera['K'],dtype=np.float64)
    world_cam=np.array(camera['T_world_cam'],dtype=np.float64)
    with np.load(args.inputs/'reconstruction.npz') as data:
        baseline=data['T_world_mesh'].astype(np.float64)
    expected=json.loads((args.inputs/'episode.json').read_text())['frames']
    assert len(baseline)==expected
    candidates,renderer=generate(args.inputs)
    candidates=add_roll_candidates(candidates,renderer)
    chosen,meta=choose(candidates,K)
    ii=np.arange(len(chosen));changed=chosen!=0
    result=baseline.copy();cam_base=np.linalg.inv(world_cam)@baseline
    selected_R=candidates['R'][ii,chosen].copy()
    selected_t=candidates['t'][ii,chosen].copy()
    # Preserve calibrated Euclidean camera-to-centre distance in float64.
    radii=np.linalg.norm(cam_base[:,:3,3],axis=1)
    selected_t*=radii[:,None]/np.linalg.norm(selected_t,axis=1,keepdims=True)
    if changed.any():
        # Repair tiny floating-point nonorthogonality only in edited rotations.
        # Unchanged output entries retain the original baseline values exactly.
        U,_,Vh=np.linalg.svd(selected_R[changed]);rr=U@Vh
        correction=np.ones((len(rr),3));correction[:,2]=np.linalg.det(rr)
        rr=(U*correction[:,None,:])@Vh
        ct=np.tile(np.eye(4),(changed.sum(),1,1))
        ct[:,:3,:3]=rr;ct[:,:3,3]=selected_t[changed]
        result[changed]=world_cam@ct
    result=result.astype(np.float64)
    assert result.shape==(expected,4,4) and np.isfinite(result).all()
    assert np.array_equal(result[~changed],baseline[~changed])
    assert np.max(np.abs(result[:,3,:]-[0,0,0,1]))<1e-12
    distance=np.linalg.norm((np.linalg.inv(world_cam)@result)[:,:3,3],axis=1)
    assert np.max(np.abs(distance-radii))<1e-12
    assert np.max(np.abs(np.linalg.det(result[:,:3,:3])-1))<1e-5
    np.savez_compressed(args.out,T_world_mesh=result)
    base_iou=candidates['stats'][:,0,0];final_iou=candidates['stats'][ii,chosen,0]
    blue=candidates['blue_iou'];blue_visible=candidates['blue'].sum((1,2))>80
    intervals=np.where(np.diff(np.r_[False,changed,False]))[0].reshape(-1,2)
    def pct(v):return np.percentile(v,[0,25,50,75,95,100]).tolist()
    metrics=dict(frames=expected,changed_frames=int(changed.sum()),unchanged_frames=int((~changed).sum()),
        changed_intervals_inclusive=[[int(a),int(b-1)] for a,b in intervals],
        silhouette_iou_all_before=float(base_iou.mean()),silhouette_iou_all_after=float(final_iou.mean()),
        silhouette_iou_changed_before=float(base_iou[changed].mean()),silhouette_iou_changed_after=float(final_iou[changed].mean()),
        visible_blue_iou_before=float(blue[blue_visible,0].mean()),visible_blue_iou_after=float(blue[ii,chosen][blue_visible].mean()),
        rotation_change_degrees_percentiles=pct(np.degrees(meta['angle'][ii,chosen][changed])),
        pixel_shift_percentiles=pct(meta['shift'][ii,chosen][changed]),
        translation_change_m_percentiles=pct(np.linalg.norm(result[changed,:3,3]-baseline[changed,:3,3],axis=1)),
        max_camera_distance_change_m=float(np.max(np.abs(distance-radii))),
        max_rotation_orthogonality_error=float(np.max(np.abs(result[:,:3,:3].transpose(0,2,1)@result[:,:3,:3]-np.eye(3)))),
        run_seconds=time.perf_counter()-started,ground_truth_used=False)
    (args.out.parent/'metrics.json').write_text(json.dumps(metrics,indent=2))
    with open(args.out.parent/'frame_diagnostics.csv','w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['frame','changed','candidate','baseline_iou','final_iou','baseline_blue_iou','final_blue_iou','rotation_change_deg','axis_change_deg','pixel_shift','baseline_axis_pca_error_deg','image_score_before','image_score_after'])
        for i,j in enumerate(chosen):
            writer.writerow([i,int(changed[i]),int(j),base_iou[i],final_iou[i],blue[i,0],blue[i,j],np.degrees(meta['angle'][i,j]),np.degrees(meta['axis'][i,j]),meta['shift'][i,j],meta['pca'][i],candidates['scores'][i,0],candidates['scores'][i,j]])
    print(json.dumps(metrics,indent=2))
    print('Saved',args.out,flush=True)

if __name__=='__main__':
    main()
