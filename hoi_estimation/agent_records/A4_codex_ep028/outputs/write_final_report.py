from pathlib import Path
import numpy as np,json,hashlib
from scipy.spatial.transform import Rotation
p=Path('/work/outputs');T=np.load(p/'object_poses_world.npz')['T_world_mesh'];B=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float);d=np.load(p/'object_poses_world_diagnostics.npz');alpha=d['acceptance_weight'];run=json.load(open(p/'object_poses_world_run.json'));rows=json.load(open(p/'metrics_final.json'));W=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam']);C=np.linalg.inv(W)
R=np.load(p/'repro_check.npz')['T_world_mesh'];repeat_equal=np.array_equal(T,R);repeat_max=float(np.max(np.abs(T-R)));assert repeat_equal,'Final solver is not exactly repeatable'
def summarize(lo,hi):
 rr=[r for r in rows if lo<=r['frame']<hi];nums=[]
 for key in ['iou','head_error','tip_error']:
  for method in ['baseline','candidate']:
   nums.append(float(np.mean([r[method][key] for r in rr if r[method][key] is not None])))
 return nums
intervals=[(0,130),(130,500),(500,660),(660,690),(690,730),(730,790),(790,920),(920,988),(0,988)];table=[]
for lo,hi in intervals:
 v=summarize(lo,hi);label='Overall' if (lo,hi)==(0,988) else f'{lo}–{hi-1}';table.append(f'| {label} | {v[0]:.3f} | {v[1]:.3f} | {v[2]:.2f} | {v[3]:.2f} |')
v=summarize(0,988)
angles=[]
for q in [B,T]:angles.append(Rotation.from_matrix(q[1:,:3,:3]@q[:-1,:3,:3].transpose(0,2,1)).magnitude()*180/np.pi)
human=np.load('/work/inputs/reconstruction.npz')['human_keypoints_world']@C[:3,:3].T+C[:3,3];hand=[]
for i in [750,800]:
 vals=[]
 for q in [C@B,C@T]:
  shaft=np.c_[np.zeros(100),np.zeros(100),np.linspace(-.35,1.,100)]@q[i,:3,:3].T+q[i,:3,3];ds=np.linalg.norm(human[i,21:63,None,:]-shaft[None,:,:],axis=-1);vals.append(np.min(ds,axis=1).reshape(2,21).min(axis=1)*100)
 hand.append(f'At frame {i}, nearest reconstructed hand-to-shaft distances change from {vals[0][0]:.1f}/{vals[0][1]:.1f} cm to {vals[1][0]:.1f}/{vals[1][1]:.1f} cm.')
report=f'''# Squeegee trajectory correction

## Deliverable and offline reproduction

The required object_poses_world.npz contains T_world_mesh, shape **(988, 4, 4)**, **float64**, in the released world frame and original mesh-file coordinates. Every frame is present.

Run:

    python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz

The solver is self-contained and reads only supplied inputs. It requires no exploratory output, scratch file, network, model weights, or pipeline intermediates. Installed dependencies are listed in TOOLS.md. Two clean executions produced **bit-for-bit identical pose arrays** (maximum matrix difference {repeat_max:.1f}).

## Changes and evidence

The baseline fails during the turn behind the actor, around frames 700–790. Later it places the wide blade near the visible handle loop while the actual blade is at the upper end near the stool. The visible orange triangular mount, end loop, blade direction, and motion continuity provide clear evidence of this endpoint/orientation error.

1. Extract orange components and shaft geometry from supplied RGB and SAM masks. A saturated-orange threshold recovers the head where SAM omits it behind the arm.
2. Initialize the failed interval using identified blade and handle endpoints. Solve their camera-ray positions subject to the baseline center-to-camera distance. Preserve the continuous depth branch anchored before the failure.
3. Recover roll from visible blade directions. During partial occlusion, use explicit image-plane directions checked against RGB with OpenCV line detection. Constants and reasons are commented in solve.py; these are supplied-image measurements, not external 3-D annotations. The blade remains nearly vertical through frame 840, then becomes horizontal around frame 870.
4. Refine orientation and image-plane center with deterministic single-threaded CPU least squares. Fixed area-stratified samples from the exact supplied mesh are matched to object masks; person masks permit occlusion. Color landmarks, a baseline orientation prior, and temporal filtering of residual corrections constrain the fit. No random mesh sampling or GPU backward optimization is used in the final solver.
5. Render the full 95,575-face mesh for an every-frame acceptance check. Before the failed interval, require an overlap gain over 0.055, a supporting nine-frame median, noncontradictory landmark evidence, and bounded rotation/translation changes. Rejected frames retain the original world matrices exactly. The clearly failed late interval uses the endpoint-consistent reconstruction.

The final output changes **{run['changed_frames']} frames** and retains **{run['unchanged_frames']} baseline frames exactly**. Earlier changes are usually small; the large late rotation corrections are justified by the visible endpoint reversal.

An earlier differentiable GPU optimizer was tried and rejected after repeat testing exposed nondeterministic backward reductions. The final CPU optimizer was selected specifically to satisfy deterministic reproduction. OpenCV precise distance-transform results also varied slightly across processes; replacing that operation with SciPy Euclidean distance_transform_edt removed the remaining instability. Experimental outputs are not dependencies of solve.py.

## Distance and coordinate safeguards

The mesh-file origin equals its surface-area centroid to within about 4 nanometers. Its per-frame Euclidean camera distance is fixed to the baseline. The maximum measured difference is **{run['max_camera_distance_error_m']:.3g} m**. Camera z can change geometrically when the image ray changes; no independent depth or scale correction is estimated.

No ground-plane snapping, world alignment, rescaling, symmetry equivalence, or mesh-coordinate recentering is applied. All transforms use the original mesh-file frame and released camera-to-world matrix. Unchanged poses retain their small original float32 rotation errors; maximum rotation orthogonality error is **{run['max_rotation_orthogonality_error']:.3g}**. All transforms are finite, have valid homogeneous bottom rows, and positive determinants near one.

## Image-consistency measurements

These diagnostics are **not ground-truth ADD scores**. Masks and orange cues also inform fitting, so they are not a held-out accuracy test. No ground truth was accessed; actual ADD improvement cannot be certified locally.

The diagnostic renderer uses the full mesh at 768×576, discounts person-occluded predicted pixels, and compares with supplied masks. This summary covers {len(rows)} frames: every third frame plus 980–987. Landmark errors use only detected corresponding orange components.

| Frames | Baseline IoU | Final IoU | Baseline head error, px | Final head error, px |
|---|---:|---:|---:|---:|
'''+ '\n'.join(table)+f'''

Mean handle-loop error falls from **{v[4]:.2f} to {v[5]:.2f} pixels**. Maximum adjacent rotation step falls from **{angles[0].max():.2f}° to {angles[1].max():.2f}°**. Steps above 25° change from **{int((angles[0]>25).sum())} to {int((angles[1]>25).sum())}**. Median rotation step changes from {np.median(angles[0]):.2f}° to {np.median(angles[1]):.2f}°.

Additional non-optimized evidence: {' '.join(hand)} Reconstructed hands are approximate and were not treated as ground truth.

Evidence files include baseline_contact_sheet.jpg, rgb_details.jpg, transition.jpg, head_details.jpg, final_render.jpg, and metrics_final.json. Green contours are the supplied object masks.

## Confidence and limitations

Confidence is high that the late endpoint reversal is corrected and image alignment improves. Confidence is moderate in fine roll and shaft tilt through heavy occlusion, and lower in absolute metric accuracy where the fixed baseline distance is imperfect. The final stationary-looking segment deliberately retains baseline per-frame camera-distance drift. No ground-plane constraint was used to override it.

The 180° roll branch is propagated continuously from the correctly tracked blade before the turn, rather than treating the blade as symmetric. Short occluded intervals retain some roll uncertainty. Early frames with weak support for an edit preserve the baseline.

## Runtime and verification

The final raw-input run took **{run['runtime_seconds']:.2f} seconds ({run['runtime_seconds']/60:.2f} minutes)**. Deterministic CPU fitting took **{run['cpu_fit_seconds']:.2f} seconds**. Only forward full-mesh validation uses the selected logical CUDA device 0 ({run['device']}); no other GPU is selected. Total GPU work across experiments and checks remained below the 45-minute limit (conservative cumulative process-time upper bound below 40 minutes).

Both final clean runs used only the inputs and installed dependencies. Pose arrays are exactly equal, not merely approximately equal. Exact baseline equality was verified for all rejected frames. All 988 transforms passed finite-value, shape, dtype, determinant, orthogonality, homogeneous-row, and fixed-distance checks.

object_poses_world_run.json records runtime and structural checks. object_poses_world_diagnostics.npz records acceptance weights, candidate image scores, detected endpoints, and fitting losses. REPRODUCIBILITY.json records the repeat comparison and artifact hashes.
'''
(p/'REPORT.md').write_text(report)
validation=dict(bitwise_identical_pose_arrays=repeat_equal,max_absolute_difference=repeat_max,shape=list(T.shape),dtype=str(T.dtype),exact_unchanged_frames=int(np.sum(np.all(T==B,axis=(1,2)))),files={name:hashlib.sha256((p/name).read_bytes()).hexdigest() for name in ['solve.py','object_poses_world.npz','repro_check.npz']})
(p/'REPRODUCIBILITY.json').write_text(json.dumps(validation,indent=2)+'\n');print(json.dumps(validation,indent=2))
