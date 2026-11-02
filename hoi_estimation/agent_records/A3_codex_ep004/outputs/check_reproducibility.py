"""Compare the two completed offline solver runs, including all candidates."""
from pathlib import Path
import hashlib,json
import numpy as np

root=Path('/work/outputs')
a=np.load(root/'object_poses_world.npz')['T_world_mesh']
b=np.load(root/'repro/object_poses_world.npz')['T_world_mesh']
x=np.load(root/'fit_diagnostics.npz');y=np.load(root/'repro/fit_diagnostics.npz')
s=np.load(root/'selection_diagnostics.npz');t=np.load(root/'repro/selection_diagnostics.npz')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
record=dict(poses_bitwise_equal=bool(np.array_equal(a,b)),
    maximum_pose_element_difference=float(np.max(np.abs(a-b))),
    all_fitted_candidates_bitwise_equal=bool(np.array_equal(x['candidates'],y['candidates'])),
    all_candidate_losses_bitwise_equal=bool(np.array_equal(x['losses'],y['losses'])),
    selection_decisions_identical=bool(np.array_equal(s['changed'],t['changed'])),
    first_output_sha256=sha(root/'object_poses_world.npz'),
    second_output_sha256=sha(root/'repro/object_poses_world.npz'),
    solver_sha256=sha(root/'solve.py'),
    first_run_seconds=json.loads((root/'validation.json').read_text())['total_seconds'],
    second_run_seconds=json.loads((root/'repro/validation.json').read_text())['total_seconds'])
(root/'reproducibility.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
assert record['poses_bitwise_equal'] and record['all_fitted_candidates_bitwise_equal']
assert record['all_candidate_losses_bitwise_equal'] and record['selection_decisions_identical']
