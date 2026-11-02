"""Collect <out>/work/ep<NN>/result.json of all 30 episodes into <out>/proxies.csv (Tier-2-only proxies)."""
import csv, json, os, sys
import numpy as np
out = os.path.abspath(sys.argv[1])
PARTS = ['rw_bp', 'rw_br', 'b0', 'acc_p', 'acc_r', 'zv', 'rw_dt', 'rw_dr', 'd0', 'feet', 'hands_surf', 'hands_slip', 'floor', 'box']
rows = []
for ep in range(30):
    f = os.path.join(out, 'work', f'ep{ep:02d}', 'result.json')
    if not os.path.exists(f) or not os.path.exists(os.path.join(out, f'pose_ep{ep:02d}.npy')):
        rows.append(dict(ep=ep, status='FAILED')); continue
    r = json.load(open(f)); s1, s2 = r['stages']['1'], r['stages']['2']
    row = dict(ep=ep, obj=r['obj'], T=r['T'], T_vis=r['T_vis'], status='ok', time_s=round(r['time_s'], 1))
    for k in ('floor_min_cm', 'floor_share_below_1cm', 'box_pen_max_cm', 'box_share_pen_gt1cm', 'hand_d_cm'):
        row['init_' + k] = r['proxy_init'][k]; row['ours_' + k] = r['proxy_ours'][k]
    row['ours_hand_d_corr_cm'] = r['proxy_ours']['hand_d_corr_cm']
    row.update(r['correction'])
    row.update(s1_status=s1['solver']['status'], s1_iters=s1['solver']['iters'], s2_status=s2['solver']['status'],
               s2_iters=s2['solver']['iters'], converged=bool(s1['solver']['converged'] and s2['solver']['converged']),
               cost_total=round(s2['solver']['cost'] * 2, 2))
    for k in PARTS:
        row['cost_' + k] = round(s2['cost_end'].get(k, 0.0), 2)
    row.update(grasp_L=s2['grasp'][0], grasp_R=s2['grasp'][1], static=s2['static'], free=s2['free'],
               sup_floor=s2['sup_floor'], sup_box=s2['sup_box'], s_hf=round(r['s_hf'], 4))
    for k, v in list(row.items()):
        if isinstance(v, float):
            row[k] = round(v, 4) if np.isfinite(v) else ''
    rows.append(row)
keys = []
for r in rows:
    keys += [k for k in r if k not in keys]
with open(os.path.join(out, 'proxies.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
print('wrote', os.path.join(out, 'proxies.csv'), len(rows), 'rows,', sum(r['status'] == 'ok' for r in rows), 'ok')
