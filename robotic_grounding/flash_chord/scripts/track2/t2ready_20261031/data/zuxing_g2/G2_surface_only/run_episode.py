"""G2_surface_only pipeline, one episode (Tier-2 data only, deterministic, no per-episode hand-set values).

usage: run_episode.py <ep> --out <dir> [--opt key=value ...] [--reuse-cache]
  writes <out>/pose_ep<NN>.npy ((T_visible,7) [x y z qx qy qz qw], metres, OpenCV world; same layout as the initial
  estimate) and <out>/work/ep<NN>/ (cache, stage states and masks, result.json with proxies, run.log).
  --opt changes an ablation switch of common.DEFAULT_OPTS for every episode alike (handfree, support, box_pts)."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.spatial.transform import Rotation as R
import prep, model as M, detect as D, proxies as PX
from common import CONST, DEFAULT_OPTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ep', type=int); ap.add_argument('--out', required=True)
    ap.add_argument('--opt', action='append', default=[]); ap.add_argument('--reuse-cache', action='store_true')
    a = ap.parse_args()
    opts = dict(DEFAULT_OPTS)
    for kv in a.opt:
        k, v = kv.split('=', 1)
        assert k in DEFAULT_OPTS, k
        opts[k] = v
    ep = a.ep; out = os.path.abspath(a.out); work = os.path.join(out, 'work', f'ep{ep:02d}')
    os.makedirs(work, exist_ok=True)
    flog = open(os.path.join(work, 'run.log'), 'w')

    def log(*s):
        msg = ' '.join(str(v) for v in s)
        print(msg, flush=True); flog.write(msg + '\n'); flog.flush()

    t_start = time.time()
    log(f'== G2_surface_only ep{ep} opts {opts} started {time.strftime("%Y-%m-%d %H:%M:%S %z")}')
    cache = os.path.join(work, 'cache.npz')
    if not (a.reuse_cache and os.path.exists(cache)):
        prep.build(ep, cache, log)
    E = M.Episode(cache, opts)
    log(f'object {E.obj} T {E.T} visible {int(E.vis.sum())} grip points/hand {E.K} s_hf {E.s_hf:.4f} s_slip {E.P["s_slip"]:.4f} '
        f's_surf {E.P["s_surf"]:.2f} s_slip_pt {E.P["s_slip_pt"]:.3f} surface samples {len(E.S)} boxes {len(E.BOXES)}')
    x_init = E.x_from_pose(E.init)
    Pb = E.Pb
    sbp = np.sqrt(np.einsum('tii->ti', Pb[:, :3, :3]).mean(1)); sbr = np.sqrt(np.einsum('tii->ti', Pb[:, 3:, 3:]).mean(1))
    st = D.stance_masks(E)
    x = x_init; res = dict(ep=ep, obj=E.obj, T=E.T, T_vis=int(E.vis.sum()), opts=opts, stages={})
    for stage in (1, 2):
        free, dmin = D.handfree(E, x, sbp, sbr)
        bfree, _ = D.bodyfree(E, x, sbp, sbr)
        static = D.static_mask(E, free, bfree)
        g, info = D.grasp_mask(E, x, sbp, sbr, static)
        sup = static & free
        sup_f, sup_b, slog = D.support_contacts(E, x, sbp, sbr, sup)
        log(f'== stage {stage}: free {int(free.sum())} {M.runs(free)[:12]} body-free {int(bfree.sum())} {M.runs(bfree)[:12]} static {int(static.sum())} {M.runs(static)[:12]}')
        for s in (0, 1):
            log(f'   grasp {s}: {int(g[s].sum())} frames {M.runs(g[s])[:12]} (near {info[s]["near"]} co-moving {info[s]["com"]})')
        log(f'   support contacts: {slog}')
        moving_nohand = (~static) & ~(g[0] | g[1])
        log(f'   non-static frames without a detected grasp: {int(moving_nohand.sum())}')
        prob = M.Problem(E, static, g, st, sup_f, sup_b, x)
        c0 = prob.cost_parts(x)
        log('   cost parts at start', {k: round(v, 1) for k, v in c0.items()})
        t0 = time.time()
        x, sinfo = prob.solve(x, iters=CONST['iters'], log=log)
        c1 = prob.cost_parts(x)
        log(f'   solved in {time.time()-t0:.0f}s ({sinfo["status"]}, {sinfo["iters"]} it); cost parts', {k: round(v, 1) for k, v in c1.items()})
        np.save(os.path.join(work, f'x_s{stage}.npy'), x)
        sbp, sbr = D.posterior_std(prob, x)
        log('   posterior std b_p (cm) / b_r (deg) every 100 frames:', np.round(sbp[::100], 2).tolist(), np.round(np.degrees(sbr[::100]), 2).tolist())
        np.savez(os.path.join(work, f'masks_s{stage}.npz'), free=free, bfree=bfree, static=static, grasp=g, sup=sup, sup_f=sup_f, sup_b=sup_b,
                 stanceL=st['L'], stanceR=st['R'], sbp=sbp, sbr=sbr)
        res['stages'][stage] = dict(free=int(free.sum()), bodyfree=int(bfree.sum()), static=int(static.sum()), grasp=[int(g[0].sum()), int(g[1].sum())],
                                    sup=int(sup.sum()), sup_floor=int(sup_f.sum()), sup_box=int(sup_b.any(1).sum()) if sup_b.size else 0,
                                    support_log=slog, cost_start=c0, cost_end=c1, solver=sinfo, time_s=time.time() - t0)
    pose = E.to_pose(x)[E.vis]
    q = pose[:, 3:]
    ok = pose.shape == E.init.shape and np.isfinite(pose).all() and np.abs(np.linalg.norm(q, axis=1) - 1).max() < 1e-9
    assert ok, 'output check failed'
    np.save(os.path.join(out, f'pose_ep{ep:02d}.npy'), pose)
    mk = np.load(os.path.join(work, 'masks_s2.npz'))
    res['proxy_init'] = PX.pose_proxies(E, x_init, mk['grasp'])
    res['proxy_ours'] = PX.pose_proxies(E, x, mk['grasp'])
    res['correction'] = PX.correction(E, x_init, x)
    res['time_s'] = time.time() - t_start
    res['s_hf'] = E.s_hf
    json.dump(res, open(os.path.join(work, 'result.json'), 'w'), indent=1, default=float)
    log(f'done ep{ep}: pose {pose.shape} finite {np.isfinite(pose).all()} in {res["time_s"]:.0f}s')
    log('proxies init', res['proxy_init']); log('proxies ours', res['proxy_ours']); log('correction', res['correction'])


if __name__ == '__main__':
    main()
