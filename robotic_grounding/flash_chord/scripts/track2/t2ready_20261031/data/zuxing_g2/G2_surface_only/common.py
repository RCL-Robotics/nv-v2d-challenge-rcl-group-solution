"""Paths, imports and the generic constants of the G2_surface_only pipeline (Tier-2 data only).

Every constant below is the same for all 30 episodes; per-episode values are derived from Tier-2 data in prep.py /
model.py by fixed rules (see README.md)."""
import sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent
KFIX = pathlib.Path('/root/autodl-tmp/zuxing.lu/tmp/tier2_noise/kfix')
sys.path.insert(0, str(KFIX / 'scripts'))
import kfilter as kf      # noqa: E402  (Haoran's Tier-2-only filter code, copied)
import kcommon as kc      # noqa: E402
sys.path.insert(0, '/root/autodl-tmp/zuxing.lu/ref_repos/v2d_submission_kit')
from v2dlb.vertices import mesh_path_for, sample_vertices   # noqa: E402,F401
STAGE = pathlib.Path('/root/autodl-tmp/haoran.zhang/data/human_motion_data_tier2/whole_body/reconstructed_stage')
INIT_FMT = str(KFIX / 'out' / 'agile+hgate__K3__{ep}.npy')     # fixed-K3, Tier-2-only initial estimate

CONST = dict(
    # --- inherited from B_hands (ep28), unchanged, same for every episode
    s_fs=0.10,            # stance-foot per-frame residual motion (cm/axis)
    slip_rw=0.0215,       # grip slip random-walk variance (cm^2/frame/axis); s_slip^2 = s_hf^2 + slip_rw
    s_c=2.5, n_corr=30,   # grip offset from the contact geometry (cm/axis); frames per independent offset sample (1 s)
    s_pen=0.3,            # floor / box penetration (cm)
    s_sup_f=1.0, s_sup_b=2.0,   # support contact on hand-free static frames, floor / box (cm)
    d0_free=5.0,          # hand-free: hand farther than d0 + 3 sigma from the object (cm)
    lever_human=45.0,     # hips-to-hand lever for the human root rotation drift (cm)
    settle=30,            # frames after the last hand contact during which a released object may still settle (1 s)
    win_h=7,              # half window (frames) of the co-motion / stance tests (ZUPT v1 H)
    box_near_m=0.6,       # box terms are evaluated on frames whose object is within this distance of a box (m)
    iters=60,             # LM iterations per stage
    # --- new in G2_surface_only
    surf_spacing_cm=0.3,  # dense surface sampling: one sample per spacing^2 of mesh area (<< s_c)
)
CHI2_1, CHI2_2, CHI2_3 = 6.634897, 9.210340, 11.344867      # chi-square 99 % quantiles, 1 / 2 / 3 dof
DEFAULT_OPTS = dict(handfree='dense', support='auto', box_pts='pts+hull', settle='body')   # ablation switches (see README)
