"""Fixes to Haoran's K1-K3 (copied code, subclassed, Tier-2 data only):
  agile : per-episode process noise q = max(pooled, the episode's own ML value) for q_a and q_al: never stiffer than
          the pooled prior, more agile when the episode's own noisy data asks for it (prevents the bias state from
          absorbing fast true motion, e.g. a bat swung through several turns).
  gate  : a K2 tilt/height anchor ('object resting alone in a stable pose') is kept only if
          (a) the object origin's height above the floor matches the anchored stable pose resting on the floor or on
              the top face of a support-stage shape, within HTOL cm, and
          (b) no object mesh point is within CONTACT cm of a support-stage shape below that shape's top face
              (touching its side = leaning on it).
  agile+gate : both.
  hgate / agile+hgate : the gate with test (a) only (height consistency); test (b) misfires on table-top objects
          whose noisy lowest points dip a little below the table top.
Writes out/<variant>__<K>__<ep>.npy (released frame, visible frames, xyzw). No clean data is read."""
import json, re, sys, pathlib
import numpy as np
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, str(pathlib.Path(__file__).parent / "scripts"))
import kfilter as kf
KIT = "/root/autodl-tmp/zuxing.lu/ref_repos/v2d_submission_kit"; sys.path.insert(0, KIT)
from v2dlb.vertices import mesh_path_for, sample_vertices
OUT = pathlib.Path(__file__).parent / "out"
T2 = pathlib.Path("/root/autodl-tmp/haoran.zhang/data/video_to_data_challenge/track_2/tier_2_synthetic_noise")
STAGE = pathlib.Path("/root/autodl-tmp/haoran.zhang/data/human_motion_data_tier2/whole_body/reconstructed_stage")
HTOL, CONTACT = 5.0, 3.0          # cm
ML = json.load(open(kf.DES / "d3_ml.json"))

def stage_boxes(seq):
    """Axis-aligned boxes (lo, hi) in the robot world (m) from the support USDA (Cube: translate/scale, size 1;
    Cylinder: translate, radius, height, axis Z)."""
    f = STAGE / f"{seq}_support.usda"
    if not f.exists(): return []
    txt = f.read_text(); boxes = []
    for kind, body in re.findall(r'def (Cube|Cylinder) "\w+"[^{]*\{(.*?)\n    \}', txt, flags=re.S):
        t = np.array([float(x) for x in re.search(r"xformOp:translate = \(([^)]*)\)", body).group(1).split(",")])
        s = re.search(r"xformOp:scale = \(([^)]*)\)", body)
        sc = np.array([float(x) for x in s.group(1).split(",")]) if s else np.ones(3)
        if kind == "Cube":
            size = float(re.search(r"double size = ([\d.eE+-]+)", body).group(1)) if "double size" in body else 2.0
            half = 0.5 * size * sc
        else:
            r = float(re.search(r"double radius = ([\d.eE+-]+)", body).group(1)); h = float(re.search(r"double height = ([\d.eE+-]+)", body).group(1))
            half = np.array([r, r, h / 2]) * sc
        boxes.append((t - half, t + half))
    return boxes

class FixEpisode(kf.Episode):
    gate = False
    height_only = False
    def __init__(self, m, prm):
        super().__init__(m, prm)
        self.boxes = stage_boxes(m["sequence_id"])
        self.verts = sample_vertices(mesh_path_for(T2 / "mesh", self.obj))      # m, object frame
        self.B, self.b = np.array(self.fr["Bcv2r"]), np.array(self.fr["bcv2r"])
        self.gate_log = []
    def admissible(self, X1, tm, hk_cm):
        rot = R.from_rotvec(-X1[tm, kf.BT]) * self.rot[tm]
        pcv = rot.apply(self.verts) + X1[tm, kf.P_] / 100.0                    # m, OpenCV world
        pr = pcv @ self.B.T + self.b                                            # m, robot world (Z up)
        h_meas = float(self.up @ X1[tm, kf.P_])
        resid = h_meas - hk_cm
        tops = [0.0] + [100.0 * hi[2] for lo, hi in self.boxes]
        ok_h = min(abs(resid - top) for top in tops) <= HTOL
        side = False
        for lo, hi in self.boxes:
            d = np.linalg.norm(np.maximum(np.maximum(lo - pr, pr - hi), 0.0), axis=1) * 100.0
            near = d <= CONTACT
            if near.any() and (pr[near, 2] * 100.0 < 100.0 * hi[2] - 1.0).any():
                side = True
        self.gate_log.append(dict(t=round(tm / 30, 2), resid=round(resid, 1), ok_h=bool(ok_h), side=bool(side),
                                  floor_z_cm=round(float(pr[:, 2].min() * 100), 1)))
        return ok_h if self.height_only else (ok_h and not side)
    def anchors(self, X1, P1):
        out, diag = super().anchors(X1, P1)
        if not self.gate: return out, diag
        keep, by_t = [], {}
        for tm, mm in out: by_t.setdefault(tm, []).append(mm)
        E = kf.horiz_basis(self.up); prm = self.prm
        for tm, mms in by_t.items():
            S = E @ P1[tm][kf.BT, kf.BT] @ E.T + np.radians(prm["sigma_tilt_deg"]) ** 2 * np.eye(2); bh = E @ X1[tm, kf.BT]
            ms = []
            for k in range(len(self.sp_h)):
                dk = self.rot[tm].apply(self.sp_n[k]); zk = E @ (-kf.min_rotvec(dk, -self.up)); nu = zk - bh; ms.append(nu @ np.linalg.solve(S, nu))
            ms = np.array(ms); inn = ms < kf.CHI2_2_99
            w = self.sp_prob[inn] * np.exp(-0.5 * ms[inn]); w = w / w.sum() if w.sum() > 0 else np.ones(inn.sum()) / inn.sum()
            hk = float((w * (100.0 * (self.sp_h[inn] - self.e) + prm["height_mu_cm"])).sum())
            if self.admissible(X1, tm, hk):
                keep += [(tm, mm) for mm in mms]
        diag["gate_kept"] = len({t for t, _ in keep}); diag["gate_dropped"] = len(by_t) - diag["gate_kept"]
        return keep, diag

base = kf.load_params(); metas = {m["episode_index"]: m for m in kf.episodes()}
eps = [int(x) for x in sys.argv[2:]] or sorted(metas)
logs = {}
for variant in sys.argv[1].split(","):
    for ep in eps:
        prm = json.loads(json.dumps(base))
        if "agile" in variant:
            prm["q_a_cm2_s3"][str(ep)] = max(base["q_a_cm2_s3"][str(ep)], ML[str(ep)]["pos"]["q_ml"])
            prm["q_al_rad2_s3"][str(ep)] = max(base["q_al_rad2_s3"][str(ep)], ML[str(ep)]["rot"]["q_ml"])
        e = FixEpisode(metas[ep], prm); e.gate = "gate" in variant; e.height_only = "hgate" in variant
        for K in ("K1", "K2", "K3"):
            pose, X, diag = e.variant(K)
            np.save(OUT / f"{variant}__{K}__{ep}.npy", pose)
        logs[f"{variant}:{ep}"] = e.gate_log
        if e.gate:
            e.gate_log = [g for g in e.gate_log]
            print(variant, ep, "anchors kept/dropped", diag.get("gate_kept"), diag.get("gate_dropped"),
                         [g for g in e.gate_log if not (g["ok_h"] and not g["side"])][:6], flush=True)
        else: print(variant, ep, "ok", flush=True)
json.dump(logs, open(OUT.parent / f"gate_log_{sys.argv[1].replace(',', '_')}.json", "w"), indent=1)
