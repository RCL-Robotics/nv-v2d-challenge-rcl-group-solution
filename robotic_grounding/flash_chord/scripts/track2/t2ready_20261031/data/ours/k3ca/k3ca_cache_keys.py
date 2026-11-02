"""k3ca_cache_keys.py (K3CA training build, read-only except --out, CPU): exact CoACD cache keys of the collision meshes
of the 7 K3CA objects, computed from each job's OWN training parquet (data/human_motion_data_tier2_k3ca; the object assets
live in data/human_motion_data_tier2) with the scene code's own import path. Same method as tmp_analysis/t2fix/
cache_keys_t2.py (T2fix build) and tmp_analysis/t1fix_stage2/cache_keys.py (T1fix Stage 2): newton add_urdf, every MESH
shape with COLLIDE_SHAPES -> flash_chord.assets.mesh._cache_key on float64 vertices / int64 faces with the default CoACD
params (build_scene passes {}). Then each key's file is compared across the convex caches: the stock runtime's (= the
official evaluator's), code/fc_r2's (the raw C3r2_tier2 runs trained there without FLASH_CHORD_CACHE_DIR, so they used
it) and nv-v2d_runin's. Run with PYTHONPATH=code/fc_r2/src (the training tree) and CUDA_VISIBLE_DEVICES= (CPU).
Usage: python -B k3ca_cache_keys.py [--out FILE] <ep>:<parquet> ...
Writes nothing except FILE (JSON) when --out is given; never writes any cache.
"""
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ME = Path("/root/autodl-tmp/haoran.zhang")
CACHES = {
    "stock": ME / "code/video_to_data_8c778f9/robotic_grounding/flash_chord/src/flash_chord/assets/.cache/convex",
    "fc_r2": ME / "code/fc_r2/src/flash_chord/assets/.cache/convex",
    "nv-v2d_runin": ME / "code/nv-v2d_runin/robotic_grounding/flash_chord/src/flash_chord/assets/.cache/convex",
}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fp:
        for b in iter(lambda: fp.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def hullset(p):
    with np.load(p) as d:
        n = int(d["n"])
        return n, sorted((len(d[f"v{i}"]), len(d[f"f{i}"]), round(float(np.abs(d[f"v{i}"].astype(np.float64)).sum()), 5))
                         for i in range(n))


args = sys.argv[1:]
out_file = None
if args[:1] == ["--out"]:
    out_file, args = Path(args[1]), args[2:]
if not args:
    raise SystemExit(__doc__)

import warp as wp  # noqa: E402

wp.config.quiet = True
import newton  # noqa: E402

from flash_chord.assets import mesh as M  # noqa: E402
from flash_chord.data import load_reference  # noqa: E402
from flash_chord.scene import objects as O  # noqa: E402

PARAMS = dict(threshold=0.05, max_convex_hull=-1, preprocess_mode="auto")  # convex_decompose defaults; build_scene passes {}
out = {"written": datetime.datetime.now().strftime("%F %T"), "script": str(Path(__file__).resolve()),
       "flash_chord_module": M.__file__, "params": PARAMS, "objects": {}}
for arg in args:
    ep, _, parquet = arg.partition(":")
    ref = load_reference(Path(parquet))
    keys = []
    for asset in ref.object_assets():
        comp = O._object_builder(newton.ModelBuilder())
        comp.add_urdf(source=asset.urdf_path, floating=True, enable_self_collisions=False, collapse_fixed_joints=True,
                      ignore_inertial_definitions=False)
        collide, mesh_type = int(newton.ShapeFlags.COLLIDE_SHAPES), int(newton.GeoType.MESH)
        for s in range(comp.shape_count):
            if int(comp.shape_type[s]) == mesh_type and (int(comp.shape_flags[s]) & collide):
                src = comp.shape_source[s]
                v = np.ascontiguousarray(np.asarray(src.vertices), dtype=np.float64)
                f = np.ascontiguousarray(np.asarray(src.indices), dtype=np.int64).reshape(-1, 3)
                keys.append({"asset": asset.name, "urdf": str(asset.urdf_path), "shape": s, "nverts": len(v),
                             "key": M._cache_key(v, f, PARAMS)})
    per = {"parquet": parquet, "parquet_sha256": sha(parquet), "keys": keys, "files": {}}
    for k in keys:
        fname = f"{k['key']}.npz"
        stock_p = CACHES["stock"] / fname
        stock_sig = hullset(stock_p) if stock_p.is_file() else None
        stock_sha = sha(stock_p) if stock_p.is_file() else None
        row = {}
        for label, d in CACHES.items():
            p = d / fname
            if not p.is_file():
                row[label] = {"present": False}
                continue
            n, sig = hullset(p)
            row[label] = {"present": True, "sha256": sha(p), "n_hulls": n,
                          "mtime": datetime.datetime.fromtimestamp(p.stat().st_mtime).strftime("%F %T"),
                          "byte_identical_to_stock": stock_sha is not None and sha(p) == stock_sha,
                          "same_hull_set_as_stock": stock_sig is not None and (n, sig) == stock_sig}
        per["files"][fname] = row
    out["objects"][str(int(ep))] = per
    print(f"== ep{int(ep):02d} ({Path(parquet).parts[-3]}): keys {[k['key'] for k in keys]} urdf {[k['urdf'] for k in keys]}")
    for fname, row in per["files"].items():
        for label, r in row.items():
            if not r["present"]:
                print(f"   {fname} {label:13s} MISSING")
            else:
                print(f"   {fname} {label:13s} sha {r['sha256'][:16]} hulls {r['n_hulls']:3d} mtime {r['mtime']} "
                      f"bytes==stock {r['byte_identical_to_stock']} hullset==stock {r['same_hull_set_as_stock']}")
if out_file is not None:
    tmp = out_file.with_name(out_file.name + ".tmp")
    tmp.write_text(json.dumps(out, indent=1) + "\n")
    os.replace(tmp, out_file)
    print(f"wrote {out_file}")
