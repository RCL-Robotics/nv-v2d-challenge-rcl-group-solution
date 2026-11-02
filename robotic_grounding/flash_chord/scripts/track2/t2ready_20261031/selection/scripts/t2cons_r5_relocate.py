#!/usr/bin/env python3
"""t2cons_r5_relocate.py: round 5 of the Tier-2 consensus selection (t2_consensus_r5_20261029), relocation step.

This file is scripts/t2cons_r4_relocate.py (7b624b85; round 4, itself round 2's AMENDMENTS_R2 A1 tool) with ONLY the
round-5 changes below. The edit itself (relocate_bytes), the read-only verification (verify_pair) and the selftest are
byte-for-byte the round-2/3/4 logic.

Metadata-only relocation of the 4 copied 5090 checkpoints of Zuxing's retest_best_2026-10-29 bundle (candidates_r5.json
ids C01 ep16 G2, C02 ep19 hard-G2, C03 ep20 K3fix, C04 ep25 K3fix). For each copy
zuxing_copies_r5/<bundle>/<label>/model_<it>.safetensors it writes a NEW file relocated_r5/<bundle>/<label>/model_<it>.safetensors
(mode 444) in which exactly two string values of the safetensors header's __metadata__["user.resolved_config_json"]
change: task.parquet and reference.parquet, from the training machine's path (candidates_r5.json
training_parquet_as_trained: /mnt/gfs/home/zuxinglu/workspace/data/<root>/... on the 5090) to the pro6000 path of the
same parquet (candidates_r5.json training_parquet: <root> = human_motion_data_tier2_g2-2026-10-24 =
zuxing.lu/data/human_motion_data_tier2_g2-2026-10-24 (read only) for G2, human_motion_data_tier2_k3fix = our
data/human_motion_data_tier2_k3fix for K3fix, as round 3's 5090 K3fix relocations). Nothing else changes: no other
metadata key, no other config field (output_dir, training.checkpoint stay), no tensor byte. The copies are only read;
Zuxing's folder is not read at all here.

Round-5 changes vs round 4: ROOT and file names (relocated_r5/, RELOCATION_R5.*); the job set (C01-C04 of
candidates_r5.json, pinned by sha256 below; this step precedes PREREG_R5.md, which pins this tool and its records before
any export); the data roots (G2 and K3fix; the new path must be the same relative path below the same root name); the
parquet sha256 checks (candidates_r5.json == TRACKS_R5.sha256 == resolution record == SOURCE.json == the input's build
record: Zuxing's G2 build_*.json for G2, our K3fix data_build.json for K3fix, as the resolver recorded in parquet_checks).

Verification (refuses on any failure): original sha256 == candidates_r5.json == the bundle's SHA256SUMS.*.zuxing line ==
SOURCE.json; every tensor's name / dtype / shape / data_offsets equal and bytes identical (raw slices + whole data block
sha256 + safetensors.safe_open numpy tobytes); every other metadata key identical; config diff == {task.parquet,
reference.parquet}; old values == as-trained path; new path = an existing file in the job's data root with the job's
sequence, sha256 == candidates_r5.json == TRACKS_R5.sha256 == SOURCE.json (build record); Path(new).resolve() is that
file (the harness uses a non-/workspace absolute path as is); original copy unchanged afterwards.

Outputs (new files only, never overwritten): relocated_r5/..., RELOCATION_R5.tsv, RELOCATION_R5.json, RELOCATION_R5.sha256.

  python -I -B scripts/t2cons_r5_relocate.py selftest   synthetic files in ROOT/tmp/relocate_selftest_<stamp>/ only
  python -I -B scripts/t2cons_r5_relocate.py run        the 4 relocations (refuses if any output exists)
  python -I -B scripts/t2cons_r5_relocate.py verify     read-only re-verification of RELOCATION_R5.tsv

Importable (the round-5 driver loads it from its pinned bytes): relocate_bytes, verify_pair, verify_record. Importing
has no side effect. No clean path is ever opened (guard below). Never signals any process, never uses a GPU.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import csv  # noqa: E402
import datetime as dt  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import socket  # noqa: E402
import struct  # noqa: E402
from pathlib import Path  # noqa: E402

ME = Path("/root/autodl-tmp/haoran.zhang")
ROOT = ME / "fc_runs/submissions/t2_consensus_r5_20261029"
CANDS = ROOT / "inputs_r5/candidates_r5.json"
CANDS_SHA = "3d03f6cfd92c8ae780a54114135ca1c9fb8f6fe067c0da2a1a9bb402d2bb6bd1"  # resolver output (T_c 04:30:46)
TRACKS = ROOT / "inputs_r5/TRACKS_R5.sha256"
TRACKS_SHA = "c578005075edbc7add8075676a1d029c009d3a654e7529cac3522da4f709922e"
SNAP = ROOT / "inputs_r5/SNAPSHOT_R5.sha256"
SNAP_SHA = "cdaed051ade8c81680315d126f32ee489d96244b8e838507679a16f286829bf0"
COPIES = ROOT / "zuxing_copies_r5"
OUT_DIR = ROOT / "relocated_r5"
OUT_TSV, OUT_JSON, OUT_SHA = ROOT / "RELOCATION_R5.tsv", ROOT / "RELOCATION_R5.json", ROOT / "RELOCATION_R5.sha256"
DATA_ROOTS = {"g2": Path("/root/autodl-tmp/zuxing.lu/data/human_motion_data_tier2_g2-2026-10-24"),  # read only
              "k3fix": ME / "data/human_motion_data_tier2_k3fix"}  # fmt: skip
BUILD_KEY = {"g2": "g2_build_record", "k3fix": "k3fix_build_record"}  # the resolver's parquet_checks key per input
AS_TRAINED_PREFIXES = ("/mnt/gfs/home/zuxinglu/workspace/data/", "/home/zuxinglu/workspace/data/")  # 5090, robolab
META_KEY = "user.resolved_config_json"
FIELDS = ("task.parquet", "reference.parquet")  # the only two values that change (as round 2, AMENDMENTS_R2 A1)
C_IDS = ("C01", "C02", "C03", "C04")  # the relocated round-5 jobs (name C_IDS kept from round 2's code)
FORBIDDEN = ("clean", "local_not_for_upload", "packed_", "readme", "train_rl.log", "t2_zx07_20261028")
TSV_COLS = ["id", "job", "episode", "iteration", "status", "original_copy", "original_sha256", "sha256sums_zuxing_line",
            "relocated_file", "relocated_sha256", "fields_rewritten", "old_parquet_path", "new_parquet_path",
            "parquet_sha256_new_path", "parquet_sha_matches", "n_tensors", "tensors_identical", "safe_open_identical",
            "data_block_sha256", "other_metadata_identical", "config_diff_paths", "header_len_old", "header_len_new",
            "resolves_to", "original_unchanged_after", "written_at"]  # fmt: skip


# ================================================================================================ helpers
def now() -> str:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")


def guard(p) -> Path:
    s = str(p).lower()
    if any(tok in s for tok in FORBIDDEN):
        raise PermissionError(f"t2cons_r5_relocate may not open {p}")
    return Path(p)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with guard(p).open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def flat(o, prefix: str = "") -> dict:
    out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out.update(flat(v, f"{prefix}.{k}" if prefix else str(k)))
    else:
        out[prefix] = o
    return out


def get_path(cfg: dict, dotted: str):
    a, b = dotted.split(".")
    return cfg[a][b]


def set_path(cfg: dict, dotted: str, value) -> None:
    a, b = dotted.split(".")
    cfg[a][b] = value


# ================================================================================================ safetensors
def split_st(raw: bytes) -> tuple[int, bytes, dict, bytes]:
    """(header length N, header bytes, parsed header (order kept), data block)."""
    if len(raw) < 8:
        raise ValueError("file shorter than 8 bytes")
    n = struct.unpack("<Q", raw[:8])[0]
    if n <= 0 or 8 + n > len(raw):
        raise ValueError(f"bad header length {n}")
    hb = raw[8:8 + n]
    return n, hb, json.loads(hb), raw[8 + n:]


def relocate_bytes(raw: bytes, new_path: str, expect_old: str) -> tuple[bytes, dict]:
    """The A1 edit on bytes: only FIELDS of META_KEY change; header re-serialised exactly as stored; data block as is."""
    n, hb, h, data = split_st(raw)
    if json.dumps(h, separators=(",", ":")).encode() != hb.rstrip(b" "):
        raise ValueError("original header does not round-trip (compact JSON, stored key order)")
    md = h.get("__metadata__")
    if not isinstance(md, dict) or META_KEY not in md:
        raise ValueError(f"no __metadata__[{META_KEY!r}]")
    cfg_s = md[META_KEY]
    cfg = json.loads(cfg_s)
    if json.dumps(cfg, sort_keys=True, separators=(",", ":")) != cfg_s:
        raise ValueError(f"{META_KEY} does not round-trip (sort_keys, compact)")
    old = {}
    for f in FIELDS:
        old[f] = get_path(cfg, f)
        if old[f] != expect_old:
            raise ValueError(f"{f} = {old[f]!r} != the expected as-trained path {expect_old!r}")
        set_path(cfg, f, new_path)
    md2 = {k: (json.dumps(cfg, sort_keys=True, separators=(",", ":")) if k == META_KEY else v) for k, v in md.items()}
    h2 = {k: (md2 if k == "__metadata__" else v) for k, v in h.items()}
    hb2 = json.dumps(h2, separators=(",", ":")).encode()
    hb2 += b" " * ((-len(hb2)) % 8)  # 8-byte alignment, as the safetensors writer pads
    new = struct.pack("<Q", len(hb2)) + hb2 + data
    return new, {"old": old, "new": new_path, "header_len_old": n, "header_len_new": len(hb2)}


def verify_pair(orig: Path, reloc: Path, expect_old: str, expect_new: str) -> dict:
    """Read-only: the relocated file differs from the original only at FIELDS of META_KEY (old -> new); every tensor's
    bytes are identical (raw slices, whole data block, and safetensors.safe_open numpy tobytes)."""
    problems: list[str] = []
    ra, rb = guard(orig).read_bytes(), guard(reloc).read_bytes()
    na, _, ha, da = split_st(ra)
    nb, _, hb, db = split_st(rb)
    ta = {k: v for k, v in ha.items() if k != "__metadata__"}
    tb = {k: v for k, v in hb.items() if k != "__metadata__"}
    if list(ta) != list(tb):
        problems.append("tensor names / order differ")
    n_bad = 0
    for k in ta:
        a, b = ta[k], tb.get(k)
        if b is None or a.get("dtype") != b.get("dtype") or a.get("shape") != b.get("shape") or a.get("data_offsets") != b.get("data_offsets"):
            n_bad += 1
            continue
        s, e = a["data_offsets"]
        if not (0 <= s <= e <= len(da)) or da[s:e] != db[s:e]:
            n_bad += 1
    if n_bad:
        problems.append(f"{n_bad} tensors differ (dtype/shape/offsets/bytes)")
    ends = sorted(tuple(v["data_offsets"]) for v in ta.values())
    if not ends or ends[0][0] != 0 or ends[-1][1] != len(da) or any(x[1] != y[0] for x, y in zip(ends, ends[1:])):
        problems.append("original data block is not exactly covered by the tensors")
    dsa, dsb = sha256_bytes(da), sha256_bytes(db)
    if dsa != dsb or len(da) != len(db):
        problems.append("data block sha256 / length differ")
    # safetensors' own reader
    so_ok = True
    try:
        from safetensors import safe_open

        with safe_open(str(orig), framework="numpy") as fa, safe_open(str(reloc), framework="numpy") as fb:
            ka, kb = sorted(fa.keys()), sorted(fb.keys())
            if ka != kb:
                so_ok = False
                problems.append("safe_open: key sets differ")
            for k in ka:
                xa, xb = fa.get_tensor(k), fb.get_tensor(k)
                if xa.dtype != xb.dtype or xa.shape != xb.shape or xa.tobytes() != xb.tobytes():
                    so_ok = False
                    problems.append(f"safe_open: tensor {k} differs")
                    break
            mda, mdb = fa.metadata() or {}, fb.metadata() or {}
    except Exception as error:  # noqa: BLE001
        so_ok = False
        mda, mdb = ha.get("__metadata__", {}), hb.get("__metadata__", {})
        problems.append(f"safe_open failed: {error!r}")
    # metadata
    ma, mb = ha.get("__metadata__", {}), hb.get("__metadata__", {})
    other_ok = list(ma) == list(mb) and all(ma[k] == mb[k] for k in ma if k != META_KEY) and dict(mda) == {
        k: v for k, v in ma.items()} and dict(mdb) == {k: v for k, v in mb.items()}  # fmt: skip
    if not other_ok:
        problems.append("metadata keys other than the config differ (or safe_open metadata != header metadata)")
    ca, cb = json.loads(ma.get(META_KEY, "{}")), json.loads(mb.get(META_KEY, "{}"))
    fa_, fb_ = flat(ca), flat(cb)
    diff = sorted(k for k in set(fa_) | set(fb_) if fa_.get(k, "<absent>") != fb_.get(k, "<absent>"))
    if diff != sorted(FIELDS):
        problems.append(f"config differs at {diff}, expected exactly {sorted(FIELDS)}")
    for f in FIELDS:
        if fa_.get(f) != expect_old:
            problems.append(f"original {f} {fa_.get(f)!r} != as-trained {expect_old!r}")
        if fb_.get(f) != expect_new:
            problems.append(f"relocated {f} {fb_.get(f)!r} != {expect_new!r}")
    if mb.get(META_KEY) != json.dumps(cb, sort_keys=True, separators=(",", ":")):
        problems.append("relocated config is not serialised as stored (sort_keys, compact)")
    if nb % 8:
        problems.append(f"relocated header length {nb} not a multiple of 8")
    return {"ok": not problems, "problems": problems, "n_tensors": len(ta), "tensors_identical": n_bad == 0 and dsa == dsb,
            "safe_open_identical": so_ok, "data_block_sha256": dsa, "other_metadata_identical": other_ok,
            "config_diff_paths": diff, "header_len_old": na, "header_len_new": nb}  # fmt: skip


def verify_record(row: dict) -> list[str]:
    """Read-only re-check of one RELOCATION_R5 row (used by the round-5 driver at every start)."""
    problems = []
    orig, reloc = Path(row["original_copy"]), Path(row["relocated_file"])
    if not str(orig).startswith(str(COPIES) + "/") or not str(reloc).startswith(str(OUT_DIR) + "/"):
        return [f"unexpected paths {orig} / {reloc}"]
    if not orig.is_file() or sha256_file(orig) != row["original_sha256"]:
        problems.append(f"original copy {orig} missing or != {row['original_sha256'][:16]}")
    if not reloc.is_file() or sha256_file(reloc) != row["relocated_sha256"]:
        problems.append(f"relocated file {reloc} missing or != {row['relocated_sha256'][:16]}")
    if problems:
        return problems
    v = verify_pair(orig, reloc, row["old_parquet_path"], row["new_parquet_path"])
    problems += v["problems"]
    newp = Path(row["new_parquet_path"])
    if not newp.is_file() or sha256_file(newp) != row["parquet_sha256_new_path"]:
        problems.append(f"{newp} missing or sha256 != {row['parquet_sha256_new_path'][:16]}")
    elif not newp.resolve().is_file():
        problems.append(f"{newp} does not resolve to a file")
    return problems


# ================================================================================================ inputs and pins
def check_pins() -> list[str]:
    """The resolver's outputs, pinned here (this step precedes PREREG_R5.md, which then pins this tool and its records)."""
    bad = []
    for f, want in ((CANDS, CANDS_SHA), (TRACKS, TRACKS_SHA), (SNAP, SNAP_SHA)):
        got = sha256_file(f) if f.is_file() else "missing"
        if got != want:
            bad.append(f"{f.name} sha256 {got[:16]} != pinned {want[:16]}")
    return bad


def prereg_c_table() -> dict[str, dict]:
    """Round 5: the candidates_r5.json resolutions (resolver record) of C_IDS: id -> job, model sha256, parquet sha256
    (stands in for round 2's PREREG_R2 2.1 table, since this step precedes PREREG_R5.md)."""
    cands = json.loads(guard(CANDS).read_text())
    out = {}
    for r in cands.get("resolutions", []):
        if r.get("id") in C_IDS and r.get("status") == "new":
            out[r["id"]] = {"job": r["job"], "model_sha256": r["sha256"], "parquet_prefix": r["parquet_sha256"]}
    return out


def track_records() -> dict[str, str]:
    out = {}
    for line in guard(TRACKS).read_text().splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            out[os.path.normpath(parts[1].strip())] = parts[0]
    return out


def root_rel(path: str, name: str) -> str | None:
    """The path below the data root name `name` (same on every machine), or None."""
    parts = Path(path).parts
    if name not in parts:
        return None
    return "/".join(parts[parts.index(name) + 1:])


def layout_seq(p: Path) -> str | None:
    """The sequence id of a .../sequence_id=<seq>/robot_name=g1/data.parquet path (round 4/5: SOURCE.json check)."""
    par = p.parent.parent.name
    return par.split("=", 1)[1] if par.startswith("sequence_id=") else None


def plan_job(nj: dict, table: dict, tracks: dict) -> tuple[dict, list[str]]:
    p: list[str] = []
    jid, it = nj["job"], int(nj["iteration"])
    orig = Path(nj["checkpoint"])
    rec = {"id": nj["id"], "job": jid, "episode": int(nj["episode"]), "iteration": it, "original_copy": str(orig)}
    if not str(orig).startswith(str(COPIES) + "/") or orig.name != f"model_{it}.safetensors":
        return rec, [f"{jid}: checkpoint {orig} is not a model_{it} copy in zuxing_copies_r5/"]
    bundle, label = orig.parent.parent, orig.parent.name
    osha = sha256_file(orig)
    rec["original_sha256"] = osha
    want_name = f"./{label}/{orig.name}"
    lines = []
    for sums in sorted(bundle.glob("SHA256SUMS.*.zuxing")):  # the bundle's copied checksum lists
        lines += [ln.split() + [sums.name] for ln in guard(sums).read_text().splitlines()]
    hits = [f for f in lines if len(f) == 3 and os.path.normpath(f[1]) == os.path.normpath(want_name)]
    rec["sha256sums_zuxing_line"] = f"{hits[0][0]}  {hits[0][1]} ({hits[0][2]})" if len(hits) == 1 else "MISSING"
    t = table.get(nj["id"], {})
    if not (len(hits) == 1 and osha == nj["sha256"] == hits[0][0] == t.get("model_sha256")) or t.get("job") != jid:
        p.append(f"{jid}: original sha256 {osha[:16]} vs candidates {nj['sha256'][:16]} / SHA256SUMS.*.zuxing "
                 f"{rec['sha256sums_zuxing_line'][:16]} / resolution {str(t.get('model_sha256'))[:16]} {t.get('job')}")
    src = json.loads(guard(orig.parent / "SOURCE.json").read_text())
    old, new = nj["training_parquet_as_trained"], nj["training_parquet"]
    rec["old_parquet_path"], rec["new_parquet_path"] = old, new
    if src.get("training_parquet_as_trained") != old or src.get("training_parquet_pro6000") != new or src.get("model_sha256") != osha:
        p.append(f"{jid}: SOURCE.json disagrees with candidates_r5.json (paths / model sha256)")
    newp = Path(new)
    droot = DATA_ROOTS.get(str(nj.get("input")))
    if droot is None:
        return rec, p + [f"{jid}: input {nj.get('input')} has no relocation data root"]
    name = droot.name
    if not str(newp).startswith(str(droot) + "/whole_body/soma/sequence_id=") or newp.name != "data.parquet" \
            or newp.parent.name != "robot_name=g1" or root_rel(old, name) is None or root_rel(old, name) != root_rel(new, name) \
            or old == new or not any(old.startswith(px + name + "/") for px in AS_TRAINED_PREFIXES):
        p.append(f"{jid}: new path {new} is not the same {name} relative path as {old} in {droot}")
    if not newp.is_file():
        p.append(f"{jid}: {new} does not exist")
        return rec, p
    psha = sha256_file(newp)
    rec["parquet_sha256_new_path"] = psha
    bc = src.get("parquet_checks") or {}
    bkey = BUILD_KEY[str(nj.get("input"))]
    m = {"candidates_r5.json": psha == nj["parquet_sha256"], "TRACKS_R5.sha256": tracks.get(os.path.normpath(new)) == psha,
         "resolution_record": bool(t.get("parquet_prefix")) and psha == t["parquet_prefix"],
         f"SOURCE.json({bkey})": src.get("parquet_sha256") == psha and bc.get(bkey) is True
                                 and bc.get("sequence_id") == (layout_seq(newp) or "?")}  # fmt: skip
    rec["parquet_sha_matches"] = ";".join(f"{k}={v}" for k, v in m.items())
    if not all(m.values()):
        p.append(f"{jid}: parquet sha256 {psha[:16]} checks {m}")
    res = newp.resolve()
    rec["resolves_to"] = str(res)
    if not res.is_file():
        p.append(f"{jid}: Path({new}).resolve() is not a file")
    rec["relocated_file"] = str(OUT_DIR / bundle.name / label / orig.name)
    return rec, p


# ================================================================================================ commands
def write_new(path: Path, data: bytes, mode: int = 0o444) -> None:
    if not str(path).startswith(str(ROOT) + "/"):
        raise PermissionError(f"refusing to write outside {ROOT}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as fp:  # never overwrite
        fp.write(data)
        fp.flush()
        os.fsync(fp.fileno())
    os.chmod(path, mode)


def cmd_run() -> int:
    bad = check_pins()
    if bad:
        for b in bad:
            print(f"PIN FAILED: {b}")
        return 1
    existing = [str(x) for x in (OUT_TSV, OUT_JSON, OUT_SHA, OUT_DIR) if x.exists()]
    if existing:
        print(f"refusing: outputs exist {existing} (use verify)")
        return 1
    cands = json.loads(guard(CANDS).read_text())
    cj = sorted((j for j in cands["new_jobs"] if str(j.get("id")) in C_IDS), key=lambda j: j["id"])
    if [j["id"] for j in cj] != list(C_IDS):
        print(f"refusing: candidates B jobs {[j['id'] for j in cj]} != {list(C_IDS)}")
        return 1
    table, tracks = prereg_c_table(), track_records()
    plans, problems = [], []
    for nj in cj:
        rec, p = plan_job(nj, table, tracks)
        plans.append((nj, rec))
        problems += p
    if problems:
        for x in problems:
            print(f"REFUSED: {x}")
        return 1
    # relocate in memory first (all four), then write, then verify from disk
    blobs = []
    for nj, rec in plans:
        raw = guard(Path(rec["original_copy"])).read_bytes()
        if sha256_bytes(raw) != rec["original_sha256"]:
            print(f"REFUSED: {rec['job']}: original changed while reading")
            return 1
        new, info = relocate_bytes(raw, rec["new_parquet_path"], rec["old_parquet_path"])
        blobs.append((nj, rec, new, info))
    rows = []
    for nj, rec, new, info in blobs:
        out = Path(rec["relocated_file"])
        write_new(out, new)
        v = verify_pair(Path(rec["original_copy"]), out, rec["old_parquet_path"], rec["new_parquet_path"])
        after = sha256_file(rec["original_copy"])
        rsha = sha256_file(out)
        rec.update(relocated_sha256=rsha, fields_rewritten="user.resolved_config_json:" + ",".join(FIELDS),
                   n_tensors=v["n_tensors"], tensors_identical=v["tensors_identical"],
                   safe_open_identical=v["safe_open_identical"], data_block_sha256=v["data_block_sha256"],
                   other_metadata_identical=v["other_metadata_identical"], config_diff_paths=",".join(v["config_diff_paths"]),
                   header_len_old=v["header_len_old"], header_len_new=v["header_len_new"],
                   original_unchanged_after=(after == rec["original_sha256"]), written_at=now(),
                   status="ok" if v["ok"] and after == rec["original_sha256"] and rsha == sha256_bytes(new) else "FAILED",
                   problems=v["problems"])  # fmt: skip
        rows.append(rec)
        print(f"{rec['id']} {rec['job']}: {rec['status']} original {rec['original_sha256'][:16]} -> relocated {rsha[:16]}; "
              f"{v['n_tensors']} tensors identical={v['tensors_identical']} safe_open={v['safe_open_identical']}; "
              f"config diff {v['config_diff_paths']}; parquet {rec['parquet_sha256_new_path'][:16]} ({rec['parquet_sha_matches']})")
        for x in v["problems"]:
            print(f"    PROBLEM: {x}")
    ok = all(r["status"] == "ok" for r in rows)
    payload = {"written_at": now(), "host": socket.gethostname(), "tool": str(Path(__file__).resolve()),
               "tool_sha256": sha256_file(Path(__file__).resolve()), "candidates_r5_sha256": CANDS_SHA,
               "tracks_r5_sha256": TRACKS_SHA, "snapshot_r5_sha256": SNAP_SHA,
               "rule": "as round 2 (AMENDMENTS_R2 A1): only __metadata__['user.resolved_config_json'] task.parquet and "
                       "reference.parquet change (as-trained 5090 /mnt/gfs/... path -> pro6000 path of the same G2 / K3fix parquet); "
                       "weights and every other byte of metadata unchanged; new files only (mode 444)",
               "status": "ok" if ok else "FAILED", "rows": rows}  # fmt: skip
    if not ok:
        write_new(ROOT / f"RELOCATION_R5_FAILED_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                  (json.dumps(payload, indent=1) + "\n").encode(), 0o644)
        print("FAILED: no RELOCATION_R5.tsv written")
        return 1
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=TSV_COLS, delimiter="\t", lineterminator="\n", extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    write_new(OUT_TSV, buf.getvalue().encode())
    write_new(OUT_JSON, (json.dumps(payload, indent=1) + "\n").encode())
    lines = [f"{sha256_file(OUT_TSV)}  RELOCATION_R5.tsv", f"{sha256_file(OUT_JSON)}  RELOCATION_R5.json"]
    lines += [f"{r['relocated_sha256']}  {Path(r['relocated_file']).relative_to(ROOT)}" for r in rows]
    write_new(OUT_SHA, ("\n".join(lines) + "\n").encode())
    print(f"wrote {OUT_TSV.name}, {OUT_JSON.name}, {OUT_SHA.name}")
    return 0


def load_tsv() -> list[dict]:
    with guard(OUT_TSV).open(newline="") as fp:
        return list(csv.DictReader(fp, delimiter="\t"))


def cmd_verify() -> int:
    bad = check_pins()
    for b in bad:
        print(f"PIN FAILED: {b}")
    want = {}
    for line in guard(OUT_SHA).read_text().splitlines():
        f = line.split(None, 1)
        if len(f) == 2:
            want[f[1].strip()] = f[0]
    for rel, sha in want.items():
        got = sha256_file(ROOT / rel) if (ROOT / rel).is_file() else "missing"
        if got != sha:
            bad.append(f"{rel}: {got[:16]} != {sha[:16]}")
    rows = load_tsv()
    if sorted(r["id"] for r in rows) != list(C_IDS):
        bad.append(f"RELOCATION_R5.tsv ids {[r['id'] for r in rows]}")
    for r in rows:
        p = verify_record(r)
        print(f"{r['id']} {r['job']}: {'ok' if not p else 'FAILED'} original {r['original_sha256'][:16]} relocated "
              f"{r['relocated_sha256'][:16]}" + "".join(f"\n    {x}" for x in p))
        bad += p
    print("verify: OK" if not bad else f"verify: {len(bad)} problems: {bad}")
    return 0 if not bad else 1


def _synthetic(cfg: dict, extra_meta: dict) -> bytes:
    import numpy as np

    arrays = {"state.0001": np.arange(6, dtype=np.float32).reshape(2, 3), "state.0002": np.array([7, -8], dtype=np.int32),
              "state.0003": np.array([1, 2, 3], dtype=np.uint32)}  # fmt: skip
    code = {np.dtype(np.float32): "F32", np.dtype(np.int32): "I32", np.dtype(np.uint32): "U32"}
    md = {"format": "x", META_KEY: json.dumps(cfg, sort_keys=True, separators=(",", ":")), **extra_meta}
    h, data, off = {"__metadata__": md}, b"", 0
    for k, a in arrays.items():
        b = a.tobytes()
        h[k] = {"dtype": code[a.dtype], "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        data += b
        off += len(b)
    hb = json.dumps(h, separators=(",", ":")).encode()
    hb += b" " * ((-len(hb)) % 8)
    return struct.pack("<Q", len(hb)) + hb + data


def cmd_selftest() -> int:
    d = ROOT / "tmp" / f"relocate_selftest_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    d.mkdir(parents=True, exist_ok=False)
    old, new = "/mnt/x/workspace/data/q/whole_body/soma/sequence_id=s/robot_name=g1/data.parquet", "/root/y/data.parquet"
    cfg = {"task": {"parquet": old, "motion_speed": 1.0}, "reference": {"parquet": old, "num_frames": 9}, "output_dir": "/mnt/x/run"}
    raw = _synthetic(cfg, {"user.iteration": "5"})
    a = d / "orig.safetensors"
    write_new(a, raw, 0o644)
    reloc, _ = relocate_bytes(raw, new, old)
    b = d / "reloc.safetensors"
    write_new(b, reloc, 0o644)
    results = {"clean relocation passes": verify_pair(a, b, old, new)["ok"]}
    n, hb, h, data = split_st(reloc)
    bad1 = reloc[:-1] + bytes([reloc[-1] ^ 1])  # one tensor byte flipped
    c1 = d / "bad_tensor.safetensors"
    write_new(c1, bad1, 0o644)
    results["flipped tensor byte is caught"] = not verify_pair(a, c1, old, new)["ok"]
    cfg2 = json.loads(json.dumps(cfg))
    cfg2["output_dir"] = "/root/elsewhere"
    raw2 = _synthetic(cfg2, {"user.iteration": "5"})
    rel2, _ = relocate_bytes(raw2, new, old)
    c2 = d / "bad_other_field.safetensors"
    write_new(c2, rel2, 0o644)
    results["changed output_dir is caught"] = not verify_pair(a, c2, old, new)["ok"]
    raw3 = _synthetic(cfg, {"user.iteration": "6"})
    rel3, _ = relocate_bytes(raw3, new, old)
    c3 = d / "bad_other_key.safetensors"
    write_new(c3, rel3, 0o644)
    results["changed other metadata key is caught"] = not verify_pair(a, c3, old, new)["ok"]
    try:
        relocate_bytes(raw, new, "/not/the/old/path")
        results["wrong expected old path refused"] = False
    except ValueError:
        results["wrong expected old path refused"] = True
    results["relocated header 8-aligned and data block identical"] = (n % 8 == 0) and data == split_st(raw)[3]
    for k, v in results.items():
        print(f"selftest: {k}: {'PASS' if v else 'FAIL'}")
    ok = all(results.values())
    print(f"selftest: {'OK' if ok else 'FAILED'} ({d})")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("selftest", "run", "verify"))
    a = ap.parse_args()
    return {"selftest": cmd_selftest, "run": cmd_run, "verify": cmd_verify}[a.cmd]()


if __name__ == "__main__":
    raise SystemExit(main())
