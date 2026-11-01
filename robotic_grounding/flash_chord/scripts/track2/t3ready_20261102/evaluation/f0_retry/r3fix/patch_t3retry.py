import os, sys
from pathlib import Path
F = Path("/root/autodl-tmp/haoran.zhang/tmp_analysis/tier3/f0_retry/t3retry.py")
src = F.read_text()
R = []
R.append(('''  own-NUMA cores), under the shared GPU-choice locks (read-only), one start per pass, >= 180 s apart, episode order.
''', '''  own-NUMA cores), under the shared GPU-choice locks (read-only), one start per pass, >= 180 s apart, episode order.
  v2 (2026-10-26, review R3): RETRY_GO must also carry 'tier3_board_scores_seen: none' and no template placeholder; it is
  refused if RETRY_NOGO exists, if it is not older than fc_runs/submissions/TIER3_FIRST_UPLOAD, or unless
  t3_f0_final/AMENDMENT_1.md is a byte copy of the staged AMENDMENT_1_t3_f0_final.md (the route of a REPLACE verdict into
  the packed file). Before a clip's first start its hydra config check must pass (cfgcheck/epNN.json, write-once).
  Recipes frozen later are written read-only and logged in recipes.sha256. SUBMISSION_CHOICE records the chosen retry
  run (closest to the mean own-input ADD-AUC of the ok retry runs, exact; ties -> lower run) for a REPLACE episode.
'''))
R.append(('''from pathlib import Path

sys.dont_write_bytecode = True
''', '''from fractions import Fraction
from pathlib import Path

sys.dont_write_bytecode = True
'''))
R.append(('''GO, STOPF, PIDS, LOG, HB = RA / "RETRY_GO", RA / "STOP", RA / "pids.tsv", RA / "driver.log", RA / "heartbeat.log"
''', '''GO, STOPF, PIDS, LOG, HB = RA / "RETRY_GO", RA / "STOP", RA / "pids.tsv", RA / "driver.log", RA / "heartbeat.log"
NOGO = RA / "RETRY_NOGO"  # v2: the user declined the retry queue
FINAL_ROOT = ME / "fc_runs/submissions/t3_f0_final"
AMEND_STAGED = RA / "AMENDMENT_1_t3_f0_final.md"  # v2: pinned in RETRY_FILES.sha256 (R3 must-fix 1)
AMEND_FINAL, AMEND_FINAL_SHA = FINAL_ROOT / "AMENDMENT_1.md", FINAL_ROOT / "AMENDMENT_1.sha256"
TIER3_FIRST_UPLOAD = ME / "fc_runs/submissions/TIER3_FIRST_UPLOAD"  # v2: marker the main session writes after our 1st Tier-3 upload
RECIPE_LOG, CFGD = RA / "recipes.sha256", RA / "cfgcheck"  # v2: later recipes' sha256 log; write-once pre-launch cfgchecks
'''))
R.append(('''def go_status() -> tuple[bool, str]:
    if not GO.is_file():
        return False, "RETRY_GO absent (nothing starts until the main session writes it after the user's approval)"
    text = GO.read_text()
    ok_sha, _ = prereg_status()
    try:
        want = PREREG_SHA.read_text().split()[0]
    except (OSError, IndexError):
        want = "?"
    if f"prereg_sha256 {want}" not in text:
        return False, "RETRY_GO does not name the current PREREG_retry sha256 (line 'prereg_sha256 <sha>')"
    if not re.search(r"^user_approval: .+", text, re.M):
        return False, "RETRY_GO lacks a 'user_approval: <verbatim quote, date>' line"
    return ok_sha, "RETRY_GO present, names this PREREG and the user's approval"
''', '''def go_status() -> tuple[bool, str]:
    if NOGO.exists():
        return False, "RETRY_NOGO present: the user declined the retry queue (nothing starts)"
    if not GO.is_file():
        return False, "RETRY_GO absent (nothing starts until the main session writes it after the user's approval)"
    text = GO.read_text()
    ok_sha, _ = prereg_status()
    try:
        want = PREREG_SHA.read_text().split()[0]
    except (OSError, IndexError):
        want = "?"
    if not re.search(rf"^prereg_sha256 {re.escape(want)}\\s*$", text, re.M):
        return False, "RETRY_GO does not name the current PREREG_retry sha256 (line 'prereg_sha256 <sha>')"
    m = re.search(r"^user_approval: (.+)$", text, re.M)
    if not m or "<" in m.group(1) or ">" in m.group(1) or not re.match(r'^".+", \\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}', m.group(1)):
        return False, "RETRY_GO lacks a filled 'user_approval: \\"<verbatim>\\", YYYY-MM-DD HH:MM CST' line (template text refused)"
    if not re.search(r"^tier3_board_scores_seen: none\\s*$", text, re.M):
        return False, "RETRY_GO lacks the line 'tier3_board_scores_seen: none' (the GO must predate any Tier-3 board score)"
    if TIER3_FIRST_UPLOAD.exists() and GO.stat().st_mtime >= TIER3_FIRST_UPLOAD.stat().st_mtime:
        return False, "RETRY_GO is not older than fc_runs/submissions/TIER3_FIRST_UPLOAD: refused (written after a Tier-3 upload)"
    staged = sha256(AMEND_STAGED)
    if staged is None or sha256(AMEND_FINAL) != staged:
        return False, ("t3_f0_final/AMENDMENT_1.md is not installed as a byte copy of f0_retry/AMENDMENT_1_t3_f0_final.md "
                       "(R3 must-fix 1: the route of a REPLACE verdict into the packed file)")
    try:
        named = AMEND_FINAL_SHA.read_text().split()[0]
    except (OSError, IndexError):
        named = ""
    if named != staged:
        return False, "t3_f0_final/AMENDMENT_1.sha256 missing or does not name AMENDMENT_1.md's sha256"
    return ok_sha, "RETRY_GO present: names this PREREG, the user's approval, no Tier-3 board score seen; AMENDMENT_1 installed"
'''))
R.append(('''    if write:
        atomic_write(path, json.dumps(rec, indent=1))
        log(f"ep{ep2(ep)} recipe frozen: {rec['termination_overrides'] or 'no termination override'} ({path})")
    return rec
''', '''    if write:
        atomic_write(path, json.dumps(rec, indent=1))
        if where == RECIPES:  # v2 (R3): a recipe frozen later is read-only and logged (append-only)
            os.chmod(path, 0o444)
            with RECIPE_LOG.open("a") as f:
                f.write(f"# {now()}\\n{sha256(path)}  recipes/{path.name}\\n")
        log(f"ep{ep2(ep)} recipe frozen: {rec['termination_overrides'] or 'no termination override'} ({path})")
    return rec
'''))
R.append(('''            res[ep2(ep)] = {"ok": all(checks.values()), "checks": checks, "f0_hydra_config": str(hyd / "config.yaml"), "diff": diff}
    return res
''', '''            res[ep2(ep)] = {"ok": all(checks.values()), "checks": checks, "f0_hydra_config": str(hyd / "config.yaml"), "diff": diff}
    return res


def cfg_gate(ep: int, rec: dict) -> tuple[bool, str]:
    """v2 (R3): the hydra config check of a clip's live recipe before its first start, recorded write-once in
    cfgcheck/epNN.json. A recorded FAIL holds the clip for good (a human decides); an exception is not recorded."""
    p = CFGD / f"ep{ep2(ep)}.json"
    rsha = sha256(RECIPES / f"ep{ep2(ep)}.json")
    if p.is_file():
        r = jload(p)
    else:
        try:
            res = cfgcheck([ep], {ep: rec})[ep2(ep)]
        except Exception as error:  # noqa: BLE001
            return False, f"cfgcheck error (not recorded, retried next pass): {type(error).__name__}: {error}"
        r = {"written": now(), "recipe_sha256": rsha, **res}
        atomic_write(p, json.dumps(r, indent=1, default=str))
        os.chmod(p, 0o444)
        log(f"ep{ep2(ep)} cfgcheck {'OK' if r.get('ok') else 'FAIL'} recorded ({p})")
    if r.get("recipe_sha256") != rsha:
        return False, "recipe differs from the one its cfgcheck checked: held (a human decides)"
    if not r.get("ok"):
        return False, f"cfgcheck FAIL recorded: held (a human decides): {r.get('checks') or r.get('why')}"
    return True, "cfgcheck OK"
'''))
R.append(('''def verdict(write: bool) -> int:
''', '''def closest_run(rows: list[dict]) -> dict | None:
    """t3_f0_final PREREG section 3 rule on the retry arm (AMENDMENT_1 (b)): among the ok runs, the run whose own-input
    ADD-AUC is closest to their mean, exact rational arithmetic on the float64 values; ties -> lower run index."""
    ok = [r for r in rows if r.get("ok") and isinstance(r.get("auc"), (int, float))]
    if not ok:
        return None
    vals = [(r, Fraction(float(r["auc"]))) for r in ok]
    mean = sum((v for _, v in vals), Fraction(0)) / len(vals)
    return min(vals, key=lambda rv: (abs(rv[1] - mean), int(rv[0]["run"])))[0]


def verdict(write: bool) -> int:
'''))
R.append(('''            if st["decision"].startswith("REPLACE"):
                use = {"use": "retry_final", "checkpoint": st["retry_final"], "sha256": st["retry_final_sha256"], "why": st["decision"]}
''', '''            if st["decision"].startswith("REPLACE"):
                use = {"use": "retry_final", "checkpoint": st["retry_final"], "sha256": st["retry_final_sha256"], "why": st["decision"]}
                best = closest_run(st["runs"]["retry"])
                if best is not None:
                    w = EVR / f"ep{ep2(ep)}" / "retry" / f"r{best['run']}_tol{float(best['tolerance']):g}_try{best['attempt']}"
                    js, pq = w / "results" / f"episode_{ep:06d}.json", w / "results" / f"episode_{ep:06d}.parquet"
                    ev = best.get("evaluated_checkpoint")
                    use["chosen_run"] = {"rule": "AMENDMENT_1 (b): closest to the mean own-input ADD-AUC of the ok retry runs "
                                                 "(exact rational arithmetic); ties -> lower run index",
                                         "run": best["run"], "dir": str(w), "auc": best["auc"],
                                         "results_json": str(js), "results_json_sha256": sha256(js),
                                         "results_parquet": str(pq), "results_parquet_sha256": sha256(pq),
                                         "evaluated_checkpoint": ev, "evaluated_checkpoint_sha256": sha256(ev) if ev else None}
'''))
R.append(('''    last_beat, last_launch, posts = 0.0, 0.0, {}
''', '''    last_beat, last_launch, posts = 0.0, 0.0, {}
    cfg_said: dict = {}
'''))
R.append(('''                    log(f"ep{ep2(ep)} LATE: projected end after 2026-11-02 12:00; not started (f0 final stays)")
                    continue
                line = T().plan(ep, "")
''', '''                    log(f"ep{ep2(ep)} LATE: projected end after 2026-11-02 12:00; not started (f0 final stays)")
                    continue
                okc, whyc = cfg_gate(ep, recs[ep])  # v2 (R3): hydra config check before the first start
                if not okc:
                    if cfg_said.get(ep) != whyc:
                        cfg_said[ep] = whyc
                        log(f"ep{ep2(ep)} not started: {whyc}")
                    continue
                line = T().plan(ep, "")
'''))
for old, new in R:
    n = src.count(old)
    if n != 1:
        sys.exit(f"anchor count {n} != 1 for: {old[:90]!r}")
    src = src.replace(old, new)
compile(src, str(F), "exec")
os.chmod(F, 0o644)
F.write_text(src)
os.chmod(F, 0o444)
print("patched", F, len(R), "replacements")
