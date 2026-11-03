"""test_v3.py: sandbox tests of t3retry_v3.py (go_status gates, closest_run, verdict chosen_run, cfg_gate) plus the v3
changes (allowlist on the module and on the helper object, cap, frozen list, launch refusal on GPUs 3/4, hash files).
Writes only under f0_retry/v3fix/sandbox. Must run after PREREG_v3.sha256 / RETRY_FILES_v3.sha256 exist and before the
real RETRY_GO is written."""
import importlib.util, json, os, shutil, sys, time
from pathlib import Path
RA = Path("/root/autodl-tmp/haoran.zhang/tmp_analysis/tier3/f0_retry")
SB = RA / "v3fix/sandbox"
shutil.rmtree(SB, ignore_errors=True); SB.mkdir(parents=True)
spec = importlib.util.spec_from_file_location("t3r3", RA / "t3retry_v3.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
real = {k: getattr(m, k) for k in ("GO", "NOGO")}
assert not real["GO"].exists() and not real["NOGO"].exists(), "real RETRY_GO/NOGO must not exist"
res = []
def ok(name, cond, info=""):
    res.append(bool(cond)); print(("PASS" if cond else "FAIL"), name, info)
# v3 specifics (real files, read-only)
ok("v3 PREREG files", m.PREREG.name == "PREREG_v3.md" and m.PREREG_SHA.name == "PREREG_v3.sha256" and m.FILES_SHA.name == "RETRY_FILES_v3.sha256")
ok("prereg_status", m.prereg_status()[0], m.prereg_status()[1])
ok("files_status", m.files_status()[0], m.files_status()[1][:200])
ok("allowlist", tuple(m.ALLOWED_GPUS) == (0, 1, 2, 5, 6, 7), m.ALLOWED_GPUS)
T = m.T()
ok("helper object allowlist", tuple(T.ALLOWED_GPUS) == (0, 1, 2, 5, 6, 7) and T.MAX_TRAIN_PER_GPU == 3 == m.MAX_TRAIN_PER_GPU,
   (T.ALLOWED_GPUS, T.MAX_TRAIN_PER_GPU))
ok("frozen list", tuple(m.FROZEN_CANDIDATES) == (1, 2, 4, 5, 11, 13, 14, 16, 17, 19, 21, 23, 25, 28))
cands, _ = m.candidates()
ok("rule output == frozen list", sorted(cands) == list(m.FROZEN_CANDIDATES), cands)
ok("14 recipes frozen read-only", all((m.RECIPES / f"ep{e:02d}.json").is_file() and ((m.RECIPES / f"ep{e:02d}.json").stat().st_mode & 0o222) == 0
                                     for e in m.FROZEN_CANDIDATES))
for g in (3, 4):
    try:
        m.launch(1, {}, g, "0", 1); ok(f"launch gpu{g} refused", False)
    except RuntimeError as e:
        ok(f"launch gpu{g} refused", "not allowed" in str(e), str(e))
line = T.plan(1, "", assume={5: 3, 6: 3, 7: 2, 0: 3, 1: 3, 2: 3})
ok("plan with 0 1 2 5 6 7 full -> WAIT (never 3/4)", line.startswith("WAIT"), line[:200])
# go_status gates (sandbox copies of the control files)
m.GO, m.NOGO = SB / "RETRY_GO", SB / "RETRY_NOGO"
m.AMEND_FINAL, m.AMEND_FINAL_SHA = SB / "final/AMENDMENT_1.md", SB / "final/AMENDMENT_1.sha256"
m.TIER3_FIRST_UPLOAD, m.RA, m.EVR, m.CFGD, m.LOG = SB / "TIER3_FIRST_UPLOAD", SB, SB / "eval", SB / "cfgcheck", SB / "driver.log"
sha = m.PREREG_SHA.read_text().split()[0]
def chk(name, want_ok, want_sub=""):
    o, why = m.go_status()
    ok(name, (o == want_ok) and (want_sub in why), f"-> {o} {why}")
chk("no GO", False, "absent")
tpl = (RA / "RETRY_GO.template").read_text()
m.GO.write_text(tpl); chk("template verbatim", False, "sha256")
v2sha = (RA / "PREREG_retry.sha256").read_text().split()[0]
good_tail = 'user_approval: "yes go ahead", 2026-10-31 00:20 CST\ntier3_board_scores_seen: none\nwritten_by: test\n'
m.GO.write_text(f"prereg_sha256 {v2sha}\n" + good_tail); chk("GO naming the v2 PREREG sha refused", False, "sha256")
m.GO.write_text(f"prereg_sha256 {sha}\n" + 'user_approval: "<the user>", <YYYY-MM-DD HH:MM> CST\ntier3_board_scores_seen: none\n'); chk("placeholder approval", False, "user_approval")
m.GO.write_text(f"prereg_sha256 {sha}\n" + good_tail.replace("tier3_board_scores_seen: none\n", "")); chk("missing board line", False, "tier3_board_scores_seen")
m.GO.write_text(f"prereg_sha256 {sha}\n" + good_tail); chk("amendment not installed", False, "AMENDMENT_1.md is not installed")
(SB / "final").mkdir(); shutil.copyfile(m.AMEND_STAGED, m.AMEND_FINAL)
chk("amendment installed, no sha file", False, "AMENDMENT_1.sha256")
m.AMEND_FINAL_SHA.write_text(f"{m.sha256(m.AMEND_FINAL)}  AMENDMENT_1.md\n"); chk("all ok", True, "AMENDMENT_1 installed")
m.AMEND_FINAL.write_text(m.AMEND_STAGED.read_text() + "x"); chk("amendment altered", False, "not installed"); shutil.copyfile(m.AMEND_STAGED, m.AMEND_FINAL)
m.TIER3_FIRST_UPLOAD.write_text("x"); os.utime(m.TIER3_FIRST_UPLOAD, (time.time() - 3600, time.time() - 3600)); chk("GO newer than upload marker", False, "TIER3_FIRST_UPLOAD")
m.TIER3_FIRST_UPLOAD.unlink()
m.NOGO.write_text("x"); chk("NOGO present", False, "RETRY_NOGO"); m.NOGO.unlink()
# closest_run (unchanged rule)
def run(k, auc, okk=True): return {"run": k, "auc": auc, "ok": okk}
for rows, want in [([run(1, .5), run(2, .6), run(3, .7)], 2), ([run(1, .5), run(2, .7)], 1), ([run(1, .5, False), run(2, .6), run(3, .7)], 2),
                   ([run(1, .1), run(2, .2), run(3, .3), run(4, .4), run(5, 1.0)], 4), ([run(1, .3, False)], None), ([run(3, .2), run(1, .2)], 1)]:
    got = m.closest_run(rows); got = got["run"] if got else None
    ok("closest_run", got == want, f"{[(r['run'], r['auc'], r['ok']) for r in rows]} -> {got} want {want}")
# decide (unchanged acceptance; MPPE not used)
mk = lambda k, auc, sp, mp, mppe=1.0: {"run": k, "auc": auc, "sp_sr": sp, "mp_sr": mp, "ok": True, "mppe_cm": mppe}
f0 = [mk(k, .30, 0, 0) for k in range(1, 6)]
ok("decide (i)", m.decide({"f0final": f0, "retry": [mk(k, .31, 1, 1, 99.0) for k in range(1, 5)] + [mk(5, .31, 0, 0)]}).startswith("REPLACE (i)"))
ok("decide (ii)", m.decide({"f0final": f0, "retry": [mk(k, .32, 0, 0) for k in range(1, 6)]}).startswith("REPLACE (ii)"))
ok("decide keep", m.decide({"f0final": f0, "retry": [mk(k, .305, 0, 0) for k in range(1, 6)]}).startswith("KEEP_F0"))
ok("decide incomplete", m.decide({"f0final": f0, "retry": [mk(k, .9, 1, 1) for k in range(1, 5)]}).startswith("KEEP_F0 (incomplete"))
# cfg_gate on a real frozen v3 recipe (record in the sandbox)
rec = m.recipe(13, write=False)
o1, w1 = m.cfg_gate(13, rec); o2, w2 = m.cfg_gate(13, rec)
ok("cfg_gate ep13 (fresh + cached)", o1 and o2 and oct((SB / "cfgcheck/ep13.json").stat().st_mode & 0o777) == "0o444", f"{w1} | {w2}")
assert not real["GO"].exists() and not real["NOGO"].exists(), "real RETRY_GO/NOGO appeared"
print(f"TOTAL {sum(res)}/{len(res)} PASS")
sys.exit(0 if all(res) else 1)
