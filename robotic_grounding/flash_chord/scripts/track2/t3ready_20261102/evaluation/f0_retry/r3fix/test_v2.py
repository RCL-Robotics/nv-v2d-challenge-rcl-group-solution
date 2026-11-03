import importlib.util, json, os, shutil, sys, time
from pathlib import Path
RA = Path("/root/autodl-tmp/haoran.zhang/tmp_analysis/tier3/f0_retry")
SB = RA / "r3fix/sandbox"
shutil.rmtree(SB, ignore_errors=True); SB.mkdir(parents=True)
spec = importlib.util.spec_from_file_location("t3r", RA / "t3retry.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
real = {k: getattr(m, k) for k in ("GO", "NOGO", "AMEND_FINAL", "AMEND_FINAL_SHA", "TIER3_FIRST_UPLOAD", "RA", "EVR", "CFGD", "LOG")}
assert not real["GO"].exists() and not real["NOGO"].exists(), "real RETRY_GO/NOGO must not exist"
m.GO, m.NOGO = SB / "RETRY_GO", SB / "RETRY_NOGO"
m.AMEND_FINAL, m.AMEND_FINAL_SHA = SB / "final/AMENDMENT_1.md", SB / "final/AMENDMENT_1.sha256"
m.TIER3_FIRST_UPLOAD, m.RA, m.EVR, m.CFGD, m.LOG = SB / "TIER3_FIRST_UPLOAD", SB, SB / "eval", SB / "cfgcheck", SB / "driver.log"
sha = m.PREREG_SHA.read_text().split()[0]
res = []
def chk(name, want_ok, want_sub=""):
    ok, why = m.go_status()
    good = (ok == want_ok) and (want_sub in why)
    res.append(good); print(("PASS" if good else "FAIL"), name, "->", ok, why)
chk("no GO", False, "absent")
tpl = (RA / "RETRY_GO.template").read_text()
m.GO.write_text(tpl); chk("template verbatim", False, "sha256")
m.GO.write_text(tpl.replace("prereg_sha256 <sha256 of PREREG_retry.md, as in PREREG_retry.sha256>", f"prereg_sha256 {sha}")); chk("template with sha, placeholder approval", False, "user_approval")
good_go = f'prereg_sha256 {sha}\nuser_approval: "yes go ahead with the retries", 2026-10-27 09:00 CST\ntier3_board_scores_seen: none\nwritten_by: test\n'
m.GO.write_text(good_go.replace("tier3_board_scores_seen: none\n", "")); chk("missing board line", False, "tier3_board_scores_seen")
m.GO.write_text(f'prereg_sha256 {"0"*64}\n' + good_go.split("\n", 1)[1]); chk("wrong prereg sha", False, "sha256")
m.GO.write_text(good_go); chk("amendment not installed", False, "AMENDMENT_1.md is not installed")
(SB / "final").mkdir(); shutil.copyfile(m.AMEND_STAGED, m.AMEND_FINAL)
chk("amendment installed, no sha file", False, "AMENDMENT_1.sha256")
m.AMEND_FINAL_SHA.write_text(f"{m.sha256(m.AMEND_FINAL)}  AMENDMENT_1.md\n"); chk("all ok", True, "AMENDMENT_1 installed")
m.AMEND_FINAL.write_text(m.AMEND_STAGED.read_text() + "x"); chk("amendment altered", False, "not installed"); shutil.copyfile(m.AMEND_STAGED, m.AMEND_FINAL)
m.TIER3_FIRST_UPLOAD.write_text("x"); os.utime(m.TIER3_FIRST_UPLOAD, (time.time() - 3600, time.time() - 3600)); chk("GO newer than upload marker", False, "TIER3_FIRST_UPLOAD")
os.utime(m.TIER3_FIRST_UPLOAD, (time.time() + 3600, time.time() + 3600)); chk("GO older than upload marker", True)
m.TIER3_FIRST_UPLOAD.unlink()
m.NOGO.write_text("x"); chk("NOGO present", False, "RETRY_NOGO"); m.NOGO.unlink()
# closest_run
def run(k, auc, ok=True): return {"run": k, "auc": auc, "ok": ok}
cases = [([run(1, .5), run(2, .6), run(3, .7)], 2), ([run(1, .5), run(2, .7)], 1), ([run(1, .5, False), run(2, .6), run(3, .7)], 2),
         ([run(1, .1), run(2, .2), run(3, .3), run(4, .4), run(5, 1.0)], 4), ([run(1, .3, False)], None), ([run(3, .2), run(1, .2)], 1),
         ([run(1, 0.1), run(2, 0.2), run(3, 0.30000000000000004)], 2)]
for rows, want in cases:
    got = m.closest_run(rows); got = got["run"] if got else None
    res.append(got == want); print(("PASS" if got == want else "FAIL"), "closest_run", [(r["run"], r["auc"], r["ok"]) for r in rows], "->", got, "want", want)
# verdict(write=True) in the sandbox: chosen_run recorded for a REPLACE episode
ep = 1
w = SB / "eval/ep01/retry/r3_tol0.0001_try1"; (w / "results").mkdir(parents=True)
(w / "results/episode_000001.json").write_text("{}"); (w / "results/episode_000001.parquet").write_bytes(b"pq")
evf = w / "ep01_export.safetensors"; evf.write_bytes(b"ev")
mk = lambda k, auc, ok=True, tol=0.0001: {"arm": "retry", "run": k, "auc": auc, "sp_sr": 1, "mp_sr": 1, "ok": ok, "tolerance": tol, "attempt": 1, "evaluated_checkpoint": str(evf) if k == 3 else None}
post = {"status": "done", "decision": "REPLACE (i): test", "retry_final": "x", "retry_final_sha256": "y",
        "runs": {"f0final": [mk(k, .2) for k in range(1, 6)], "retry": [mk(1, .80), mk(2, .90), mk(3, .84), mk(4, .86), mk(5, .80)]}}
(SB / "eval/ep01").mkdir(parents=True, exist_ok=True); (SB / "eval/ep01/post.json").write_text(json.dumps(post))
rc = m.verdict(write=True)
sc = json.loads((SB / "SUBMISSION_CHOICE.json").read_text())["episodes"]
cr = sc["01"].get("chosen_run", {})
good = rc == 0 and sc["01"]["use"] == "retry_final" and cr.get("run") == 3 and cr.get("results_json_sha256") == m.sha256(w / "results/episode_000001.json") and cr.get("evaluated_checkpoint_sha256") == m.sha256(evf) and sc["00"]["use"] == "prereg_f0"
res.append(good); print(("PASS" if good else "FAIL"), "verdict chosen_run", json.dumps(cr)[:300])
# cfg_gate on the real frozen recipe of ep01 (record into the sandbox)
rec = m.recipe(1, write=False)
ok1, why1 = m.cfg_gate(1, rec); ok2, why2 = m.cfg_gate(1, rec)
good = ok1 and ok2 and (SB / "cfgcheck/ep01.json").is_file() and oct((SB / "cfgcheck/ep01.json").stat().st_mode & 0o777) == "0o444"
res.append(good); print(("PASS" if good else "FAIL"), "cfg_gate ep01 (fresh + cached):", why1, "|", why2)
for k, v in real.items():
    assert not (k in ("GO", "NOGO") and v.exists()), "real RETRY_GO/NOGO appeared"
print(f"TOTAL {sum(res)}/{len(res)} PASS")
