#!/usr/bin/env python3
"""r8_driver.py: the round-8 orchestrator (fc_runs/submissions/t2_consensus_r8_20261103; PREREG_R8.md section 7).

Steps (each resumable; a finished step is recognised by its output and never repeated):
  1. wait until T_c = 2026-11-03 12:00:00 CST (sleeping; heartbeat every 60 s);
  2. freeze the pool: r8_pool.py freeze (POOL_R8.json + .sha256, copies, relocations; read-only), then r8_pool.py verify;
  3. phase S: r8_eval.py run --phase screen (exports on CPU at once; each evaluation starts only on GPU 0 / 4 / 6 / 7
     (AMENDMENTS_R8 A1; before A1: 6 / 7) while no other compute process is listed there; it waits and logs otherwise);
     then r8_select.py check / gate / score /
     select --phase S (SELECTION_S8.sha256 last);
  4. phase F (only if final_r8.json names a finalist): r8_eval.py run --phase final; then r8_select.py check / gate /
     score / select --phase F (SELECTION_R8.json, selection_r8_episodes.csv, SELECTION_R8.sha256 last). Without a
     finalist: select --phase F directly (every episode keeps its round-7 pick);
  5. deadlines (PREREG_R8 7.3): phase S not finished by 2026-11-04 08:00 CST -> STOP file, running evaluations finish,
     then r8_select.py keep-r7 (every episode keeps its round-7 pick); phase F not finished by 2026-11-04 12:00 CST ->
     STOP file + DEADLINE_F.json, running evaluations finish, then the phase-F selection (a finalist without 10 ok
     runs does not replace the round-7 pick).
A STOP file created by anyone else (not by this driver at a deadline) stops this driver at the next step boundary;
nothing is selected then. An evaluation driver that exits early is restarted after 10 minutes (at most 6 times; then
phase S keeps every round-7 pick, phase F halts); a pin / preflight / integrity failure halts (the R7 file stays in
force). No second r8_eval.py is ever started while one is alive (read from /proc), and waiting for one is never counted
as a restart (PREREG_R8 7.4-7.6). Scorer steps whose verified outputs exist are not repeated (r8_select.py gate / score
answer 'already done'). Never commits, pushes, packs or uploads; never signals any process; CPU-only scoring.

  setsid nohup conda_envs/flashchord/bin/python -I -B r8_driver.py run >> r8_driver_stdout.log 2>&1 < /dev/null &
  conda_envs/flashchord/bin/python -I -B r8_driver.py status
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import fcntl  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ME = Path("/root/autodl-tmp/haoran.zhang")
ROOT = ME / "fc_runs/submissions/t2_consensus_r8_20261103"
PY = ME / "conda_envs/flashchord/bin/python"
KPY = ME / "conda_envs/v2dkit/bin/python"
PREREG_SHA = "bb912c57180dde8255776884de169e12467349f57d584782308b88ff935abf8c"
TZ8 = dt.timezone(dt.timedelta(hours=8))
T_C = dt.datetime(2026, 11, 3, 12, 0, 0, tzinfo=TZ8).timestamp()
S_CUT = dt.datetime(2026, 11, 4, 8, 0, 0, tzinfo=TZ8).timestamp()
F_DEADLINE = dt.datetime(2026, 11, 4, 12, 0, 0, tzinfo=TZ8).timestamp()
HB = ROOT / "R8_DRIVER_HEARTBEAT.json"
LOG = ROOT / "r8_driver.log"
STOP = ROOT / "STOP"
DEADLINE_F = ROOT / "DEADLINE_F.json"
S_CUT_FILE = ROOT / "S_CUTOFF.json"
RETRIES, RETRY_WAIT = 6, 600.0
MINE = Path(__file__).resolve()


def now() -> str:
    return dt.datetime.now(TZ8).strftime("%Y-%m-%d %H:%M:%S")


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log(msg: str) -> None:
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as fp:
        fp.write(line + "\n")


def write_atomic(p: Path, text: str) -> None:
    tmp = p.with_name(p.name + ".tmp_r8d")
    tmp.write_text(text)
    os.replace(tmp, p)


STATE = {"step": "start", "detail": ""}


def beat(step: str | None = None, detail: str | None = None) -> None:
    if step is not None:
        STATE["step"] = step
    if detail is not None:
        STATE["detail"] = detail
    hb = {"updated_at": now(), "pid": os.getpid(), **STATE, "t_c": "2026-11-03 12:00:00 CST",
          "phase_s_cutoff": "2026-11-04 08:00:00 CST", "phase_f_deadline": "2026-11-04 12:00:00 CST",
          "stop_file": STOP.exists(), "pool_frozen": (ROOT / "POOL_R8.sha256").is_file(),
          "selection_s8": (ROOT / "SELECTION_S8.sha256").is_file(), "selection_r8": (ROOT / "SELECTION_R8.sha256").is_file()}  # fmt: skip
    for tag in ("s8", "f8"):
        p = ROOT / f"eval_r8_heartbeat_{tag}.json"
        if p.is_file():
            try:
                e = json.loads(p.read_text())
                hb[f"eval_{tag}"] = {k: e.get(k) for k in ("updated_at", "state", "exports", "evals", "eta_hours", "running")}
            except (OSError, ValueError):
                pass
    write_atomic(HB, json.dumps(hb, indent=1) + "\n")


def docs_problems() -> list[str]:
    """PREREG_R8.md == PREREG_R8.sha256 == hard-coded; PREREG_R8_PINS.md == its sha file, pins this script, every pin
    holds; LOCK_R8.sha256 holds."""
    out = []
    pre = ROOT / "PREREG_R8.md"
    got = sha256_file(pre) if pre.is_file() else "missing"
    rec = ((ROOT / "PREREG_R8.sha256").read_text().split() or [""])[0] if (ROOT / "PREREG_R8.sha256").is_file() else ""
    if not got == rec == PREREG_SHA:
        out.append(f"PREREG_R8.md {got[:16]} != sha file {rec[:16]} / hard-coded {PREREG_SHA[:16]}")
    pins = ROOT / "PREREG_R8_PINS.md"
    if not pins.is_file():
        return out + ["PREREG_R8_PINS.md missing"]
    prec = ((ROOT / "PREREG_R8_PINS.sha256").read_text().split() or [""])[0] if (ROOT / "PREREG_R8_PINS.sha256").is_file() else ""
    if sha256_file(pins) != prec:
        out.append("PREREG_R8_PINS.md != PREREG_R8_PINS.sha256")
    plist = [(f[1], f[2]) for f in (ln.split() for ln in pins.read_text().splitlines()) if len(f) == 3 and f[0] == "pin"]
    if ("r8_driver.py", sha256_file(MINE)) not in plist:
        out.append("PREREG_R8_PINS.md does not pin this script")
    for rel, h0 in plist:
        q = Path(rel) if rel.startswith("/") else ROOT / rel
        if (sha256_file(q) if q.is_file() else "missing") != h0:
            out.append(f"pin {rel} changed")
    lock = ROOT / "LOCK_R8.sha256"
    if not lock.is_file():
        out.append("LOCK_R8.sha256 missing")
    else:
        for ln in lock.read_text().splitlines():
            f = ln.split(None, 1)
            if len(f) == 2 and len(f[0]) == 64:
                q = Path(f[1].strip()) if f[1].strip().startswith("/") else ROOT / f[1].strip()
                if (sha256_file(q) if q.is_file() else "missing") != f[0]:
                    out.append(f"LOCK_R8.sha256: {f[1].strip()} changed")
    return out


def child_env(cpu_only: bool) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CUDA_MPS_", "PYTHON")) and k != "CUDA_VISIBLE_DEVICES"}
    env.update(TZ="Asia/Shanghai", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               MKL_NUM_THREADS="2")  # fmt: skip
    if cpu_only:
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def run_step(name: str, cmd: list[str], cpu_only: bool, out_log: str) -> int:
    """A foreground child in its own session (never signalled by us); its output is appended to out_log."""
    log(f"step {name}: {' '.join(cmd)}")
    with (ROOT / out_log).open("a") as fp:
        fp.write(f"\n===== [{now()}] {' '.join(cmd)}\n")
        fp.flush()
        proc = subprocess.Popen(["nice", "-n", "10", *cmd], cwd="/tmp", env=child_env(cpu_only), stdout=fp,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)  # fmt: skip
        while proc.poll() is None:
            beat()
            time.sleep(30)
    log(f"step {name}: exit {proc.returncode}")
    return proc.returncode


def foreign_stop() -> bool:
    """A STOP file this driver did not create at a deadline."""
    return STOP.exists() and not DEADLINE_F.exists() and not S_CUT_FILE.exists()


def live_evals() -> list[str]:
    """Live r8_eval.py run processes (any phase), read from /proc only (nothing is signalled). A plan-only --dry-run, a
    zombie (empty cmdline) and this process do not count (PREREG_R8 7.6)."""
    out = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) == os.getpid():
            continue
        try:
            args = [x.decode(errors="replace") for x in (d / "cmdline").read_bytes().split(b"\0") if x]
        except OSError:
            continue
        if any(x.endswith("r8_eval.py") for x in args) and "run" in args and "--dry-run" not in args:
            out.append(f"pid {d.name}: {' '.join(args)[:200]}")
    return out


def hb_state(p: Path) -> str | None:
    try:
        return json.loads(p.read_text()).get("state") if p.is_file() else None
    except (OSError, ValueError):
        return None


def mark_deadline(phase: str, marker: Path) -> None:
    write_atomic(marker, json.dumps({"written_at": now(), "phase": phase, "rule": "PREREG_R8 7.3",
                                     "note": "deadline reached: STOP file created by r8_driver.py; running "
                                             "evaluations finish and are recorded"}, indent=1) + "\n")  # fmt: skip
    STOP.touch()
    log(f"deadline for phase {phase} reached: STOP file created ({marker.name})")


def eval_phase(phase: str, deadline: float, marker: Path) -> str:
    """Runs r8_eval.py for a phase until its heartbeat says done. Returns 'done', 'deadline', 'stopped', 'halted' (a pin /
    preflight / integrity failure of r8_eval.py, PREREG_R8 7.4) or 'failed' (6 restarts used). Never starts a second
    r8_eval.py while one is alive, and never counts such a wait as a restart (PREREG_R8 7.6)."""
    tag = "s8" if phase == "screen" else "f8"
    hbf = ROOT / f"eval_r8_heartbeat_{tag}.json"
    outf = ROOT / f"eval_r8_driver_stdout_{tag}.log"
    tries = 0
    while True:
        if hb_state(hbf) == "done" and not live_evals():
            return "done"
        if foreign_stop():
            return "stopped"
        if time.time() >= deadline and not marker.exists():
            mark_deadline(phase, marker)
        live = live_evals()
        if marker.exists():
            if not live:
                return "deadline"
            beat(f"phase {phase}: deadline stop, waiting for running evaluations")
            time.sleep(60)
            continue
        if live:
            if STATE.get("waiting_for") != live[0]:
                log(f"phase {phase}: an r8_eval.py started earlier still runs ({live[0]}): waiting for it; no new start, "
                    "no restart counted (PREREG_R8 7.6)")  # fmt: skip
                STATE["waiting_for"] = live[0]
            beat(f"phase {phase}: waiting for a running r8_eval.py started earlier")
            time.sleep(60)
            continue
        STATE.pop("waiting_for", None)
        beat(f"phase {phase}: evaluations")
        off = outf.stat().st_size if outf.is_file() else 0
        with outf.open("a") as fp:
            proc = subprocess.Popen(["nice", "-n", "10", str(PY), "-I", "-B", str(ROOT / "r8_eval.py"), "run", "--phase", phase,
                                     "--gpus", "0,4,6,7", "--max-parallel", "4", "--lane-max", "2", "--cooldown", "10"],
                                    # AMENDMENTS_R8 A1 (was --gpus 6,7 --max-parallel 2)
                                    cwd=str(ROOT), env=child_env(False), stdout=fp, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True)  # fmt: skip
        log(f"phase {phase}: r8_eval.py started (pid {proc.pid})")
        while proc.poll() is None:
            beat()
            if time.time() >= deadline and not marker.exists():
                mark_deadline(phase, marker)
            time.sleep(30)
        rc = proc.returncode
        state = hb_state(hbf)
        try:
            with outf.open("rb") as fp:
                fp.seek(off)
                new_out = fp.read().decode(errors="replace")
        except OSError:
            new_out = ""
        log(f"phase {phase}: r8_eval.py exit {rc}, heartbeat state {state}")
        if state == "done":
            continue  # returns 'done' at the loop top once no r8_eval.py is alive
        if marker.exists():
            continue  # returns 'deadline' at the loop top once no r8_eval.py is alive
        if foreign_stop():
            return "stopped"
        if state == "halted" or "preflight FAILED" in new_out or "sha256" in new_out and "!= pinned" in new_out:
            log(f"phase {phase}: r8_eval.py reports a pin / preflight / integrity failure (heartbeat {state}); see "
                f"{outf.name} (PREREG_R8 7.4)")  # fmt: skip
            return "halted"
        if live_evals():
            continue  # another r8_eval.py holds the lane (it was started by someone else): wait for it, no restart counted
        tries += 1
        if tries >= RETRIES:
            return "failed"
        log(f"phase {phase}: restart in {RETRY_WAIT:.0f} s (attempt {tries + 1} of {RETRIES})")
        time.sleep(RETRY_WAIT)


def select(cmds: list[list[str]], label: str) -> bool:
    for c in cmds:
        rc = run_step(f"{label} {' '.join(c)}", [str(KPY), "-I", "-B", str(ROOT / "r8_select.py"), *c], True, "r8_select_stdout.log")
        if rc != 0:
            log(f"HALT: r8_select.py {' '.join(c)} exit {rc} (see r8_select_stdout.log); nothing further is selected")
            return False
    return True


def cmd_run(a) -> int:
    lockfd = os.open(ROOT / ".r8_driver.lock", os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("another r8_driver.py runs", file=sys.stderr)
        return 1
    log(f"r8_driver.py start pid {os.getpid()} (sha256 {sha256_file(MINE)[:16]})")
    probs = docs_problems()
    if probs:
        log("HALT: documents do not hold: " + "; ".join(probs))
        beat("halted", "; ".join(probs))
        return 1
    if (ROOT / "SELECTION_R8.sha256").is_file():
        log("SELECTION_R8.sha256 exists: round 8 is final")
        beat("done", "SELECTION_R8.sha256 exists")
        return 0
    while time.time() < T_C:  # 1
        if foreign_stop():
            log("STOP file before T_c: driver ends")
            beat("stopped before T_c")
            return 0
        beat("waiting for T_c", f"{(T_C - time.time()) / 3600:.2f} h left")
        time.sleep(min(60.0, max(1.0, T_C - time.time())))
    if not (ROOT / "POOL_R8.sha256").is_file():  # 2
        beat("freezing the pool")
        rc = run_step("freeze", [str(PY), "-I", "-B", str(ROOT / "r8_pool.py"), "freeze"], True, "r8_pool_stdout.log")
        if rc != 0 or not (ROOT / "POOL_R8.sha256").is_file():
            log(f"HALT: pool freeze failed (exit {rc}); see r8_pool_stdout.log")
            beat("halted", "pool freeze failed")
            return 1
    rc = run_step("pool verify", [str(PY), "-I", "-B", str(ROOT / "r8_pool.py"), "verify"], True, "r8_pool_stdout.log")
    if rc != 0:
        log("HALT: POOL_R8.sha256 / RELOCATION_R8.tsv do not verify")
        beat("halted", "pool verify failed")
        return 1
    if not (ROOT / "SELECTION_S8.sha256").is_file():  # 3
        res = eval_phase("screen", S_CUT, S_CUT_FILE)
        if res == "halted":
            log("HALT: phase S evaluation driver reported a pin / preflight / integrity failure; nothing is selected; the R7 "
                "file stays in force (PREREG_R8 7.4)")  # fmt: skip
            beat("halted", "phase S: r8_eval.py pin / preflight / integrity failure")
            return 1
        if res in ("deadline", "failed"):
            reason = ("phase S was not complete by 2026-11-04 08:00 CST" if res == "deadline" else
                      "phase S evaluations failed repeatedly") + "; every episode keeps its round-7 pick (PREREG_R8 7.3)"
            ok = select([["keep-r7", "--reason", reason]], "keep-r7")
            beat("done" if ok else "halted", reason)
            return 0 if ok else 1
        if res == "stopped":
            log("STOP file (not a deadline): driver ends after phase S evaluations; nothing selected")
            beat("stopped", "STOP file")
            return 0
        beat("phase S: scoring and selection")
        if not select([["check", "--phase", "S"], ["gate", "--phase", "S"], ["score", "--phase", "S"], ["select", "--phase", "S"]],
                      "phase S"):  # fmt: skip
            beat("halted", "phase-S scoring / selection failed")
            return 1
    fin = json.loads((ROOT / "final_r8.json").read_text())
    if fin.get("finalists"):  # 4
        if foreign_stop():
            log("STOP file (not a deadline) before phase F: driver ends; nothing further selected")
            beat("stopped", "STOP file")
            return 0
        res = eval_phase("final", F_DEADLINE, DEADLINE_F)
        if res == "stopped":
            log("STOP file (not a deadline) during phase F: driver ends; nothing further selected")
            beat("stopped", "STOP file")
            return 0
        if res in ("failed", "halted"):
            why = ("phase F evaluations failed repeatedly" if res == "failed" else
                   "phase F evaluation driver reported a pin / preflight / integrity failure")  # fmt: skip
            log(f"HALT: {why}; nothing further is selected; the R7 file stays in force (PREREG_R8 7.4)")
            beat("halted", why)
            return 1
        beat("phase F: scoring and selection")
        steps = [["check", "--phase", "F"], ["gate", "--phase", "F"], ["score", "--phase", "F"], ["select", "--phase", "F"]]
    else:
        log("final_r8.json names no finalist: phase F has no job; every episode keeps its round-7 pick")
        steps = [["select", "--phase", "F"]]
    ok = select(steps, "phase F")
    beat("done" if ok else "halted", "SELECTION_R8.sha256 written" if ok else "phase-F scoring / selection failed")
    log("round 8 " + ("final: SELECTION_R8.sha256 " + sha256_file(ROOT / "SELECTION_R8.sha256") if ok else "halted"))
    return 0 if ok else 1


def cmd_status(a) -> int:
    print(HB.read_text() if HB.is_file() else "no heartbeat")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("run", "status"))
    a = ap.parse_args()
    if not sys.flags.dont_write_bytecode or not sys.flags.isolated:
        print("run with python -I -B")
        return 2
    return cmd_run(a) if a.cmd == "run" else cmd_status(a)


if __name__ == "__main__":
    raise SystemExit(main())
