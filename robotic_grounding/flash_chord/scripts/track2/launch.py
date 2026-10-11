#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Track 2 Tier-1 training queue: one FlashCHORD PPO job per episode, K jobs per GPU.

Python port of ``/root/autodl-tmp/haoran.zhang/scripts/track2_queue.sh`` (docs/coord_impl_spec.md 7.2).
Run it after ``source /root/autodl-tmp/haoran.zhang/scripts/fc_env.sh``::

    python scripts/track2/launch.py --arm C3_wp0_rsiv3 --experiment g1_recon_body_coord_rsi \
        --episodes all --gpus 0,1,2,3 --jobs-per-gpu 2 --iterations 20000 \
        --out /root/autodl-tmp/haoran.zhang/fc_runs/C3_wp0_rsiv3 \
        --wandb-mode online --wandb-entity ENTITY --wandb-project PROJECT

Per episode (from ``$HUMAN_MOTION_DATA_DIR/whole_body/soma/track2_manifest.json``):

* output ``<out>/ep{NN}_{object}``; the job is done when ``model_{iterations}.safetensors`` exists;
* otherwise it resumes from the newest ``model_(\\d+).safetensors`` (anchored), and for RSI arms it
  requires the matching ``rsi_{N}.safetensors`` sidecar (or the job fails loudly);
* ``scene.support=false`` when the support stage is missing or has no ``def Cylinder|def Cube``
  (the ``scripts/fc_train.sh`` rule);
* ``CUDA_VISIBLE_DEVICES=<gpu>``, ``PYTHONPATH=<code>/src``, CPU affinity of the GPU's socket;
* stdout/stderr to ``<out>/logs/ep{NN}.log``, events to ``<out>/status.tsv`` and ``<out>/queue.log``;
* W&B through the stock logging keys only (``logging.experiment=<arm>``,
  ``logging.run_name=ep{NN}_{object}``, ``logging.upload_checkpoints=false``). The launcher never
  reads, prints, or writes ``WANDB_API_KEY``; the process environment is inherited as is;
* a local copy of every logged metric record in ``<out>/ep{NN}_{object}/metrics.jsonl``
  (``+logging.local_jsonl=true``, read by DEV ``scripts/train_rl.py``; ``--no-local-metrics`` turns it off).

On SIGINT/SIGTERM it terminates only the process groups of the jobs it started.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

MODEL_PATTERN = re.compile(r"^model_(\d+)\.safetensors$")
SUPPORT_PATTERN = re.compile(r"def (Cylinder|Cube)")
DEFAULT_CODE = Path(__file__).resolve().parents[2]
DEFAULT_CPU_MAP = ("0,1,2=0-51,104-155", "3,4,5,6,7=52-103,156-207")
RSI_RESET_PATTERN = re.compile(r"override\s+/reset\s*:\s*recon_body_rsi\b")


@dataclass
class Job:
    episode: int
    sequence: str
    object_name: str
    seed: int
    tag: str
    run_name: str
    output: Path
    parquet: Path
    support: bool
    attempts: int = 0


def now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_int_list(text: str) -> list[int]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            low, high = part.split("-", 1)
            values.extend(range(int(low), int(high) + 1))
        else:
            values.append(int(part))
    return values


def parse_cpu_map(entries) -> dict[int, set[int]]:
    mapping: dict[int, set[int]] = {}
    for entry in entries:
        gpus, _, cpus = entry.partition("=")
        if not cpus:
            raise ValueError(f"--cpu-map entry {entry!r} must look like '0,1,2=0-51,104-155'")
        for gpu in parse_int_list(gpus):
            mapping[gpu] = set(parse_int_list(cpus))
    return mapping


def load_manifest(data_dir: Path) -> dict[int, tuple[str, str]]:
    manifest_path = data_dir / "whole_body" / "soma" / "track2_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    episodes: dict[int, tuple[str, str]] = {}
    for sequence, record in manifest.items():
        episode = int(record["episode_index"])
        if episode in episodes:
            raise ValueError(f"{manifest_path}: episode {episode} appears twice")
        episodes[episode] = (str(record.get("sequence_id", sequence)), str(record["object"]))
    return episodes


def support_available(data_dir: Path, sequence: str) -> bool:
    support = data_dir / "whole_body" / "reconstructed_stage" / f"{sequence}_support.usda"
    if not support.is_file():
        return False
    return SUPPORT_PATTERN.search(support.read_text(errors="replace")) is not None


def checkpoints(directory: Path) -> list[tuple[int, Path]]:
    if not directory.is_dir():
        return []
    found = []
    for path in directory.iterdir():
        match = MODEL_PATTERN.match(path.name)
        if match and path.is_file():
            found.append((int(match.group(1)), path))
    return sorted(found)


def is_rsi_experiment(code: Path, experiment: str, extra: list[str]) -> bool:
    if any(re.match(r"^\+?reset=recon_body_rsi$", item) for item in extra):
        return True
    path = code / "src" / "flash_chord" / "configs" / "experiment" / f"{experiment}.yaml"
    return RSI_RESET_PATTERN.search(path.read_text()) is not None


def hydra_quote(value: str) -> str:
    if "'" in value:
        raise ValueError(f"path contains a single quote, which Hydra overrides cannot carry: {value!r}")
    return f"'{value}'"


class Launcher:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.out: Path = args.out.resolve()
        self.code: Path = args.code.resolve()
        self.python: str = args.python
        self.extra = shlex.split(args.extra) if args.extra else []
        self.rsi = {"yes": True, "no": False}.get(args.rsi)
        if self.rsi is None:
            self.rsi = is_rsi_experiment(self.code, args.experiment, self.extra)
        self.cpu_map = {} if args.no_pin else parse_cpu_map(args.cpu_map or DEFAULT_CPU_MAP)
        self.available_cpus = set(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else set()
        self.running: dict[tuple[int, int], tuple[Job, subprocess.Popen, object]] = {}
        self.stop_requested = False
        self.failed: list[str] = []

    # ---------------------------------------------------------------- bookkeeping
    def log(self, message: str) -> None:
        line = f"[{now()}] {message}"
        print(line, flush=True)
        if not self.args.dry_run:
            with (self.out / "queue.log").open("a") as stream:
                stream.write(line + "\n")

    def status(self, job: Job, gpu: int | str, event: str, code: int | str = "") -> None:
        if self.args.dry_run:
            return
        path = self.out / "status.tsv"
        new = not path.exists()
        with path.open("a") as stream:
            if new:
                stream.write("time\ttag\tobject\tgpu\tattempt\tevent\texit\n")
            stream.write(f"{now()}\t{job.tag}\t{job.object_name}\tgpu{gpu}\t{job.attempts}\t{event}\t{code}\n")

    def done(self, job: Job) -> bool:
        return (job.output / f"model_{self.args.iterations}.safetensors").is_file()

    # ---------------------------------------------------------------- job construction
    def jobs(self) -> list[Job]:
        data_dir = Path(os.environ["HUMAN_MOTION_DATA_DIR"])
        manifest = load_manifest(data_dir)
        if self.args.episodes == "all":
            episodes = sorted(manifest)
        else:
            episodes = parse_int_list(self.args.episodes)
            unknown = sorted(set(episodes) - set(manifest))
            if unknown:
                raise SystemExit(f"episodes {unknown} are not in the manifest")
        seeds = [(episode, self.args.seed) for episode in episodes]
        for item in self.args.extra_seeds.split(",") if self.args.extra_seeds else []:
            episode, seed = (int(value) for value in item.split(":"))
            if episode not in manifest:
                raise SystemExit(f"--extra-seeds episode {episode} is not in the manifest")
            seeds.append((episode, seed))
        jobs = []
        for episode, seed in seeds:
            sequence, object_name = manifest[episode]
            suffix = "" if seed == self.args.seed else f"_seed{seed}"
            tag = f"ep{episode:02d}_{object_name}{suffix}"
            parquet = data_dir / "whole_body" / "soma" / f"sequence_id={sequence}" / "robot_name=g1" / "data.parquet"
            jobs.append(
                Job(
                    episode=episode,
                    sequence=sequence,
                    object_name=object_name,
                    seed=seed,
                    tag=tag,
                    run_name=tag,
                    output=self.out / tag,
                    parquet=parquet,
                    support=support_available(data_dir, sequence),
                )
            )
        return jobs

    def resume_arguments(self, job: Job) -> list[str]:
        found = [(number, path) for number, path in checkpoints(job.output) if number < self.args.iterations]
        if not found:
            return []
        if self.rsi:
            usable = [(n, p) for n, p in found if (job.output / f"rsi_{n}.safetensors").is_file()]
            newest = found[-1]
            if not usable or usable[-1][0] != newest[0]:
                if not self.args.resume_fallback or not usable:
                    raise FileNotFoundError(
                        f"{job.tag}: RSI arm resume needs {job.output / f'rsi_{newest[0]}.safetensors'} "
                        f"next to {newest[1].name}; it is missing (use --resume-fallback to resume from the "
                        "newest model/rsi pair instead)"
                    )
                self.log(f"{job.tag}: rsi_{newest[0]} missing; --resume-fallback resumes from model_{usable[-1][0]}")
                newest = usable[-1]
        else:
            newest = found[-1]
        return ["training.resume=true", f"training.checkpoint={hydra_quote(str(newest[1]))}"]

    def command(self, job: Job, resume: list[str]) -> list[str]:
        args = self.args
        command = [
            self.python,
            str(self.code / "scripts" / "train_rl.py"),
            f"experiment={args.experiment}",
            f"task.parquet={hydra_quote(str(job.parquet))}",
            f"reset.seed={job.seed}",
            f"output_dir={hydra_quote(str(job.output))}",
            f"hydra.run.dir={hydra_quote(str(job.output / 'hydra' / dt.datetime.now().strftime('%Y%m%d_%H%M%S')))}",
            f"training.max_iterations={args.iterations}",
            f"logging.mode={args.wandb_mode}",
        ]
        if args.wandb_entity:
            command.append(f"logging.entity={args.wandb_entity}")
        if args.wandb_project:
            command.append(f"logging.project={args.wandb_project}")
        command += [
            f"logging.experiment={args.arm}",
            f"logging.run_name={job.run_name}",
            "logging.upload_checkpoints=false",
        ]
        if args.local_metrics:
            command.append("+logging.local_jsonl=true")
        if not job.support:
            command.append("scene.support=false")
        return command + resume + self.extra

    def environment(self, gpu: int) -> dict[str, str]:
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        environment["PYTHONPATH"] = str(self.code / "src")
        environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        return environment

    def cpus(self, gpu: int) -> set[int] | None:
        cpus = self.cpu_map.get(gpu)
        if not cpus:
            return None
        if self.available_cpus:
            cpus = cpus & self.available_cpus
        return cpus or None

    # ---------------------------------------------------------------- execution
    def start(self, slot: tuple[int, int], job: Job) -> bool:
        gpu = slot[0]
        job.attempts += 1
        try:
            resume = self.resume_arguments(job)
        except FileNotFoundError as error:
            self.log(f"ERROR {error}")
            self.status(job, gpu, "error_missing_rsi_sidecar", "")
            self.failed.append(job.tag)
            return False
        command = self.command(job, resume)
        cpus = self.cpus(gpu)
        if self.args.dry_run:
            pin = f"cpus={min(cpus)}..{max(cpus)} ({len(cpus)})" if cpus else "unpinned"
            print(f"[dry-run] gpu{gpu} slot{slot[1]} {pin} {job.tag}:\n  " + shlex.join(command), flush=True)
            return True
        job.output.mkdir(parents=True, exist_ok=True)
        log_path = self.out / "logs" / f"{job.tag.split('_')[0] if job.seed == self.args.seed else job.tag}.log"
        stream = log_path.open("a")
        stream.write(f"\n===== [{now()}] {job.tag} attempt {job.attempts} gpu{gpu}\n{shlex.join(command)}\n")
        stream.flush()

        def preexec():
            if cpus:
                os.sched_setaffinity(0, cpus)

        process = subprocess.Popen(
            command,
            cwd=self.code,
            env=self.environment(gpu),
            stdout=stream,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            preexec_fn=preexec,
        )
        self.running[slot] = (job, process, stream)
        what = "resume " + Path(resume[1].split("=", 1)[1].strip("'")).name if resume else "fresh"
        self.log(f"start {job.tag} ({what}, seed {job.seed}) on gpu{gpu} slot {slot[1]} pid {process.pid}")
        self.status(job, gpu, "start", "")
        return True

    def reap(self, queue: collections.deque) -> None:
        for slot, (job, process, stream) in list(self.running.items()):
            code = process.poll()
            if code is None:
                continue
            stream.close()
            del self.running[slot]
            finished = self.done(job)
            self.status(job, slot[0], "finished" if finished else "failed", code)
            if finished:
                self.log(f"{job.tag} finished (exit {code})")
            elif job.attempts <= self.args.max_retries:
                queue.append(job)
                self.log(f"{job.tag} failed (exit {code}); requeued, will resume (attempt {job.attempts + 1})")
            else:
                self.failed.append(job.tag)
                self.log(f"{job.tag} FAILED after {job.attempts} attempt(s) (exit {code}); giving up")

    def terminate_children(self, grace: float = 60.0) -> None:
        for _, process, _ in self.running.values():
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + grace
        for _, process, _ in self.running.values():
            remaining = max(deadline - time.monotonic(), 0.1)
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        for slot, (job, process, stream) in list(self.running.items()):
            stream.close()
            self.status(job, slot[0], "interrupted", process.returncode)
        self.running.clear()

    def run(self) -> int:
        args = self.args
        jobs = self.jobs()
        pending = collections.deque()
        for job in jobs:
            if self.done(job):
                self.log(f"skip {job.tag} (model_{args.iterations}.safetensors exists)")
            else:
                pending.append(job)
        slots = [(gpu, k) for k in range(args.jobs_per_gpu) for gpu in parse_int_list(args.gpus)]
        self.log(
            f"arm={args.arm} experiment={args.experiment} rsi={self.rsi} iterations={args.iterations} "
            f"gpus={args.gpus} jobs_per_gpu={args.jobs_per_gpu} code={self.code} "
            f"wandb={args.wandb_mode}:{args.wandb_entity}/{args.wandb_project} jobs={len(jobs)} pending={len(pending)}"
        )
        if args.dry_run:
            for slot, job in zip(slots * (len(pending) // max(len(slots), 1) + 1), list(pending)):
                self.start(slot, job)
            return 0

        def request_stop(signum, _frame):
            self.stop_requested = True
            self.log(f"received signal {signum}; terminating this launcher's jobs")

        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
        try:
            while (pending or self.running) and not self.stop_requested:
                self.reap(pending)
                for slot in slots:
                    if self.stop_requested or not pending:
                        break
                    if slot in self.running:
                        continue
                    if self.start(slot, pending.popleft()) and args.stagger > 0:
                        self._sleep(args.stagger)
                if pending or self.running:
                    self._sleep(args.poll)
        finally:
            if self.running:
                self.terminate_children()
        complete = sum(self.done(job) for job in jobs)
        self.log(f"QUEUE_DONE: {complete}/{len(jobs)} jobs have model_{args.iterations}.safetensors")
        if self.failed:
            self.log(f"failed: {', '.join(self.failed)}")
        return 0 if complete == len(jobs) else 1

    def _sleep(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while not self.stop_requested and time.monotonic() < deadline:
            time.sleep(max(min(1.0, deadline - time.monotonic()), 0.0))


def preflight(args: argparse.Namespace) -> None:
    errors = []
    if not os.environ.get("HUMAN_MOTION_DATA_DIR"):
        errors.append("HUMAN_MOTION_DATA_DIR is not set (source scripts/fc_env.sh)")
    if "nvidia/cu13/lib" not in os.environ.get("LD_LIBRARY_PATH", ""):
        errors.append("LD_LIBRARY_PATH lacks nvidia/cu13/lib (source scripts/fc_env.sh)")
    code = args.code.resolve()
    if not (code / "scripts" / "train_rl.py").is_file():
        errors.append(f"{code}/scripts/train_rl.py does not exist")
    experiment = code / "src" / "flash_chord" / "configs" / "experiment" / f"{args.experiment}.yaml"
    if not experiment.is_file():
        errors.append(f"experiment config {experiment} does not exist")
    if args.wandb_mode != "disabled" and (not args.wandb_entity or not args.wandb_project):
        errors.append("--wandb-entity and --wandb-project are required unless --wandb-mode disabled")
    if args.jobs_per_gpu < 1 or args.iterations < 1 or args.max_retries < 0:
        errors.append("--jobs-per-gpu and --iterations must be positive, --max-retries non-negative")
    if errors:
        raise SystemExit("preflight failed:\n  - " + "\n  - ".join(errors))
    probe = subprocess.run(
        [args.python, "-c", "import flash_chord, sys; sys.stdout.write(flash_chord.__file__)"],
        env={**os.environ, "PYTHONPATH": str(code / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    expected = (code / "src" / "flash_chord").resolve()
    if probe.returncode != 0 or Path(probe.stdout.strip()).resolve().parent != expected:
        raise SystemExit(f"{args.python} with PYTHONPATH={code}/src imports {probe.stdout.strip()!r}, not {expected}")


def write_launch_record(args: argparse.Namespace) -> None:
    code = args.code.resolve()

    def git(*command):
        result = subprocess.run(["git", "-C", str(code), *command], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None

    record = {
        "time": now(),
        "argv": sys.argv,
        "arm": args.arm,
        "experiment": args.experiment,
        "iterations": args.iterations,
        "code": str(code),
        "code_commit": git("rev-parse", "HEAD"),
        "code_dirty_files": len((git("status", "--porcelain") or "").splitlines()),
        "python": args.python,
    }
    with (args.out / "launch_history.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True, help="arm name; also logging.experiment (W&B group)")
    parser.add_argument("--experiment", required=True, help="Hydra experiment, e.g. g1_recon_body_coord_rsi")
    parser.add_argument("--episodes", default="all", help="'all' or a list such as 0,2,7 or 0-5")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--wandb-mode", default="disabled", choices=("online", "offline", "disabled"))
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-project")
    parser.add_argument("--extra", default="", help="extra Hydra overrides (shell-quoted string)")
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--code", type=Path, default=DEFAULT_CODE, help="training code root (DEV)")
    parser.add_argument("--seed", type=int, default=42, help="reset.seed (and training.seed via interpolation)")
    parser.add_argument("--extra-seeds", default="", help="extra seed jobs as ep:seed pairs, e.g. 19:7,28:7")
    parser.add_argument("--rsi", default="auto", choices=("auto", "yes", "no"),
                        help="require rsi_N sidecars on resume (auto: experiment overrides /reset to recon_body_rsi)")
    parser.add_argument("--resume-fallback", action="store_true",
                        help="RSI arms: resume from the newest model/rsi pair when the newest model lacks its sidecar")
    parser.add_argument("--cpu-map", action="append",
                        help=f"GPU-to-CPU affinity, repeatable (default: {' '.join(DEFAULT_CPU_MAP)})")
    parser.add_argument("--no-pin", action="store_true", help="do not set CPU affinity")
    parser.add_argument("--local-metrics", action=argparse.BooleanOptionalAction, default=True,
                        help="also append every logged metric record to <job dir>/metrics.jsonl (default on)")
    parser.add_argument("--stagger", type=float, default=15.0, help="seconds between job launches")
    parser.add_argument("--poll", type=float, default=30.0, help="seconds between queue polls")
    args = parser.parse_args(argv)
    preflight(args)
    args.out = args.out.resolve()
    if not args.dry_run:
        (args.out / "logs").mkdir(parents=True, exist_ok=True)
        write_launch_record(args)
    return Launcher(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
