#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export -> official kit evaluator -> pack -> report, for one arm (docs/coord_impl_spec.md 7.3).

    python scripts/track2/eval_all.py --arm-dir /root/autodl-tmp/haoran.zhang/fc_runs/C3_wp0_rsiv3 \
        --arm C3_wp0_rsiv3 --iteration 20000 --episodes all --gpus 4,5 \
        --code-commit https://github.com/<owner>/<repo>/commit/<sha>

Steps:

1. **Select** ``<arm-dir>/ep{NN}_{object}/model_{iteration}.safetensors`` per episode (the
   pre-registered last checkpoint). With ``--checkpoint-from-final``, each episode's FINAL checkpoint instead:
   the one its ``final.json`` names (written by the training runner at early stop or max_iterations, so the
   iteration differs per episode), else ``model_{iteration}``.
2. **Export** every non-B0 arm with ``python -m flash_chord.training.ppo.coordinate.export`` (DEV
   code, CPU, stock-runtime check) into ``<arm-dir>/export/ep{NN}/``; the report JSON is kept.
   ``--export-tolerance`` sets the exporter's equivalence tolerance (default: the exporter's 1e-5; long runs
   need 1e-4, see scripts/track2_pre_export.sh). An existing ok export of the same source is reused.
3. **Evaluate** with the kit's ``examples/run_policy_evaluation.py`` and the kit's
   ``flash_chord/scripts/evaluate_policy.py`` against the STOCK runtime (``PYTHONPATH=$STOCK/src``,
   never the DEV tree), one episode per invocation, episodes spread over ``--gpus``.
4. **Pack** with ``<pack-python> <kit>/tools/pack_submission.py`` into ``<arm-dir>/<arm>_tier1.csv``
   when every scored episode has a trajectory and ``--code-commit`` is given. ``--partial-pack``
   packs a subset through a restricted copy of the kit's tracks.json (diagnostic only; the file is
   named ``*_PARTIAL.csv``). Nothing is ever uploaded.
5. **Report** a per-episode table (evaluator ``add_auc``, ``chord_sr``, ``maniptrans_sr``,
   ``mppe_cm``; export diffs; the packer's local scores) as CSV, Markdown and JSON in ``<arm-dir>``.

The script never writes inside ``$STOCK``, the kit, the kit zip, or the dataset (evaluator child
processes run with ``PYTHONDONTWRITEBYTECODE=1``).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime as dt
import hashlib
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import zipfile
from pathlib import Path

DEFAULT_CODE = Path(__file__).resolve().parents[2]
ME = Path("/root/autodl-tmp/haoran.zhang")
DEFAULT_STOCK = ME / "code/video_to_data_8c778f9/robotic_grounding/flash_chord"
DEFAULT_KIT = ME / "v2d_submission_kit_src/v2d_submission_kit"
DEFAULT_DATASET = ME / "data/video_to_data_challenge"
DEFAULT_PACK_PYTHON = ME / "conda_envs/v2dkit/bin/python"
FINAL_MARKER = "final.json"  # training/ppo/runner.py: the run's final checkpoint (early stop or max_iterations)
METRIC_KEYS = ("add_auc", "chord_sr", "maniptrans_sr", "mppe_cm", "object_position_error_m", "object_orientation_error_deg")


def now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_int_list(text: str) -> list[int]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            low, high = part.split("-", 1)
            values.extend(range(int(low), int(high) + 1))
        elif part:
            values.append(int(part))
    return values


def read_final_marker(directory: Path) -> dict | None:
    """``<run dir>/final.json`` as (iteration, checkpoint file name, reason), validated; None if absent."""
    path = directory / FINAL_MARKER
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    iteration, checkpoint = data.get("iteration"), data.get("checkpoint")
    if (
        isinstance(iteration, bool)
        or not isinstance(iteration, int)
        or iteration <= 0
        or checkpoint != f"model_{iteration}.safetensors"
    ):
        raise ValueError(f"{path}: expected iteration > 0 and checkpoint model_<iteration>.safetensors, got {data}")
    return {"iteration": iteration, "checkpoint": checkpoint, "reason": str(data.get("reason"))}


def stock_environment(stock: Path, gpu: str | None = None) -> dict[str, str]:
    """Child environment for the official runtime: no DEV path, no bytecode writes into $STOCK."""
    environment = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"}}
    environment["PYTHONPATH"] = str(stock / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = gpu
    return environment


def extract_kit(kit_zip: Path, kit_dir: Path) -> Path:
    """Extract the kit zip into a NEW directory (path-traversal safe) and return the kit root."""
    if kit_dir.exists() and any(kit_dir.iterdir()):
        raise SystemExit(f"--kit-dir {kit_dir} must be new or empty")
    kit_dir.mkdir(parents=True, exist_ok=True)
    root = kit_dir.resolve()
    with zipfile.ZipFile(kit_zip) as archive:
        for member in archive.infolist():
            target = (root / member.filename).resolve()
            if not target.is_relative_to(root):
                raise SystemExit(f"kit zip member escapes the extraction directory: {member.filename}")
        archive.extractall(root)
    candidates = [path.parent.parent for path in root.rglob("examples/run_policy_evaluation.py")]
    if len(candidates) != 1:
        raise SystemExit(f"expected one kit root in {root}, found {candidates}")
    return candidates[0]


class Evaluation:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.arm_dir: Path = args.arm_dir.resolve()
        self.stock: Path = args.stock_runtime.resolve()
        self.code: Path = args.code.resolve()
        self.results: Path = (args.results or self.arm_dir / "results").resolve()
        self.export_root: Path = self.arm_dir / "export"
        self.kit: Path = self._kit()
        self.eval_script = self.kit / "flash_chord" / "scripts" / "evaluate_policy.py"
        self.runner = self.kit / "examples" / "run_policy_evaluation.py"
        self.packer = self.kit / "tools" / "pack_submission.py"
        self.export_enabled = {"yes": True, "no": False}.get(args.export, args.arm.upper() != "B0")
        self.rows: dict[int, dict] = {}

    # ------------------------------------------------------------------ setup
    def _kit(self) -> Path:
        args = self.args
        if args.kit_zip is not None:
            if args.kit_dir is None:
                raise SystemExit("--kit-zip needs --kit-dir (a new directory to extract into)")
            return extract_kit(args.kit_zip, args.kit_dir)
        return args.kit_root.resolve()

    def preflight(self) -> dict:
        problems = []
        for path in (self.runner, self.eval_script, self.packer, self.stock / "src" / "flash_chord" / "__init__.py"):
            if not path.is_file():
                problems.append(f"missing {path}")
        if problems:
            raise SystemExit("preflight failed:\n  - " + "\n  - ".join(problems))
        stock_script = self.stock / "scripts" / "evaluate_policy.py"
        kit_hash, stock_hash = sha256(self.eval_script), sha256(stock_script)
        if kit_hash != stock_hash and not self.args.allow_eval_script_mismatch:
            raise SystemExit(f"kit evaluate_policy.py ({kit_hash}) differs from {stock_script} ({stock_hash})")
        probe = subprocess.run(
            [self.args.python, "-c", "import flash_chord, sys; sys.stdout.write(flash_chord.__file__)"],
            env=stock_environment(self.stock),
            cwd=self.stock,
            capture_output=True,
            text=True,
            check=False,
        )
        module = Path(probe.stdout.strip()).resolve() if probe.returncode == 0 else None
        if module is None or module.parent != (self.stock / "src" / "flash_chord").resolve():
            raise SystemExit(f"evaluator python imports {probe.stdout.strip() or probe.stderr[-500:]!r}, not $STOCK")
        return {
            "kit_root": str(self.kit),
            "eval_script": str(self.eval_script),
            "eval_script_sha256": kit_hash,
            "stock_runtime": str(self.stock),
            "stock_flash_chord": str(module),
        }

    def episodes(self) -> dict[int, tuple[str, str]]:
        manifest_path = (
            self.args.manifest
            or Path(os.environ.get("HUMAN_MOTION_DATA_DIR", ME / "data/human_motion_data"))
            / "whole_body/soma/track2_manifest.json"
        )
        manifest = json.loads(Path(manifest_path).read_text())
        episodes = {int(r["episode_index"]): (str(r.get("sequence_id", s)), str(r["object"])) for s, r in manifest.items()}
        if self.args.episodes != "all":
            wanted = parse_int_list(self.args.episodes)
            unknown = sorted(set(wanted) - set(episodes))
            if unknown:
                raise SystemExit(f"episodes {unknown} are not in {manifest_path}")
            episodes = {episode: episodes[episode] for episode in wanted}
        return dict(sorted(episodes.items()))

    # ------------------------------------------------------------------ 1. select
    def select(self, episodes: dict[int, tuple[str, str]]) -> None:
        iteration = self.args.iteration
        for episode, (sequence, object_name) in episodes.items():
            row = {"episode": episode, "object": object_name, "sequence": sequence, "status": "selected"}
            directory = self.arm_dir / f"ep{episode:02d}_{object_name}"
            if not directory.is_dir():
                matches = sorted(self.arm_dir.glob(f"ep{episode:02d}_*"))
                matches = [m for m in matches if m.is_dir() and "_seed" not in m.name]
                directory = matches[0] if len(matches) == 1 else directory
            if not directory.is_dir() and (self.arm_dir / f"ep{episode:02d}").is_dir():
                directory = self.arm_dir / f"ep{episode:02d}"  # scripts/track2_queue.sh layout
            checkpoint = directory / f"model_{iteration}.safetensors"
            row["iteration"], row["selected_by"] = iteration, f"--iteration {iteration}"
            if self.args.checkpoint_from_final:
                try:
                    final = read_final_marker(directory)
                except (OSError, ValueError) as error:
                    final, row["status"] = None, "bad_final_json"
                    log(f"ep{episode:02d}: unusable {FINAL_MARKER}: {error}")
                if final is not None:
                    checkpoint = directory / final["checkpoint"]
                    row["iteration"], row["final_reason"] = final["iteration"], final["reason"]
                    row["selected_by"] = f"{FINAL_MARKER} ({final['reason']})"
                elif row["status"] != "bad_final_json":
                    row["selected_by"] = f"no {FINAL_MARKER}: --iteration {iteration}"
            row["source_checkpoint"] = str(checkpoint)
            if row["status"] == "selected" and not checkpoint.is_file():
                row["status"] = "missing_checkpoint"
            self.rows[episode] = row
        missing = [e for e, r in self.rows.items() if r["status"] != "selected"]
        if self.args.checkpoint_from_final:
            from_final = sum(r["selected_by"].startswith(FINAL_MARKER) for r in self.rows.values())
            iterations = {e: r["iteration"] for e, r in self.rows.items() if r["status"] == "selected"}
            log(f"selected {len(self.rows) - len(missing)}/{len(self.rows)} final checkpoints ({from_final} from "
                f"{FINAL_MARKER}, the rest model_{iteration}); iterations {iterations}; missing or unusable {missing}")
        else:
            log(f"selected {len(self.rows) - len(missing)}/{len(self.rows)} model_{iteration} checkpoints; missing {missing}")

    # ------------------------------------------------------------------ 2. export
    def _export_one(self, episode: int) -> None:
        row = self.rows[episode]
        source = Path(row["source_checkpoint"])
        directory = self.export_root / f"ep{episode:02d}"
        destination = directory / source.name
        report_path = directory / "export_report.json"
        row["exported_checkpoint"] = str(destination)
        row["export_report"] = str(report_path)
        if destination.is_file() and report_path.is_file() and not self.args.force_export:
            report = json.loads(report_path.read_text())
            if report.get("status") == "ok" and report.get("source_sha256") == sha256(source):
                row.update(self._export_fields(report))
                row["status"] = "exported"
                return
        directory.mkdir(parents=True, exist_ok=True)
        environment = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME"}}
        environment.update(
            {"PYTHONPATH": str(self.code / "src"), "JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""}
        )
        command = [
            self.args.python, "-m", "flash_chord.training.ppo.coordinate.export",
            "--input", str(source), "--output", str(destination),
            "--stock-runtime", str(self.stock), "--report", str(report_path),
        ]  # fmt: skip
        if self.args.export_tolerance is not None:
            command += ["--tolerance", repr(self.args.export_tolerance)]
        with (directory / "export.log").open("w") as stream:
            completed = subprocess.run(command, cwd=self.code, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        if completed.returncode != 0 or not destination.is_file():
            row["status"] = f"export_failed(exit {completed.returncode})"
            if report_path.is_file():
                row.update(self._export_fields(json.loads(report_path.read_text())))
            return
        row.update(self._export_fields(json.loads(report_path.read_text())))
        row["status"] = "exported"

    @staticmethod
    def _export_fields(report: dict) -> dict:
        stock = report.get("stock_runtime") or {}
        return {
            "export_max_abs_diff": report.get("max_abs_diff"),
            "export_max_rel_diff": report.get("max_rel_diff"),
            "export_stock_verified": stock.get("verified"),
            "export_source_kind": report.get("source_kind"),
        }

    def export(self) -> None:
        todo = [e for e, r in self.rows.items() if r["status"] == "selected"]
        if not self.export_enabled:
            for episode in todo:
                self.rows[episode]["evaluated_checkpoint"] = self.rows[episode]["source_checkpoint"]
            log(f"export disabled for arm {self.args.arm}; evaluating the training checkpoints directly")
            return
        log(f"exporting {len(todo)} checkpoints with {self.args.export_jobs} CPU workers")
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.args.export_jobs) as pool:
            for episode, future in [(e, pool.submit(self._export_one, e)) for e in todo]:
                future.result()
                row = self.rows[episode]
                log(f"ep{episode:02d}: {row['status']} max_rel_diff={row.get('export_max_rel_diff')}")
                if row["status"] == "exported":
                    row["evaluated_checkpoint"] = row["exported_checkpoint"]

    # ------------------------------------------------------------------ 3. evaluate
    def _evaluated(self, episode: int, checkpoint: Path) -> dict | None:
        stem = f"episode_{episode:06d}"
        report, trajectory = self.results / f"{stem}.json", self.results / f"{stem}.parquet"
        if not (report.is_file() and trajectory.is_file()):
            return None
        data = json.loads(report.read_text())
        return data if data.get("checkpoint_sha256") == sha256(checkpoint) else None

    def _evaluate_one(self, episode: int, gpu: str) -> None:
        row = self.rows[episode]
        checkpoint = Path(row["evaluated_checkpoint"])
        # Outside the results dir: the kit packer reads every results/**/episode_*.json, and a shard (checkpoint map
        # only) sorting after the real report would replace it, so its cross-check would compare nothing.
        shard = self.results.parent / f"{self.results.name}_shards" / f"episode_{episode:06d}.json"
        shard.parent.mkdir(parents=True, exist_ok=True)
        shard.write_text(json.dumps({str(episode): str(checkpoint)}))
        command = [
            self.args.python, str(self.runner),
            "--eval-script", str(self.eval_script),
            "--runtime-root", str(self.stock),
            "--world-count", str(self.args.world_count),
            "--checkpoints", str(shard),
            "--results", str(self.results),
        ]  # fmt: skip
        if self.args.source_root:
            command += ["--source-root", str(self.args.source_root)]
        log_path = self.results / "logs" / f"episode_{episode:06d}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as stream:
            stream.write(f"[{now()}] gpu {gpu}: {' '.join(command)}\n")
            stream.flush()
            completed = subprocess.run(
                command, cwd=self.kit, env=stock_environment(self.stock, gpu), stdout=stream, stderr=subprocess.STDOUT
            )
        evaluated = self._evaluated(episode, checkpoint)
        if completed.returncode != 0 or evaluated is None:
            row["status"] = f"eval_failed(exit {completed.returncode})"
        else:
            row["status"] = "evaluated"

    def evaluate(self) -> None:
        todo = []
        for episode, row in self.rows.items():
            if "evaluated_checkpoint" not in row:
                continue
            if not self.args.force_eval and self._evaluated(episode, Path(row["evaluated_checkpoint"])) is not None:
                row["status"] = "evaluated"
                continue
            todo.append(episode)
        gpus = [gpu.strip() for gpu in self.args.gpus.split(",") if gpu.strip()]
        log(f"evaluating {len(todo)} episodes on gpus {gpus} ({self.args.jobs_per_gpu} per gpu)")
        work: queue.Queue[int] = queue.Queue()
        for episode in todo:
            work.put(episode)

        def worker(gpu: str) -> None:
            while True:
                try:
                    episode = work.get_nowait()
                except queue.Empty:
                    return
                log(f"ep{episode:02d}: evaluating on gpu {gpu}")
                self._evaluate_one(episode, gpu)
                log(f"ep{episode:02d}: {self.rows[episode]['status']}")

        threads = [
            threading.Thread(target=worker, args=(gpu,), daemon=True)
            for gpu in gpus
            for _ in range(self.args.jobs_per_gpu)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    # ------------------------------------------------------------------ 4. pack
    def pack(self) -> dict:
        args = self.args
        evaluated = sorted(e for e, r in self.rows.items() if r["status"] == "evaluated")
        result = {"packed": False, "episodes": evaluated}
        if args.skip_pack:
            result["reason"] = "--skip-pack"
            return result
        if not args.code_commit:
            result["reason"] = "no --code-commit given; the packer requires a GitHub /commit/<sha> URL"
            log(f"skip packing: {result['reason']}")
            return result
        track_config = json.loads((self.kit / "config" / "tracks.json").read_text())
        if args.track not in track_config["tracks"]:
            raise SystemExit(f"unknown track {args.track}")
        complete = args.episodes == "all" and all(r["status"] == "evaluated" for r in self.rows.values())
        command = [
            str(args.pack_python), str(self.packer),
            "--track", args.track,
            "--results", str(self.results),
            "--dataset-root", str(args.dataset_root),
            "--code-commit", args.code_commit,
            "--eval-script", str(self.eval_script),
        ]  # fmt: skip
        suffix = args.track.replace("track_2_", "")
        if complete:
            output = self.arm_dir / f"{args.arm}_{suffix}.csv"
        elif args.partial_pack and evaluated:
            partial = self.arm_dir / "pack_partial"
            partial.mkdir(parents=True, exist_ok=True)
            track_config["tracks"][args.track]["scored_episodes"] = evaluated
            (partial / "tracks.json").write_text(json.dumps(track_config, indent=2))
            command += ["--config", str(partial / "tracks.json")]
            output = self.arm_dir / f"{args.arm}_{suffix}_PARTIAL.csv"
            result["partial_config"] = str(partial / "tracks.json")
        else:
            result["reason"] = "not every scored episode was evaluated (use --partial-pack for a diagnostic subset)"
            log(f"skip packing: {result['reason']}")
            return result
        command += ["-o", str(output)]
        log_path = self.arm_dir / "pack.log"
        environment = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME"}}
        environment["PYTHONDONTWRITEBYTECODE"] = "1"  # the packer imports kit modules; keep the kit clean
        completed = subprocess.run(command, cwd=self.kit, env=environment, capture_output=True, text=True)
        log_path.write_text(f"$ {' '.join(command)}\n{completed.stdout}\n{completed.stderr}")
        result.update({"command": command, "exit": completed.returncode, "log": str(log_path)})
        if completed.returncode == 0 and output.is_file():
            result.update({"packed": True, "csv": str(output), "csv_sha256": sha256(output)})
            match = re.search(r"^locally computed: (.*)$", completed.stdout, re.MULTILINE)
            if match:
                result["local_scores"] = dict(item.split("=", 1) for item in match.group(1).split())
            result["crosscheck_ok"] = "cross-check: the trajectory reproduces" in completed.stdout
        log(f"pack exit {completed.returncode}: {result.get('csv', completed.stderr.strip()[-300:])}")
        return result

    # ------------------------------------------------------------------ 5. report
    def report(self, provenance: dict, packing: dict) -> None:
        for episode, row in self.rows.items():
            data = json.loads((self.results / f"episode_{episode:06d}.json").read_text()) if row["status"] == "evaluated" else {}
            metrics = data.get("metrics", {})
            for key in METRIC_KEYS:
                row[key] = metrics.get(key)
            row["reference_success_rate"] = (data.get("reference_tracking") or {}).get("success_rate")
            row["checkpoint_sha256"] = data.get("checkpoint_sha256")
        columns = [
            "episode", "object", "status", "iteration", "selected_by", "add_auc", "chord_sr", "maniptrans_sr",
            "mppe_cm",
            "object_position_error_m", "object_orientation_error_deg", "reference_success_rate",
            "export_max_abs_diff", "export_max_rel_diff", "export_stock_verified", "export_source_kind",
            "source_checkpoint", "evaluated_checkpoint", "checkpoint_sha256", "sequence",
        ]  # fmt: skip
        csv_path = self.arm_dir / "eval_report.csv"
        with csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in self.rows.values():
                writer.writerow(row)
        evaluated = [row for row in self.rows.values() if row["status"] == "evaluated"]

        def mean(key):
            values = [float(row[key]) for row in evaluated if row.get(key) is not None]
            return sum(values) / len(values) if values else None

        means = {key: mean(key) for key in ("add_auc", "chord_sr", "maniptrans_sr", "mppe_cm")}
        lines = [
            f"# {self.args.arm}: Track 2 evaluation ("
            + (f"final checkpoints: {FINAL_MARKER}, else model_{self.args.iteration}"
               if self.args.checkpoint_from_final else f"model_{self.args.iteration}") + ")",
            "",
            f"- generated {now()}; arm dir `{self.arm_dir}`",
            f"- evaluator `{provenance['eval_script']}` (sha256 `{provenance['eval_script_sha256'][:16]}...`), "
            f"runtime `{provenance['stock_runtime']}`",
            f"- export: {'E2 (coordinate_e2_v1)' if self.export_enabled else 'none (stock checkpoints)'}",
            f"- evaluated {len(evaluated)}/{len(self.rows)} episodes; means over evaluated episodes: "
            + ", ".join(f"{k}={v:.4f}" if v is not None else f"{k}=n/a" for k, v in means.items()),
            f"- packer: {packing.get('csv') or packing.get('reason') or 'failed, see ' + str(packing.get('log'))}"
            + (f"; local scores {packing['local_scores']}" if packing.get("local_scores") else ""),
            "",
            "| ep | object | iteration | status | AUC | CHORD SR | MP-SR | MPPE cm | export max rel diff | stock verified |",
            "|---:|---|---:|---|---:|---:|---:|---:|---:|---|",
        ]

        def fmt(value, spec=".4f"):
            return "" if value is None else format(float(value), spec)

        for row in self.rows.values():
            lines.append(
                f"| {row['episode']} | {row['object']} | {row.get('iteration', '')} | {row['status']} | "
                f"{fmt(row.get('add_auc'))} | "
                f"{fmt(row.get('chord_sr'), '.3f')} | {fmt(row.get('maniptrans_sr'), '.3f')} | "
                f"{fmt(row.get('mppe_cm'), '.2f')} | {fmt(row.get('export_max_rel_diff'), '.1e')} | "
                f"{'' if row.get('export_stock_verified') is None else row['export_stock_verified']} |"
            )
        (self.arm_dir / "eval_report.md").write_text("\n".join(lines) + "\n")
        summary = {"provenance": provenance, "packing": packing, "means": means, "episodes": list(self.rows.values())}
        (self.arm_dir / "eval_report.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
        log(f"report: {csv_path}, {self.arm_dir / 'eval_report.md'}")

    def run(self) -> int:
        provenance = self.preflight()
        self.select(self.episodes())
        self.export()
        self.evaluate()
        packing = self.pack()
        self.report(provenance, packing)
        failed = [e for e, r in self.rows.items() if r["status"] != "evaluated"]
        if failed:
            log(f"not evaluated: {failed}")
        return 0 if not failed else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm-dir", required=True, type=Path, help="launcher output dir with ep{NN}_{object}/ runs")
    parser.add_argument("--arm", required=True, help="arm name; 'B0' disables export by default")
    parser.add_argument("--iteration", type=int, default=20000, help="checkpoint to evaluate (pre-registered last)")
    parser.add_argument(
        "--checkpoint-from-final", action="store_true",
        help=f"per episode, evaluate the checkpoint its {FINAL_MARKER} names (early stop or max_iterations); "
             "episodes without one fall back to model_<--iteration>",
    )  # fmt: skip
    parser.add_argument("--episodes", default="all")
    parser.add_argument("--kit-root", type=Path, default=DEFAULT_KIT, help="an extracted submission kit")
    parser.add_argument("--kit-zip", type=Path, help="extract this kit zip into --kit-dir and use it")
    parser.add_argument("--kit-dir", type=Path, help="NEW directory for --kit-zip extraction")
    parser.add_argument("--stock-runtime", type=Path, default=DEFAULT_STOCK)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--source-root", type=Path, help="evaluation.source_root remap (normally unset)")
    parser.add_argument("--track", default="track_2_tier1")
    parser.add_argument("--code-commit", help="GitHub /commit/<sha> URL for the packer")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument("--world-count", type=int, default=1)
    parser.add_argument("--pack-python", type=Path, default=DEFAULT_PACK_PYTHON)
    parser.add_argument("--python", default=sys.executable, help="FlashCHORD-env python for export and evaluation")
    parser.add_argument("--code", type=Path, default=DEFAULT_CODE, help="DEV root (export only)")
    parser.add_argument("--manifest", type=Path, help="track2_manifest.json (default: $HUMAN_MOTION_DATA_DIR)")
    parser.add_argument("--results", type=Path, help="results dir (default: <arm-dir>/results)")
    parser.add_argument("--export", default="auto", choices=("auto", "yes", "no"))
    parser.add_argument("--export-jobs", type=int, default=4)
    parser.add_argument("--export-tolerance", type=float, help="exporter equivalence tolerance (default: its 1e-5)")
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--skip-pack", action="store_true")
    parser.add_argument("--partial-pack", action="store_true", help="pack the evaluated subset (diagnostic only)")
    parser.add_argument("--allow-eval-script-mismatch", action="store_true")
    args = parser.parse_args(argv)
    if shutil.which(str(args.pack_python)) is None and not args.pack_python.is_file() and not args.skip_pack:
        parser.error(f"--pack-python {args.pack_python} not found")
    return Evaluation(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
