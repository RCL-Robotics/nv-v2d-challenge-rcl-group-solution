# SPDX-License-Identifier: Apache-2.0
"""scripts/track2/eval_all.py round-2 options (CPU, no evaluator): --checkpoint-from-final selects each episode's
final.json checkpoint, --export-tolerance reaches the exporter, and checkpoint shards stay out of --results."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "track2" / "eval_all.py"


@pytest.fixture(scope="module")
def eval_all():
    spec = importlib.util.spec_from_file_location("track2_eval_all", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _arm(tmp_path):
    """ep00: early-stopped (final.json at 17000); ep01: no final.json; ep02: final.json naming a missing file;
    ep03: malformed final.json; ep04: final.json at max_iterations."""
    arm = tmp_path / "arm"
    for ep in range(5):
        (arm / f"ep{ep:02d}").mkdir(parents=True)
    for ep, iteration in ((0, 17000), (1, 45000), (4, 45000)):
        (arm / f"ep{ep:02d}" / f"model_{iteration}.safetensors").write_bytes(b"x")
    (arm / "ep00" / "model_45000.safetensors").write_bytes(b"stale")  # must not be picked over final.json
    markers = {
        0: {"iteration": 17000, "reason": "help_zero_tail", "checkpoint": "model_17000.safetensors"},
        2: {"iteration": 30000, "reason": "help_zero_tail", "checkpoint": "model_30000.safetensors"},
        3: {"iteration": 30000, "reason": "help_zero_tail", "checkpoint": "../elsewhere/model_30000.safetensors"},
        4: {"iteration": 45000, "reason": "max_iterations", "checkpoint": "model_45000.safetensors"},
    }
    for ep, marker in markers.items():
        (arm / f"ep{ep:02d}" / "final.json").write_text(json.dumps(marker))
    return arm


def _evaluation(eval_all, tmp_path, *extra):
    """Parse argv with eval_all.main (its parser lives there), then build the real Evaluation from those args."""
    arm = _arm(tmp_path)
    argv = ["--arm-dir", str(arm), "--arm", "C3r2", "--iteration", "45000", "--skip-pack",
            "--kit-root", str(tmp_path / "kit"), "--results", str(tmp_path / "results"), *extra]  # fmt: skip
    captured = {}

    class Capture:
        def __init__(self, args):
            captured["args"] = args

        def run(self):
            return 0

    original = eval_all.Evaluation
    eval_all.Evaluation = Capture
    try:
        assert eval_all.main(argv) == 0
    finally:
        eval_all.Evaluation = original
    return original(captured["args"]), arm


def _episodes():
    return {ep: (f"seq{ep}", f"obj{ep}") for ep in range(5)}


def test_checkpoint_from_final_selects_each_episodes_final_checkpoint(eval_all, tmp_path):
    evaluation, arm = _evaluation(eval_all, tmp_path, "--checkpoint-from-final")
    evaluation.select(_episodes())
    rows = evaluation.rows
    assert (rows[0]["status"], rows[0]["iteration"], Path(rows[0]["source_checkpoint"]).name) == (
        "selected", 17000, "model_17000.safetensors")
    assert rows[0]["selected_by"] == "final.json (help_zero_tail)" and rows[0]["final_reason"] == "help_zero_tail"
    assert (rows[1]["status"], rows[1]["iteration"], rows[1]["selected_by"]) == (
        "selected", 45000, "no final.json: --iteration 45000")
    assert (rows[2]["status"], rows[2]["iteration"]) == ("missing_checkpoint", 30000)
    assert rows[3]["status"] == "bad_final_json"
    assert (rows[4]["status"], rows[4]["iteration"], rows[4]["final_reason"]) == ("selected", 45000, "max_iterations")
    assert all(Path(r["source_checkpoint"]).parent.parent == arm for r in rows.values())


def test_without_the_flag_the_selection_is_unchanged(eval_all, tmp_path):
    evaluation, _ = _evaluation(eval_all, tmp_path)
    evaluation.select(_episodes())
    names = {ep: (r["status"], Path(r["source_checkpoint"]).name) for ep, r in evaluation.rows.items()}
    assert names == {
        0: ("selected", "model_45000.safetensors"), 1: ("selected", "model_45000.safetensors"),
        2: ("missing_checkpoint", "model_45000.safetensors"), 3: ("missing_checkpoint", "model_45000.safetensors"),
        4: ("selected", "model_45000.safetensors"),
    }  # fmt: skip


def test_read_final_marker_validates(eval_all, tmp_path):
    assert eval_all.read_final_marker(tmp_path) is None
    for bad in ({"iteration": 0, "checkpoint": "model_0.safetensors"}, {"iteration": True, "checkpoint": "x"},
                {"iteration": 5, "checkpoint": "model_6.safetensors"}, {"iteration": "5"}):
        (tmp_path / "final.json").write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            eval_all.read_final_marker(tmp_path)


@pytest.mark.parametrize("tolerance", [None, 1e-4])
def test_export_tolerance_reaches_the_exporter(eval_all, tmp_path, monkeypatch, tolerance):
    extra = ["--checkpoint-from-final"] + ([] if tolerance is None else ["--export-tolerance", str(tolerance)])
    evaluation, _ = _evaluation(eval_all, tmp_path, *extra)
    evaluation.select(_episodes())
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(eval_all.subprocess, "run", fake_run)
    evaluation._export_one(0)
    command = commands[0]
    assert command[command.index("--input") + 1].endswith("ep00/model_17000.safetensors")
    assert command[command.index("--output") + 1].endswith("export/ep00/model_17000.safetensors")
    if tolerance is None:
        assert "--tolerance" not in command
    else:
        assert command[command.index("--tolerance") + 1] == "0.0001"


def test_checkpoint_shards_stay_out_of_the_results_dir(eval_all, tmp_path, monkeypatch):
    evaluation, _ = _evaluation(eval_all, tmp_path, "--checkpoint-from-final")
    evaluation.select(_episodes())
    evaluation.rows[0]["evaluated_checkpoint"] = evaluation.rows[0]["source_checkpoint"]
    commands = []
    monkeypatch.setattr(eval_all.subprocess, "run",
                        lambda command, **kwargs: commands.append(command) or subprocess.CompletedProcess(command, 1))
    evaluation._evaluate_one(0, "0")
    shard = Path(commands[0][commands[0].index("--checkpoints") + 1])
    results = (tmp_path / "results").resolve()
    assert shard.parent == results.parent / "results_shards" and not shard.is_relative_to(results)
    assert json.loads(shard.read_text()) == {"0": evaluation.rows[0]["source_checkpoint"]}
