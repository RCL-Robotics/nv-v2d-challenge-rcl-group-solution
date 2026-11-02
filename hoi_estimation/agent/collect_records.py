#!/usr/bin/env python3
"""Collect every sandboxed agent run (agent_runs/<run>/) into a shareable record tree:
  <out>/<run>/AGENT_TASK.md, run_manifest.txt, stderr.txt (if non-empty)
  <out>/<run>/events.jsonl.gz   raw event stream with base64 images and thinking signatures stripped
  <out>/<run>/transcript.md     readable conversation: agent text, tool calls (truncated) and tool outputs (truncated)
  <out>/<run>/outputs/, tools/  text files and small images only (no npz/npy/pyc, nothing > 2 MB)
  <out>/batch_logs/             batch logs
  <out>/INDEX.md                one row per run
Inputs (dataset copies) and scratch are not copied. Hidden directories (e.g. .codex_home, which holds credentials)
are never read. Usage: collect_records.py <out_dir>"""
import gzip, json, os, re, shutil, sys
SRC = os.environ.get("V2D_ROOT", "/mnt/nas/Group/hanzhou/v2d") + "/agent_runs"; OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
TXT = {".md", ".py", ".json", ".txt", ".log", ".sh", ".csv", ".yaml", ".tsv"}; IMG = {".jpg", ".png"}
SERIES = {"A1": "Codex GPT 修正轨迹，第一版任务", "A2": "Codex GPT 修正轨迹", "A3": "Codex GPT 修正轨迹，开放网络",
          "A4": "Codex GPT 时序修正（temporal 任务）", "C1": "Codex DeepSeek 估计相机", "CL1": "Claude Opus 完整轨迹，第一批",
          "CL2": "Claude Opus 完整轨迹 v2（tier3_opus 发布用，ep0-10）", "CL2b": "Claude Opus v2（ep11-21）",
          "CL2c": "Claude Opus v2（ep22-29）", "D1": "Codex GPT 修正深度/位置", "T1": "Codex DeepSeek 完整轨迹（Tier 1 输入）",
          "smoke": "冒烟测试"}
def cut(s, n): s = str(s); return s if len(s) <= n else s[:n] + f" …[+{len(s) - n} chars]"
def clean(o):
    if isinstance(o, dict):
        if o.get("type") == "image" or ("data" in o and o.get("type") == "base64"): return {"type": "image", "omitted": True}
        return {k: ("<omitted>" if k == "signature" else clean(v)) for k, v in o.items()}
    if isinstance(o, list): return [clean(x) for x in o]
    if isinstance(o, str) and len(o) > 2000 and re.fullmatch(r"[A-Za-z0-9+/=\s]+", o[:2000]): return "<base64 omitted>"
    return o
def blocks(c): return c if isinstance(c, list) else [{"type": "text", "text": c}]
def transcript(ev, md):
    stats = {}
    for l in ev:
        try: d = json.loads(l)
        except ValueError: continue
        t = d.get("type")
        if t == "assistant":
            for b in blocks(d["message"]["content"]):
                if b.get("type") == "text" and b["text"].strip(): md.append(f"**Agent:** {b['text'].strip()}\n")
                elif b.get("type") == "thinking" and b.get("thinking", "").strip(): md.append(f"> *thinking:* {cut(b['thinking'].strip(), 1500)}\n")
                elif b.get("type") == "tool_use":
                    inp = b["input"]; s = inp.get("command") or inp.get("file_path") or json.dumps(inp, ensure_ascii=False)
                    md.append(f"**{b['name']}**\n```\n{cut(s, 1500)}\n```\n")
        elif t == "user":
            for b in blocks(d["message"]["content"]):
                if b.get("type") == "tool_result":
                    c = b.get("content"); c = "\n".join(x.get("text", "[image]") for x in c) if isinstance(c, list) else c
                    md.append(f"<details><summary>output</summary>\n\n```\n{cut(c, 800)}\n```\n</details>\n")
        elif t == "result":
            stats = dict(cost_usd=d.get("total_cost_usd"), turns=d.get("num_turns"), api_s=(d.get("duration_api_ms") or 0) / 1000)
            if d.get("result"): md.append(f"## Final\n\n{d['result']}\n")
        elif t == "item.completed":
            it = d["item"]
            if it["type"] == "command_execution":
                md.append(f"**shell** (exit {it.get('exit_code')})\n```\n{cut(it['command'], 1500)}\n```\n<details><summary>output</summary>\n\n```\n{cut(it.get('aggregated_output', ''), 800)}\n```\n</details>\n")
            elif it["type"] == "agent_message": md.append(f"**Agent:** {it['text']}\n")
            elif it["type"] == "reasoning" and it.get("text"): md.append(f"> *reasoning:* {cut(it['text'], 1500)}\n")
            elif it["type"] == "file_change": md.append(f"**file change:** {cut(json.dumps(it.get('changes'), ensure_ascii=False), 500)}\n")
        elif t == "turn.completed": stats = dict(tokens=d.get("usage"))
    return stats
rows = []
for run in sorted(os.listdir(SRC)):
    r = f"{SRC}/{run}"
    if run.startswith(".") or run == "vis" or not os.path.isdir(r): continue
    if run == "T1_batch" or not os.path.isfile(f"{r}/AGENT_TASK.md"): continue
    o = f"{OUT}/{run}"; os.makedirs(o, exist_ok=True); shutil.copy(f"{r}/AGENT_TASK.md", o)
    lg = f"{r}/logs"; man = {}
    if os.path.isfile(f"{lg}/run_manifest.txt"):
        shutil.copy(f"{lg}/run_manifest.txt", o)
        for l in open(f"{lg}/run_manifest.txt"):
            if ":" in l and not l.startswith(" "): k, v = l.split(":", 1); man.setdefault(k.strip(), v.strip())
    evf = next((f"{lg}/{f}" for f in ("claude_events.jsonl", "codex_events.jsonl") if os.path.isfile(f"{lg}/{f}")), None)
    if evf is None and os.path.isfile(f"{r}/events.jsonl"): evf = f"{r}/events.jsonl"
    for se in (f"{lg}/claude_stderr.txt", f"{lg}/codex_stderr.txt", f"{r}/stderr.txt"):
        if os.path.isfile(se) and os.path.getsize(se): shutil.copy(se, f"{o}/stderr.txt")
    stats = {}
    if evf:
        lines = open(evf, errors="replace").read().splitlines()
        with gzip.open(f"{o}/events.jsonl.gz", "wt") as g:
            for l in lines:
                try: g.write(json.dumps(clean(json.loads(l)), ensure_ascii=False) + "\n")
                except ValueError: g.write(l + "\n")
        md = [f"# {run}\n", f"{SERIES.get(run.split('_')[0], '')}  \n模型: {man.get('model', '?')}  \n日期: {man.get('date_utc', '?')}\n"]
        stats = transcript(lines, md); open(f"{o}/transcript.md", "w").write("\n".join(md))
    for sub in ("outputs", "tools"):
        for root, dirs, files in os.walk(f"{r}/{sub}"):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__pycache__", "torch_home")]
            for f in files:
                ext = os.path.splitext(f)[1].lower(); p = f"{root}/{f}"
                if os.path.islink(p): continue
                sz = os.path.getsize(p)
                if (ext in TXT and sz < 2e6) or (ext in IMG and sz < 2e6):
                    d = f"{o}/{os.path.relpath(root, r)}"; os.makedirs(d, exist_ok=True); shutil.copy(p, d)
    rep = f"{r}/outputs/REPORT.md"; head = ""
    if os.path.isfile(rep): head = next((l.strip("# \n") for l in open(rep) if l.strip()), "")
    ep = re.search(r"ep(\d{3})", run)
    rows.append((run, SERIES.get(run.split("_")[0], ""), man.get("model", "?").split(" ")[0], ep.group(1) if ep else "-",
                 man.get("date_utc", "?")[:10], man.get("wall_s", "?"), stats.get("cost_usd"), "是" if os.path.isfile(rep) else "", cut(head, 60)))
    print(run, flush=True)
bd = f"{OUT}/batch_logs"; os.makedirs(bd, exist_ok=True)
for f in os.listdir(SRC):
    if f.endswith((".log", ".runlog")) and os.path.getsize(f"{SRC}/{f}"): shutil.copy(f"{SRC}/{f}", bd)
if os.path.isfile(f"{SRC}/T1_batch/batch.log"): shutil.copy(f"{SRC}/T1_batch/batch.log", f"{bd}/T1_batch.log")
with open(f"{OUT}/INDEX.md", "w") as f:
    f.write("# Agent 运行记录索引\n\n每个目录：`AGENT_TASK.md`（任务说明）、`transcript.md`（可读对话记录）、`events.jsonl.gz`（原始事件流，去掉了图片和签名）、"
            "`run_manifest.txt`（环境、输入哈希、耗时）、`outputs/`（报告、求解脚本、检查图，不含大数组）。\n\n"
            "| run | 系列 | 模型 | episode | 日期 | 耗时(s) | 花费(USD) | 有报告 | 报告标题 |\n|---|---|---|---|---|---|---|---|---|\n")
    for x in rows: f.write("| " + " | ".join("" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v)) for v in x) + " |\n")
