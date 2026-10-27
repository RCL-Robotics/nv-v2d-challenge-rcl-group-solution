#!/usr/bin/env python3
"""Audit one agent run: token usage, commands executed, network requests, suspicious accesses. Usage: audit_run.py <workspace>"""
import json
import re
import sys

ws = sys.argv[1]
usage, cmds, msgs, files = {}, [], [], []
for line in open(f"{ws}/logs/codex_events.jsonl"):
    try:
        e = json.loads(line)
    except json.JSONDecodeError:
        continue
    if e.get("type") == "turn.completed":
        for k, v in e.get("usage", {}).items():
            usage[k] = usage.get(k, 0) + v
    it = e.get("item", {})
    if e.get("type") == "item.completed":
        if it.get("type") == "command_execution":
            cmds.append((it.get("command", ""), it.get("exit_code")))
        elif it.get("type") == "agent_message":
            msgs.append(it.get("text", ""))
        elif it.get("type") == "file_change":
            files += [c.get("path") for c in it.get("changes", [])]
proxy = open(f"{ws}/logs/proxy.log").read().splitlines() if __import__("os").path.isfile(f"{ws}/logs/proxy.log") else []
sus = re.compile(r"curl|wget|pip |huggingface|hf_hub|requests\.|urllib|git clone|/mnt|ref_pose|parquet|diag_only|ssh ")
print("usage:", json.dumps(usage))
print(f"commands: {len(cmds)}  failed: {sum(1 for _, c in cmds if c not in (0, None))}  agent messages: {len(msgs)}  files changed: {len(set(files))}")
print("proxy:", sum("ALLOW" in p for p in proxy), "allowed,", sum("DENY" in p for p in proxy), "denied:",
      sorted({p.split()[3] for p in proxy if "DENY" in p}))
flag = [c for c, _ in cmds if sus.search(c)]
print("flagged commands:", len(flag))
for c in flag:
    print("  !", c[:200].replace("\n", " "))
if msgs:
    print("last message:\n", msgs[-1][:3000])
