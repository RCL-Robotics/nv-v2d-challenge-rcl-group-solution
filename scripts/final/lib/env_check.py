#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""env_check.py: compare the packages installed in the running Python with a recorded freeze file.

  <env python> -I -B env_check.py [--report-only] environment/<env>.pip-freeze.txt

Prints every package whose installed version differs from the record, is missing, or is extra, and the totals.
Lines of the form `name @ url` and `-e git+...#egg=name` are compared by presence only.
Exit status: 1 if any package differs, is missing or is extra (default), 0 if the env equals the record.
--report-only: list the differences and exit 0 (for an install on another machine that is known to differ).
A comment line of the freeze file may contain `env_check: allow-local-editable-extras=N`: up to N extra packages that
are editable installs from a local directory (another project's code that shared the env; it was left out of the
record) are then reported but not counted as differences. Standard library only.
"""
from __future__ import annotations

import json
import re
import sys
from importlib import metadata

sys.dont_write_bytecode = True


def canon(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def local_editable(dist) -> bool:
    try:
        info = json.loads(dist.read_text("direct_url.json") or "{}")
    except (OSError, ValueError):
        return False
    return bool(info.get("dir_info", {}).get("editable")) and str(info.get("url", "")).startswith("file://")


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--report-only"]
    report_only = len(args) != len(sys.argv) - 1
    if len(args) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    path = args[0]
    recorded: dict[str, str | None] = {}
    allow_local = 0
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            m = re.search(r"env_check: allow-local-editable-extras=(\d+)", line)
            if m:
                allow_local = int(m.group(1))
            continue
        if line.startswith("-e "):
            m = re.search(r"#egg=([A-Za-z0-9_.\-]+)", line)
            if m:
                recorded[canon(m.group(1))] = None
            continue
        if " @ " in line:
            recorded[canon(line.split(" @ ", 1)[0])] = None
            continue
        if "==" in line:
            n, v = line.split("==", 1)
            recorded[canon(n)] = v.strip()
    dists = {canon(d.metadata["Name"]): d for d in metadata.distributions() if d.metadata["Name"]}
    installed = {n: d.version for n, d in dists.items()}
    differ = sorted(n for n, v in recorded.items() if v and n in installed and installed[n] != v)
    missing = sorted(n for n in recorded if n not in installed)
    extra = sorted(n for n in installed if n not in recorded)
    allowed = [n for n in extra if local_editable(dists[n])][:allow_local]
    extra = [n for n in extra if n not in allowed]
    for n in differ:
        print(f"[env differs] {n}: installed {installed[n]}, recorded {recorded[n]}")
    for n in missing:
        print(f"[env missing] {n}")
    for n in extra:
        print(f"[env extra] {n}=={installed[n]}")
    for n in allowed:
        print(f"[env extra, allowed] {n}=={installed[n]} (a local editable install; the record omits it, "
              f"allow-local-editable-extras={allow_local})")
    bad = len(differ) + len(missing) + len(extra)
    verdict = "OK" if bad == 0 else ("DIFFERS (--report-only: not an error)" if report_only else "DIFFERS")
    print(f"[env_check] {sys.executable} vs {path}: {len(recorded)} recorded packages; {len(differ)} differ, "
          f"{len(missing)} missing, {len(extra)} extra" + (f", {len(allowed)} allowed extra" if allowed else "")
          + f": {verdict}")
    return 0 if (bad == 0 or report_only) else 1


if __name__ == "__main__":
    sys.exit(main())
