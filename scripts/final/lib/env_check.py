#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""env_check.py: compare the packages installed in the running Python with a recorded freeze file.

  <env python> -I -B env_check.py environment/<env>.pip-freeze.txt

Prints every package whose installed version differs from the record, is missing, or is extra, and the totals.
Lines of the form `name @ url` and `-e git+...#egg=name` are compared by presence only. Exit 0 always (informational:
an install on another machine can differ in packages that no step uses); the summary line says how many differ.
Standard library only.
"""
from __future__ import annotations

import re
import sys
from importlib import metadata

sys.dont_write_bytecode = True


def canon(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def main() -> int:
    path = sys.argv[1]
    recorded: dict[str, str | None] = {}
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
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
    installed = {canon(d.metadata["Name"]): d.version for d in metadata.distributions() if d.metadata["Name"]}
    differ = sorted(n for n, v in recorded.items() if v and n in installed and installed[n] != v)
    missing = sorted(n for n in recorded if n not in installed)
    extra = sorted(n for n in installed if n not in recorded)
    for n in differ:
        print(f"[env differs] {n}: installed {installed[n]}, recorded {recorded[n]}")
    for n in missing:
        print(f"[env missing] {n}")
    for n in extra:
        print(f"[env extra] {n}=={installed[n]}")
    print(f"[env_check] {sys.executable} vs {path}: {len(recorded)} recorded packages; {len(differ)} differ, "
          f"{len(missing)} missing, {len(extra)} extra")
    return 0


if __name__ == "__main__":
    sys.exit(main())
