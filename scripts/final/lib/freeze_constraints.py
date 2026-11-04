#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""freeze_constraints.py: a pip constraints file (`name==version` lines) made from a recorded freeze file.

  <python> -I -B freeze_constraints.py FREEZE OUT [--pinned-by FILE]... [--unversioned-in SCRIPT --expect NAME...]

FREEZE   environment/<env>.pip-freeze.txt. Comment, blank, editable (`-e ...`) and URL (`name @ url`) lines are skipped.
OUT      written: one `name==version` line per remaining package, after a comment header.
--pinned-by FILE   a constraints file that pip reads together with OUT (PIP_CONSTRAINT="FILE OUT"). Every package
                   named in both must have the same version (exit 1 otherwise); it is left out of OUT, so the two
                   files never pin one package twice.
--unversioned-in SCRIPT --expect NAME...   SCRIPT's pip installs (lines with "${PIP[@]}") request these packages
                   without a version: the set must equal the --expect names, and each must have a version in OUT or
                   a --pinned-by file (exit 1 otherwise). Their pins are printed.
Exit status 0, 1 on a conflict or a missing pin, 2 on a usage or format error. Standard library only.
"""
from __future__ import annotations

import argparse
import os
import re
import shlex
import sys

sys.dont_write_bytecode = True
NAME = r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"


def canon(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def same_version(a: str, b: str) -> bool:
    """PEP 440 equality for the forms pip freeze writes (release, suffix, +local): 5.1.0 == 5.1.0.0."""
    def norm(v: str):
        m = re.match(r"^v?(\d+(?:\.\d+)*)(.*)$", v.strip().lower())
        if not m:
            return (v.strip().lower(),)
        rel = [int(x) for x in m.group(1).split(".")]
        while len(rel) > 1 and rel[-1] == 0:
            rel.pop()
        return tuple(rel), m.group(2)
    return norm(a) == norm(b)


def read_pins(path: str, what: str) -> tuple[dict[str, tuple[str, str]], list[str]]:
    """{canonical name: (name as written, version)} of the `name==version` lines; skipped lines listed."""
    pins: dict[str, tuple[str, str]] = {}
    skipped: list[str] = []
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("-e ") or " @ " in line:
                skipped.append(line)
                continue
            m = re.fullmatch(rf"({NAME})==([^\s;#]+)", line)
            if not m:
                raise SystemExit(f"[freeze_constraints] ERROR: {what} {path}: not a name==version line: {line!r}")
            key = canon(m.group(1))
            if key in pins and not same_version(pins[key][1], m.group(2)):
                raise SystemExit(f"[freeze_constraints] ERROR: {what} {path}: {m.group(1)} pinned twice")
            pins[key] = (m.group(1), m.group(2))
    return pins, skipped


def unversioned_requests(script: str) -> list[str]:
    """Packages that the script's `"${PIP[@]}" ...` commands name without a version (options and -e paths skipped)."""
    text = open(script).read().replace("\\\n", " ")
    names: list[str] = []
    for line in text.splitlines():
        if '"${PIP[@]}"' not in line:
            continue
        args = re.split(r"&&|\|\||;", line.split('"${PIP[@]}"', 1)[1], maxsplit=1)[0]
        tokens = shlex.split(args)
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok in ("-e", "--editable", "-r", "--requirement", "-c", "--constraint"):
                i += 2
                continue
            if tok.startswith("-"):
                i += 1
                continue
            m = re.match(rf"({NAME})(\[[^\]]*\])?(.*)$", tok)
            if not m:
                raise SystemExit(f"[freeze_constraints] ERROR: {script}: cannot read the requirement {tok!r}")
            if "==" not in m.group(3):
                names.append(m.group(1))
            i += 1
    return names


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("freeze")
    ap.add_argument("out")
    ap.add_argument("--pinned-by", action="append", default=[])
    ap.add_argument("--unversioned-in")
    ap.add_argument("--expect", nargs="+", default=[])
    a = ap.parse_args()
    if bool(a.unversioned_in) != bool(a.expect):
        ap.error("--unversioned-in and --expect go together")

    freeze, skipped = read_pins(a.freeze, "freeze")
    other: dict[str, tuple[str, str, str]] = {}
    for path in a.pinned_by:
        pins, _ = read_pins(path, "constraints")
        for key, (name, version) in pins.items():
            other[key] = (name, version, os.path.basename(path))

    conflicts, overlap = [], []
    for key, (name, version, src) in sorted(other.items()):
        if key in freeze:
            if same_version(freeze[key][1], version):
                overlap.append(f"{name}=={version}")
            else:
                conflicts.append(f"{name}: {src} pins {version}, the freeze records {freeze[key][1]}")
    for c in conflicts:
        print(f"[freeze_constraints] CONFLICT {c}")
    if conflicts:
        print(f"[freeze_constraints] ERROR: {len(conflicts)} conflict(s); {a.out} not written")
        return 1

    lines = [f"{name}=={version}" for key, (name, version) in sorted(freeze.items()) if key not in other]
    status = 0
    if a.unversioned_in:
        found = {canon(n): n for n in unversioned_requests(a.unversioned_in)}
        expect = {canon(n): n for n in a.expect}
        script = os.path.basename(a.unversioned_in)
        for key in sorted(set(found) - set(expect)):
            print(f"[freeze_constraints] ERROR: {script} installs {found[key]} without a version; it is not in --expect")
            status = 1
        for key in sorted(set(expect) - set(found)):
            print(f"[freeze_constraints] ERROR: --expect {expect[key]}: {script} does not install it without a version")
            status = 1
        for key in sorted(expect):
            if key in freeze:
                print(f"[freeze_constraints] {script} asks for {expect[key]} without a version: pinned to "
                      f"{freeze[key][0]}=={freeze[key][1]} (recorded)")
            elif key in other:
                print(f"[freeze_constraints] {script} asks for {expect[key]} without a version: pinned to "
                      f"{other[key][0]}=={other[key][1]} ({other[key][2]})")
            else:
                print(f"[freeze_constraints] ERROR: {expect[key]} has no recorded version")
                status = 1
    if status:
        print(f"[freeze_constraints] ERROR: {a.out} not written")
        return status

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        f.write(f"# generated by scripts/final/lib/freeze_constraints.py from {os.path.basename(a.freeze)}: "
                f"{len(lines)} name==version pins\n")
        if a.pinned_by:
            f.write(f"# left out ({len(overlap)}, same version in {', '.join(os.path.basename(p) for p in a.pinned_by)}): "
                    + " ".join(overlap) + "\n")
        f.write(f"# skipped ({len(skipped)} editable or URL lines): "
                + " ".join(s.split(" @ ")[0] if " @ " in s else s.rsplit("#egg=", 1)[-1].split("&")[0] for s in skipped)
                + "\n")
        f.write("\n".join(lines) + "\n")
    print(f"[freeze_constraints] wrote {a.out}: {len(lines)} pins from {os.path.basename(a.freeze)}; "
          f"{len(overlap)} packages pinned by --pinned-by with the same version left out, 0 conflicts; "
          f"{len(skipped)} editable/URL lines skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
