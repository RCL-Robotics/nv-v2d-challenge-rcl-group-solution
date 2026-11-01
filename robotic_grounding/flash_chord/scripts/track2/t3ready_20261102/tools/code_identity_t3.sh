#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# code_identity_t3.sh: is the training / export / eval-wrapper / retarget code of this commit the code that produced the
# Tier-3 FINAL_r policies and inputs? Read-only git object comparisons (no checkout, no fetch).
#
#   bash code_identity_t3.sh <this repo> <tree-ish> <fc_r2 repo> <nv repo>
#     <tree-ish>  the commit to check, or before committing the staged tree (git write-tree); its
#                 robotic_grounding/flash_chord/src must equal 41403d3's (the parent)
#     <fc_r2 repo> a clone holding 61fa5d3 (the FlashCHORD tree that trained and exported every Tier-3 policy)
#     <nv repo>    a clone holding f2923d6 (the robotic_grounding tree whose soma_to_g1.py retargeted the f0 inputs)
#   Item 6 also reads the fc_r2 working tree's asset files (not tracked in fc_r2) and compares them with this tree's
#   Git-LFS pointers (sha256 oid) or plain blobs; python: $PYTHON (default python3).
set -euo pipefail
REPO=$1; C=$2; FC=$3; NV=$4
echo "checked: $C (tree $(git -C "$REPO" rev-parse "$C^{tree}"))"
if [ "$(git -C "$REPO" cat-file -t "$C")" = commit ]; then echo "parent: $(git -C "$REPO" rev-parse "$C^")"; fi
a=$(git -C "$REPO" rev-parse "$C:robotic_grounding/flash_chord/src"); b=$(git -C "$REPO" rev-parse "41403d3:robotic_grounding/flash_chord/src")
echo "1. robotic_grounding/flash_chord/src tree: $C $a, 41403d3 $b -> $([ "$a" = "$b" ] && echo IDENTICAL || echo DIFFERENT)"
x=$(git -C "$REPO" diff --name-only 41403d3 "$C" | grep -v '^robotic_grounding/flash_chord/scripts/track2/t3ready_20261102/' | grep -v '^TIER3_SUBMISSION.md$' || true)
echo "2. paths changed vs 41403d3 outside the t3ready_20261102 bundle and TIER3_SUBMISSION.md: ${x:-none}"
T1=$(mktemp); T2=$(mktemp)
git -C "$FC" ls-tree -r 61fa5d3 | awk '{print $3"\t"$4}' | sort -k2 > "$T1"
git -C "$REPO" ls-tree -r "$C:robotic_grounding/flash_chord" | awk '{print $3"\t"$4}' | sort -k2 > "$T2"
n1=$(wc -l < "$T1"); same=$(join -t $'\t' -1 2 -2 2 "$T1" "$T2" | awk -F'\t' '$2==$3' | wc -l)
diff_list=$(join -t $'\t' -1 2 -2 2 "$T1" "$T2" | awk -F'\t' '$2!=$3{print $1}' | tr '\n' ' ')
only_fc=$(comm -23 <(cut -f2 "$T1") <(cut -f2 "$T2") | tr '\n' ' ')
echo "3. fc_r2 @ 61fa5d3 tracked files: $n1; same blob in $C:robotic_grounding/flash_chord: $same"
echo "   different blob (Git-LFS media pointers vs content): ${diff_list:-none}"
echo "   missing here: ${only_fc:-none}"
echo "   only here (not tracked in fc_r2): $(comm -13 <(cut -f2 "$T1") <(cut -f2 "$T2") | awk -F/ '{print $1"/"$2"/"$3}' | sort | uniq -c | awk '{printf "%s (%s files) ", $2, $1}')"
rm -f "$T1" "$T2"
s1=$(git -C "$REPO" rev-parse "$C:robotic_grounding/scripts/retarget/soma_to_g1.py"); s2=$(git -C "$NV" rev-parse "f2923d6:robotic_grounding/scripts/retarget/soma_to_g1.py")
echo "4. soma_to_g1.py blob: $C $s1, nv f2923d6 $s2 -> $([ "$s1" = "$s2" ] && echo IDENTICAL || echo DIFFERENT)"
for f in scripts/track2/eval_all.py src/flash_chord/training/ppo/coordinate/export.py scripts/train_rl.py; do
  echo "5. sha256 $f: $(git -C "$REPO" show "$C:robotic_grounding/flash_chord/$f" | sha256sum | cut -c1-64)"
done
echo "   pins (t3_f0_final PREREG section 10): eval_all.py 657295921e820a464160039ad9a65ba92a683108a99dcb197290ae4147363197, export.py 34e3d7f3a7dca4fdd1e7c4d26750c6332abcc1d629f6a9c6c7a965833a22206f"
"${PYTHON:-python3}" - "$REPO" "$C" "$FC/src/flash_chord/assets" <<'PYEOF'
import hashlib, os, subprocess, sys
repo, c, fc = sys.argv[1:4]
ls = subprocess.run(["git", "-C", repo, "ls-tree", "-r", f"{c}:robotic_grounding/flash_chord/src/flash_chord/assets"],
                    capture_output=True, text=True, check=True).stdout.splitlines()
n = lfs_same = plain_same = 0
bad = []
for line in ls:
    meta, path = line.split("\t", 1)
    oid = meta.split()[2]
    n += 1
    ptr = subprocess.run(["git", "-C", repo, "cat-file", "-p", oid], capture_output=True, check=True).stdout
    f = os.path.join(fc, path)
    if not os.path.isfile(f):
        bad.append(f"missing {path}")
        continue
    data = open(f, "rb").read()
    if ptr.startswith(b"version https://git-lfs"):
        want = [x.split(b":", 1)[1].decode() for x in ptr.splitlines() if x.startswith(b"oid sha256:")][0]
        if hashlib.sha256(data).hexdigest() == want:
            lfs_same += 1
        else:
            bad.append(f"lfs content differs {path}")
    elif data == ptr:
        plain_same += 1
    else:
        bad.append(f"differs {path}")
print(f"6. src/flash_chord/assets: {n} files here; fc_r2 working-tree files equal: {lfs_same} Git-LFS objects (sha256 = "
      f"pointer oid) + {plain_same} plain files; problems: {bad[:5] if bad else 'none'}")
PYEOF
