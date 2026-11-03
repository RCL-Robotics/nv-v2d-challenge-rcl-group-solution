"""Data-identity check: which of the 30 release agent runs' packed inputs can be rebuilt from data on this server.
Rebuilds the files export_inputs.py writes by copy/format (video.mp4, object_mesh.glb, ground_plane.json, task.txt,
episode.json) and compares sha256 with each run_manifest.txt. No GT/clean arrays read; no scores."""
import hashlib, json, sys, io, os
TREE, DS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
def sha(b): return hashlib.sha256(b).hexdigest()
def fsha(p): return sha(open(p, 'rb').read())
meta = {d["episode_index"]: d for d in map(json.loads, open(f"{DS}/meta/episodes_metadata.jsonl"))}
res, tot = {}, {}
for i in range(30):
    run = f"{'CL2' if i <= 10 else 'CL2b' if i <= 21 else 'CL2c'}_ep{i:03d}"
    man = open(f"{TREE}/agent_records/{run}/run_manifest.txt").read().splitlines()
    want = {l.split()[1][2:]: l.split()[0] for l in man if len(l.split()) == 2 and l.split()[1].startswith('./')}
    m = meta[i]; ep = f"episode_{i:06d}"
    got = {
      'video.mp4': fsha(f"{DS}/videos/chunk-000/observation.images.exo_camera/{ep}.mp4"),
      'object_mesh.glb': fsha(f"{DS}/{m['mesh']}"),
      'ground_plane.json': fsha(f"{DS}/{m['ground_plane']}"),
      'task.txt': sha((m['task'] + "\n").encode()),
    }
    s = io.StringIO(); json.dump({"episode_index": i, "object": m["object"], "frames": m["frames"]}, s, indent=1)
    got['episode.json'] = sha(s.getvalue().encode())
    r = {k: (got[k] == want.get(k)) for k in got}
    r['not_rebuildable_here'] = sorted(k for k in want if k not in got and k != '.preexported')
    res[i] = {'run': run, **r}
    for k in got: tot[k] = tot.get(k, 0) + int(r[k])
json.dump({'per_episode': res, 'totals_of_30': tot}, open(f"{OUT}/agent_inputs_rebuild.json", 'w'), indent=1)
print('totals (of 30):', tot)
print('not rebuildable here:', sorted({x for v in res.values() for x in v['not_rebuildable_here']}))
