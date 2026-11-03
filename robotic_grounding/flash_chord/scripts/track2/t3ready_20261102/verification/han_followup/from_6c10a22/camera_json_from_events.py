"""For each release agent run: find the camera.json text the agent printed (tool results in events.jsonl.gz), re-serialise
it the way export_inputs.py writes it, and compare sha256 with run_manifest.txt. Then compare its K and T_world_cam with
the s1 release camera.json (data identity only; no scores)."""
import gzip, hashlib, io, json, re, sys
import numpy as np
TREE, S1, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
eps = json.load(open(f"{S1}/episodes.json"))
seqs = {}
for e in (eps if isinstance(eps, list) else eps.get('episodes', eps)):
    if isinstance(e, dict):
        seqs[e.get('episode_index', e.get('index'))] = e.get('sequence_id', e.get('seq'))
def strings(o):
    if isinstance(o, str): yield o
    elif isinstance(o, dict):
        for v in o.values(): yield from strings(v)
    elif isinstance(o, list):
        for v in o: yield from strings(v)
def candidates(s):
    if '"K_pipeline"' not in s or '"T_world_cam"' not in s: return
    # strip Read-tool line-number prefixes
    t = re.sub(r'(?m)^\s*\d+[\t→]', '', s)
    for m in re.finditer(r'\{\s*"K"\s*:', t):
        depth = 0
        for j in range(m.start(), len(t)):
            if t[j] == '{': depth += 1
            elif t[j] == '}':
                depth -= 1
                if depth == 0:
                    try: yield json.loads(t[m.start():j+1])
                    except Exception: pass
                    break
res = {}; ok_sha = 0; found = 0
for i in range(30):
    run = f"{'CL2' if i <= 10 else 'CL2b' if i <= 21 else 'CL2c'}_ep{i:03d}"
    man = open(f"{TREE}/agent_records/{run}/run_manifest.txt").read()
    want = re.search(r'([0-9a-f]{64})  \./camera\.json', man).group(1)
    best = None
    with gzip.open(f"{TREE}/agent_records/{run}/events.jsonl.gz", 'rt') as f:
        for line in f:
            try: o = json.loads(line)
            except Exception: continue
            for s in strings(o):
                for d in candidates(s):
                    if set(d) >= {"K", "K_pipeline", "T_world_cam"}:
                        b = io.StringIO(); json.dump(d, b, indent=1)
                        h = hashlib.sha256(b.getvalue().encode()).hexdigest()
                        best = (d, h)
                        if h == want: break
                if best and best[1] == want: break
            if best and best[1] == want: break
    r = {'run': run, 'found': best is not None}
    if best:
        found += 1
        d, h = best; r['sha_equal_manifest'] = (h == want); ok_sha += int(h == want)
        seq = seqs.get(i)
        try:
            s1 = json.load(open(f"{S1}/reconstructed_data/{seq}/camera.json"))
            K, K1 = np.array(d['K']), np.array(s1['K']); W, W1 = np.array(d['T_world_cam']), np.array(s1['T_world_cam'])
            r['K_maxabs_vs_s1'] = float(abs(K - K1).max()); r['T_world_cam_maxabs_vs_s1'] = float(abs(W - W1).max())
            r['K_pipeline_fx'] = float(np.array(d['K_pipeline'])[0, 0]); r['K_pipeline'] = d['K_pipeline']; r['K'] = d['K']
        except Exception as e:
            r['s1_cmp_error'] = repr(e)
    res[i] = r
json.dump({'per_episode': res, 'found': found, 'sha_equal_manifest': ok_sha}, open(f"{OUT}/camera_json_from_events.json", 'w'), indent=1)
print('found', found, 'sha_equal_manifest', ok_sha, 'of 30')
for i, r in res.items():
    print(i, r.get('found'), r.get('sha_equal_manifest'), '%.2e' % r.get('K_maxabs_vs_s1', float('nan')), '%.2e' % r.get('T_world_cam_maxabs_vs_s1', float('nan')), r.get('K_pipeline_fx'), r.get('s1_cmp_error', ''))
