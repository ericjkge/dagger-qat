"""Local evidence for four robot-use-LLM efficiency ideas, from the recorded PEAR episodes (no GPU)."""
import json, glob, collections, math, statistics as st, sys
MAIN, BENCH = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for f in sorted(glob.glob(MAIN + '/libero_*.jsonl')) for l in open(f)]
eps = collections.defaultdict(list)
for r in rows: eps[r['episode']].append(r)
for v in eps.values(): v.sort(key=lambda r: r['round'])
think = lambda r: r['requests'][0]['meta']['completion_tokens']
ok = lambda r: think(r) < 40000

print('== 1. Adaptive effort: how much of reasoning length is predictable from the context? ==')
d = collections.defaultdict(list)
for f in ['ar', 'dflash2_k3', 'dflash2_k7', 'ngram']:
    for l in open(f'{BENCH}/{f}.jsonl'):
        r = json.loads(l); d[(r['episode'], r['round'])].append(r['completion_tokens'])
for (ep, rd) in d: d[(ep, rd)].append(think(next(r for r in eps[ep] if r['round'] == rd)))
logs = {k: [math.log(max(1, min(x, 8192))) for x in v] for k, v in d.items()}
allv = [x for v in logs.values() for x in v]; gm = st.mean(allv)
within = sum((x - st.mean(v)) ** 2 for v in logs.values() for x in v)
total = sum((x - gm) ** 2 for x in allv)
print(f'  12 contexts x 9 samples, log length: between-context share of variance = {1 - within / total:.2f}, within-context (sampling noise) = {within / total:.2f}')

def phase(prev):
    if prev is None: return 'first'
    a = prev['action']; arm = a['arm']; g = a['gripper']['width']
    if abs(g) > 1e-9: return 'gripper'
    if any(abs(arm[k]) > 1e-9 for k in ('roll', 'pitch', 'yaw')): return 'rotate'
    if arm['z'] < -1e-9 and abs(arm['x']) < 3 and abs(arm['y']) < 3: return 'descend'
    return 'translate'
feats = []
for ep, rs in eps.items():
    prev = None
    for r in rs:
        if ok(r): feats.append((phase(prev), (prev or {}).get('outcome', 'none'), math.log(max(1, think(r)))))
        prev = r
y = [f[2] for f in feats]; m = st.mean(y); sst = sum((v - m) ** 2 for v in y)
for name, key in [('previous action type', lambda f: f[0]), ('previous outcome', lambda f: f[1]), ('both', lambda f: (f[0], f[1]))]:
    g = collections.defaultdict(list)
    for f in feats: g[key(f)].append(f[2])
    sse = sum((v - st.mean(vs)) ** 2 for vs in g.values() for v in vs)
    print(f'  across {len(feats)} rounds, R^2 of log length explained by {name}: {1 - sse / sst:.2f}')
g = collections.defaultdict(list)
for f in feats: g[f[0]].append(math.exp(f[2]))
print('  median thinking tokens by previous action type:', {k: round(st.median(v)) for k, v in sorted(g.items())})

print('== 3. Thinking while the arm moves: motion time vs reasoning time per round ==')
mot, rea, sim = [], [], []
for r in rows:
    if not ok(r) or not r.get('executed'): continue
    mot.append(r['executed'] / 20.0)                      # LIBERO control at 20 Hz -> physical seconds
    rea.append(think(r) / 747.0)                         # PEAR's GB300 decode speed
    sim.append(r['env_s'])
print(f'  physical motion per round: mean {st.mean(mot):.2f} s, median {st.median(mot):.2f} s (env steps / 20 Hz)')
print(f'  reasoning per round at PEAR GB300 speed (747 tok/s): mean {st.mean(rea):.2f} s; at RTX 5090 (221 tok/s): {st.mean(rea) * 747 / 221:.2f} s')
for name, rs in [('GB300', st.mean(rea)), ('RTX 5090', st.mean(rea) * 747 / 221)]:
    print(f'  max round-time saved by perfect overlap ({name}): {st.mean(mot) / (st.mean(mot) + rs):.1%}')

print('== 4. Context buildup: what fills the context ==')
th, img, other, ans = 0, 0, 0, 0
for r in rows:
    if not ok(r): continue
    m0 = r['requests'][0]['meta']; th += m0['completion_tokens']; img += 512  # 2 new frames x 256 tokens per round (image_tokens counts the whole prompt)
    ans += r['requests'][1]['meta']['completion_tokens'] if len(r['requests']) > 1 else 0
first = [rs[0]['requests'][0]['meta']['prompt_tokens'] for rs in eps.values()]
tot = th + img + ans
print(f'  per-round additions: thinking {th / tot:.1%}, images {img / tot:.1%}, answer+action {ans / tot:.1%}; initial prompt ~{st.median(first)} tok')
long = [rs for rs in eps.values() if sum(think(r) for r in rs if ok(r)) > 50000]
for rs in long:
    t = sum(think(r) for r in rs if ok(r)); print(f'  long episode {rs[0]["episode"]}: {len(rs)} rounds, thinking {t} tok, non-thinking adds ~{len(rs) * 600} tok')
