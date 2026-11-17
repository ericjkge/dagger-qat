"""Where in each round's thinking does the final action first get stated? (text heuristic)"""
import json, glob, re, statistics as st, sys
D = sys.argv[1]
rows = [json.loads(l) for f in sorted(glob.glob(D + '/libero_*.jsonl')) for l in open(f)]
def nz(a):
    out = {}
    for part, q in a.items():
        for k, v in q.items():
            if abs(v) > 1e-9: out[(part, k)] = v
    return out
res = []
for r in rows:
    g = r['generated']; ct = r['completion_tokens']
    if ct > 40000 or not r.get('action'): continue
    think = g.split('</think>')[0]
    vals = nz(r['action'])
    L = len(think)
    if not vals:
        res.append(None); continue
    # earliest char position p such that every nonzero value's number string has appeared in think[:p]
    pos = []
    ok = True
    for (part, k), v in vals.items():
        s = ('%g' % abs(v))
        m = [mm.start() for mm in re.finditer(r'(?<![\d.])' + re.escape(s) + r'(?![\d])', think)]
        # require value mentioned near the quantity name within 40 chars
        cand = [p for p in m if re.search(k if k != 'width' else 'grip|width', think[max(0, p-60):p+10], re.I)]
        if not cand: ok = False; break
        pos.append(cand)
    if not ok: res.append(('never', L, ct)); continue
    # first position where all have appeared: max over components of each's first mention
    first = max(c[0] for c in pos)
    res.append((first / L, L, ct))
fr = [x for x in res if x and x[0] != 'never']
nev = [x for x in res if x and x[0] == 'never']
print('rounds with nonzero action:', len(fr) + len(nev), 'all values stated before end:', len(fr), 'never matched:', len(nev))
f = [x[0] for x in fr]
print('fraction of thinking BEFORE final values all first stated: median %.2f mean %.2f' % (st.median(f), st.mean(f)))
w = sum((1 - x[0]) * x[2] for x in fr) / sum(x[2] for x in fr)
print('token-weighted share of thinking AFTER first statement of final action: %.2f' % w)
for lo, hi in [(0, 1000), (1000, 3000), (3000, 40000)]:
    s = [x for x in fr if lo <= x[2] < hi]
    if s: print(f'  rounds {lo}-{hi} tok: n={len(s)}, median frac before={st.median([x[0] for x in s]):.2f}, tok-weighted after={sum((1-x[0])*x[2] for x in s)/sum(x[2] for x in s):.2f}')
# dithering markers
mk = re.compile(r'\b(Wait|Hmm|Actually|But wait|Let me re|Hold on|On second thought)\b')
tot_tok = sum(r['completion_tokens'] for r in rows if r['completion_tokens'] < 40000)
n_mk = sum(len(mk.findall(r['generated'].split('</think>')[0])) for r in rows if r['completion_tokens'] < 40000)
print('dithering markers per 1k thinking tokens: %.1f' % (1000 * n_mk / tot_tok))
