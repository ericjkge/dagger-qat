"""Per-context reasoning-length spread: 9 independent samples per recorded context (8 replay-bench samples from lossless
configs at temperature 1.0 + the original episode sample). Expected min over k chains = sequential tokens if k chains race."""
import json, collections, itertools, statistics as st, sys
bench, main = sys.argv[1], sys.argv[2]
d = collections.defaultdict(list)
for f in ['ar', 'dflash2_k3', 'dflash2_k7', 'ngram']:
    for l in open(f'{bench}/{f}.jsonl'):
        r = json.loads(l); d[(r['episode'], r['round'])].append(r['completion_tokens'])
for (ep, rd) in d:
    for l in open(f'{main}/{ep}.jsonl'):
        r = json.loads(l)
        if r['round'] == rd: d[(ep, rd)].append(r['requests'][0]['meta']['completion_tokens'])
emin = lambda v, k: sum(min(c) for c in itertools.combinations(v, k)) / len(list(itertools.combinations(v, k)))
tot = {k: 0.0 for k in (1, 2, 4, 8)}
for key, v in sorted(d.items()):
    v = [min(x, 8192) for x in v]
    for k in tot: tot[k] += emin(v, k)
    print(key, len(v), round(st.mean(v)), round(emin(v, 4)), min(v), max(v), round(st.pstdev(v) / st.mean(v), 2))
print({k: round(t) for k, t in tot.items()}, 'mean/Emin2=%.2f Emin4=%.2f Emin8=%.2f' % (tot[1] / tot[2], tot[1] / tot[4], tot[1] / tot[8]))
