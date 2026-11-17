"""Which tokens of round r become surprising when earlier rounds' thinking is dropped? (data: strip_A.jsonl, BF16 PEAR)"""
import json, re, sys, collections, statistics as st
from tokenizers import Tokenizer
TOK, R = sys.argv[1], sys.argv[2]
T = Tokenizer.from_file(TOK)
rs = [json.loads(l) for l in open(R + 'pear_rl_delta_quant/results/pilot/strip_A.jsonl')]
cat = collections.defaultdict(lambda: [0.0, 0])
spans, flips = [], []
for r in rs:
    rows = [x for x in map(json.loads, open(R + f"pear_specdec/results/dflash2_main/{r['episode']}.jsonl")) if x.get('generated')]
    g = rows[r['round']]['generated']
    if '</think>' not in g: continue
    th, ans = g.split('</think>', 1)
    tt = [T.decode([i]) for i in T.encode(th, add_special_tokens=False).ids]
    ta = [T.decode([i]) for i in T.encode(ans, add_special_tokens=False).ids]
    F = [t for t in r['full']['tokens'] if t[0] == 't'][1:]; S = [t for t in r['strip']['tokens'] if t[0] == 't'][1:]
    Fa = [t for t in r['full']['tokens'] if t[0] == 'a']; Sa = [t for t in r['strip']['tokens'] if t[0] == 'a']
    if len(F) != len(tt) or len(Fa) != len(ta):
        continue
    for i, w in enumerate(tt):
        d = F[i][1] - S[i][1]
        k = 'digit' if re.search(r'\d', w) else ('punct/space' if not re.search(r'[A-Za-z]', w) else 'word')
        cat[k][0] += d; cat[k][1] += 1
        if d > 3:
            spans.append((round(d, 1), r['episode'], r['round'], ''.join(tt[max(0, i - 12):i]).replace('\n', ' '), w, ''.join(tt[i + 1:i + 4]).replace('\n', ' ')))
    for i, w in enumerate(ta):
        if Fa[i][2] != Sa[i][2] or Fa[i][1] - Sa[i][1] > 1:
            flips.append((round(Fa[i][1] - Sa[i][1], 2), r['episode'], r['round'], ''.join(ta[max(0, i - 8):i]).replace('\n', ' '), w))
print('== dNLL by token type (thinking, strip - full), nats/token and share of total')
tot = sum(v[0] for v in cat.values())
for k, (s, n) in sorted(cat.items()):
    print(f'  {k:12s} n={n:7d}  mean {s / n:+.3f}  share of total dNLL {s / tot:.2f}')
print(f'== {len(spans)} thinking tokens with dNLL > 3 nats; random sample of 30:')
import random; random.seed(1)
for s in random.sample(spans, min(30, len(spans))):
    print('  ', s[0], s[1], s[2], '|', s[3][-70:], '[[' + s[4] + ']]', s[5])
print(f'== {len(flips)} action tokens that flip top-1 or lose > 1 nat:')
for f in sorted(flips, reverse=True)[:40]:
    print('  ', f)
