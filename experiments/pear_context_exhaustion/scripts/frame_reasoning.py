"""How much of PEAR's reasoning is camera<->robot frame / scale derivation, and is it re-derived every round?
Heuristic paragraph classifier over recorded BF16 episodes (dflash2_main)."""
import json, glob, re, sys, collections, statistics as st
from tokenizers import Tokenizer
T = Tokenizer.from_file(sys.argv[1]); D = sys.argv[2]
FRAME = re.compile(r"(image|camera|wrist|agentview|pixel|px)\b[^.\n]{0,80}([+-]\s?[xyz]\b|\b(left|right|up|down)\b\s*(=|->|means|corresponds))|"
                   r"([+-]\s?[xyz]\b[^.\n]{0,40}\b(image|camera|pixel|px|left|right|up|down)\b)|(px|pixels?)\s*(per|/)\s*(cm|mm)|\bcm\s*(per|/)\s*(px|pixel)|focal|FOV|\bf\s*=", re.I)
HEDGE = re.compile(r"\b(wait|hmm|actually|let me re-?(check|verify|think)|re-verify|confus)", re.I)
per_round, eps = [], collections.defaultdict(list)
for f in sorted(glob.glob(D + '/libero_*.jsonl')):
    rows = [json.loads(l) for l in open(f)]
    for r in rows:
        g = r.get('generated') or ''
        th = g.split('</think>', 1)[0]
        paras = [p for p in re.split(r'\n\s*\n', th) if p.strip()]
        ntok = lambda s: len(T.encode(s, add_special_tokens=False).ids)
        tot = ntok(th) or 1
        fr = sum(ntok(p) for p in paras if FRAME.search(p))
        hed = len(HEDGE.findall(th))
        per_round.append((r['episode'], r['round'], tot, fr, hed))
        eps[r['episode']].append((r['round'], tot, fr))
T_ = sum(x[2] for x in per_round); F_ = sum(x[3] for x in per_round)
print(f'rounds {len(per_round)}; thinking tokens {T_}; in frame/scale-derivation paragraphs {F_} ({F_ / T_:.1%})')
r0 = [x for x in per_round if x[1] == 0]; rl = [x for x in per_round if x[1] > 0]
print(f'round 0: frame share {sum(x[3] for x in r0) / sum(x[2] for x in r0):.1%} of {sum(x[2] for x in r0)} tok; later rounds: {sum(x[3] for x in rl) / sum(x[2] for x in rl):.1%} of {sum(x[2] for x in rl)} tok')
print('share of later rounds that re-derive frame/scale (>=1 frame paragraph):', f'{sum(x[3] > 0 for x in rl) / len(rl):.1%}')
big = sorted(rl, key=lambda x: -x[2])[:12]
print('largest later rounds (ep, round, tokens, frame tokens, hedges):'); [print('  ', b) for b in big]
fh = [x for x in rl if x[3] > .3 * x[2]]; ot = [x for x in rl if x[3] <= .3 * x[2]]
rate = lambda v: 1000 * sum(x[4] for x in v) / max(1, sum(x[2] for x in v))
print(f'hedges per 1k thinking tokens: frame-heavy rounds (>30% frame) {rate(fh):.2f} vs others {rate(ot):.2f}')
import random; random.seed(3)
samp = []
for f in sorted(glob.glob(D + '/libero_*.jsonl')):
    for r in map(json.loads, open(f)):
        th = (r.get('generated') or '').split('</think>', 1)[0]
        for p in re.split(r'\n\s*\n', th):
            if p.strip(): samp.append((bool(FRAME.search(p)), r['episode'], r['round'], p.strip().replace('\n', ' ')))
print('== spot check: 8 classified frame, 6 not')
for s_ in random.sample([x for x in samp if x[0]], 8) + random.sample([x for x in samp if not x[0]], 6):
    print('  ', 'FRAME' if s_[0] else 'other', s_[1], s_[2], '|', s_[3][:260])
