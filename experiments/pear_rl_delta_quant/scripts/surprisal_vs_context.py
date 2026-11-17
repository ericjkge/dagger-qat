"""Mean surprisal (-log p) of sampled reasoning tokens by absolute context position, from recorded episodes."""
import json, glob, statistics as st, collections, sys
B = [0, 8e3, 16e3, 32e3, 64e3, 128e3, 1e9]
nll = collections.defaultdict(list)
for f in glob.glob(sys.argv[1] + '/libero_*.jsonl'):
    for l in open(f):
        r = json.loads(l)
        for q in r['requests']:
            if q['phase'] != 'reason': continue
            m = q['meta']; lps = m['output_logprobs']
            if len(lps) > 40000: continue  # the excluded degenerate round
            for i, lp in enumerate(lps):
                if lp is None: continue
                c = m['prompt_tokens'] + i
                nll[max(j for j in range(len(B) - 1) if c >= B[j])].append(-lp)
for b in sorted(nll):
    v = nll[b]; print(f'{int(B[b] / 1e3)}k+ n={len(v)} mean surprisal={st.mean(v):.3f} nats frac>2nats={st.mean([x > 2 for x in v]):.3f}')
