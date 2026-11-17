"""How much of each round's thinking restates earlier rounds' thinking in the same episode?
Sentence-level: a sentence is 'repeated' if its 8-gram set overlaps >=50% with 8-grams of earlier rounds' thinking.
Also: what topics do repeated sentences cover (camera/axis conventions, object localization, gripper geometry)?"""
import json, glob, re, collections
files = [f for f in glob.glob("**/libero_*_r*.jsonl", recursive=True) if "/frames/" not in f]
seen = set(); eps = []
for f in sorted(files):
    R = [json.loads(l) for l in open(f)]
    key = f.split("/")[-1] + json.dumps(R[0].get("action"))
    if len(R) >= 5 and key not in seen and all("generated" in r for r in R): seen.add(key); eps.append(R)
def toks(s): return re.findall(r"[a-z0-9.\-]+", s.lower())
def grams(t, n=8): return {tuple(t[i:i+n]) for i in range(len(t) - n + 1)}
TOP = {"conventions": r"\+x|\+y|-x|-y|points? (toward|to)|image (left|right|top|bottom)|agentview|wrist (camera|view)|flipped|rotated|camera",
       "scale/px": r"px|pixel|cm per|scale|depth|height of|table (height|surface)",
       "localize": r"located|position of|is at|appears|i see|looking at|visible",
       "plan/task": r"task|step|next|then|plan|need to|first"}
words = rep = 0; byround = collections.defaultdict(lambda: [0, 0]); topic = collections.Counter(); topic_all = collections.Counter()
for R in eps:
    prev = set()
    for r in R:
        th = r["generated"].partition("</think>")[0]
        for s in re.split(r"(?<=[.!?\n])\s+", th):
            t = toks(s)
            if len(t) < 8: continue
            g = grams(t); hit = len(g & prev) / len(g) >= 0.5
            words += len(t); rep += len(t) * hit
            k = min(r["round"], 8); byround[k][0] += len(t); byround[k][1] += len(t) * hit
            for name, pat in TOP.items():
                if re.search(pat, s.lower()):
                    topic_all[name] += len(t); topic[name] += len(t) * hit
        prev |= grams(toks(th))
print(f"{len(eps)} episodes; thinking words in repeated sentences: {rep/words:.1%}")
print("by round:", {k: f"{v[1]/max(1,v[0]):.0%}" for k, v in sorted(byround.items())})
print("repeated share by topic:", {k: f"{topic[k]/max(1,topic_all[k]):.0%}" for k in TOP})
