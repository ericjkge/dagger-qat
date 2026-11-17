import json, re, sys, urllib.request
sys.path.insert(0, "/workspace")
FRAME = re.compile(r"(image|camera|wrist|agentview|pixel|px)\b[^.\n]{0,80}([+-]\s?[xyz]\b|\b(left|right|up|down)\b\s*(=|->|means|corresponds))|"
                   r"([+-]\s?[xyz]\b[^.\n]{0,40}\b(image|camera|pixel|px|left|right|up|down)\b)|(px|pixels?)\s*(per|/)\s*(cm|mm)|\bcm\s*(per|/)\s*(px|pixel)|focal|FOV|\bf\s*=", re.I)


def split_gen(g):
    th, sep, ans = g.partition("</think>")
    return th, ans


def geo_only(th):
    return "\n\n".join(p.strip() for p in re.split(r"\n\s*\n", th) if p.strip() and FRAME.search(p))


def post(url, body, timeout=7200):
    req = urllib.request.Request(url + "/generate", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())
