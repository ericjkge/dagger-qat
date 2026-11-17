"""Pick the signed axis map from robosuite (row,col) to the displayed (down,right) image using ONLY the agentview
convention in PEAR's environment prompt (+x -> image bottom, +y -> image left); then check it independently on the
wrist camera at yaw ~ 0 (+x -> top, +y -> right, so arm +x moves objects down, arm +y moves objects left)."""
import json, itertools, sys
import numpy as np
P = "/workspace/results/pilot/"
cards = [json.loads(l) for l in open(P + "cards.jsonl")]
units_per_cm = cards[0]["units_per_m"] / 100.0
best = None
for perm in ((0, 1), (1, 0)):
    for s in itertools.product((1, -1), repeat=2):
        m = np.zeros((2, 2)); m[0, perm[0]] = s[0]; m[1, perm[1]] = s[1]
        ok = sum(((m @ np.array(c["dA"]["x"]))[0] > abs((m @ np.array(c["dA"]["x"]))[1])) and
                 ((m @ np.array(c["dA"]["y"]))[1] < -abs((m @ np.array(c["dA"]["y"]))[0])) for c in cards)
        if best is None or ok > best[0]:
            best = (ok, m)
ok, m = best
w0 = [c for c in cards if c["yaw"] is not None and abs(c["yaw"]) < 0.2]
wx = np.mean([(m @ np.array(c["dW"]["x"]))[0] > 0 for c in w0]); wy = np.mean([(m @ np.array(c["dW"]["y"]))[1] < 0 for c in w0])
print(f"agentview convention satisfied in {ok}/{len(cards)} cards; wrist check at yaw~0 (n={len(w0)}): arm+x->objects down {wx:.2f}, arm+y->objects left {wy:.2f}")
ex = cards[len(cards) // 2]; print("example raw", ex)
json.dump({"m": m.tolist(), "units_per_cm": units_per_cm, "agentview_ok": int(ok), "n": len(cards), "wrist_x_ok": float(wx), "wrist_y_ok": float(wy)},
          open(P + "card_transform.json", "w"))
