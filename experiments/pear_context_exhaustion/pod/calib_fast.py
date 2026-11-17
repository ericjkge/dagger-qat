"""Oracle calibration cards without replay: agentview is fixed; the wrist camera is rigid on the end effector.
Per episode: one reset (camera intrinsics/extrinsics, EE->camera offset, table-object height) and one recorded action
(state units per metre). Per round: cards from the recorded state. Wrist orientation follows the recorded yaw change
about world z (rounds whose roll/pitch moved > 0.3 rad from reset are flagged approximate)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/workspace")
from pear.envs.libero.env import LiberoEnv

RES = Path("/workspace/results/dflash2_main"); OUT = Path("/workspace/results/pilot/cards.jsonl"); H = 512


def snap(sim):
    d = {}
    for cam in ("agentview", "robot0_eye_in_hand"):
        i = sim.model.camera_name2id(cam)
        d[cam] = (sim.data.cam_xpos[i].copy(), sim.data.cam_xmat[i].reshape(3, 3).copy(), float(sim.model.cam_fovy[i]))
    for n in ("gripper0_grip_site", "robot0_grip_site"):
        try:
            d["eef"] = sim.data.get_site_xpos(n).copy(); break
        except Exception:
            pass
    objz = [sim.data.body_xpos[i][2] for i in range(sim.model.nbody)
            if any(k in (sim.model.body_id2name(i) or "") for k in ("_1", "_2", "object")) and sim.data.body_xpos[i][2] < d["eef"][2] - 0.03]
    d["zt"] = float(np.median(objz)) if objz else float(d["eef"][2] - 0.2)
    return d


def pix(c, R, fovy, P):  # image with row down / col right of a correctly oriented render (MuJoCo cam: -z forward, +y up)
    q = R.T @ (P - c); f = 0.5 * H / np.tan(np.deg2rad(fovy) / 2)
    return np.array([H / 2 - f * q[1] / -q[2], H / 2 + f * q[0] / -q[2]])


def Rz(a):
    c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def main():
    global OUT
    files = sorted(RES.glob("libero_*.jsonl"))
    if len(sys.argv) > 1:
        files = [RES / f"{sys.argv[1]}.jsonl"]; OUT = OUT.with_name(f"cardsf_{sys.argv[1]}.jsonl")
    OUT.unlink(missing_ok=True)
    for p in files:
        ep = p.stem; rows = [json.loads(l) for l in p.open()]
        suite, rest = ep.rsplit("_t", 1); task, trial = rest.split("_r")
        env = LiberoEnv(suite, int(task), int(trial)); box = []
        orig = env._close_scene
        def hooked(env=env, orig=orig):
            if env.scene is not None:
                box.append(snap(env.scene.sim))
            orig()
        env._close_scene = hooked
        r0 = env.rollout([], None); s0 = r0["state"]["arm"]
        hist = []; r1 = env.rollout(hist, rows[0]["action"]) if rows[0].get("action") else None
        a, b = box[0], (box[1] if len(box) > 1 else None)
        if b is not None:
            ds = np.array([r1["state"]["arm"][k] - s0[k] for k in "xyz"]); dp = b["eef"] - a["eef"]
            upm = [ds[i] / dp[i] for i in range(3) if abs(dp[i]) > 0.005]
        else:
            upm = []
        upm = float(np.median(upm)) if upm else 100.0
        origin = a["eef"] - np.array([s0[k] for k in "xyz"]) / upm
        cA, RA, fA = a["agentview"]; cW0, RW0, fW = a["robot0_eye_in_hand"]; off = cW0 - a["eef"]
        for r, row in enumerate(rows):
            st = row["state"]["arm"]; e = origin + np.array([st[k] for k in "xyz"]) / upm
            dyaw = st["yaw"] - s0["yaw"]; Rd = Rz(dyaw)
            cW, RW = e + Rd @ off, Rd @ RW0
            u = 1.0 / upm  # one state unit in metres
            dA = {ax: (pix(cA, RA, fA, e + u * np.eye(3)[i]) - pix(cA, RA, fA, e)).tolist() for i, ax in enumerate("xyz")}
            Q = np.array([cW[0], cW[1], a["zt"]]) + RW @ np.array([0, 0, 0.0])
            dW = {ax: (pix(cW + u * np.eye(3)[i], RW, fW, Q) - pix(cW, RW, fW, Q)).tolist() for i, ax in enumerate("xy")}
            approx = abs(st["roll"] - s0["roll"]) > 0.3 or abs(st["pitch"] - s0["pitch"]) > 0.3
            card = {"ep": ep, "round": r, "dA": dA, "dW": dW, "wrist_depth_cm": round(100 * (cW[2] - a["zt"]), 1),
                    "yaw": st["yaw"], "approx": approx, "units_per_m": round(upm, 2)}
            with OUT.open("a") as f:
                f.write(json.dumps(card) + "\n")
        print(ep, "units/m", round(upm, 2), "fovy", fA, fW, "table z", round(a["zt"], 3), flush=True)
    print("CALIB_DONE")


if __name__ == "__main__":
    main()
