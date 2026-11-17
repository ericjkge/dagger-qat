"""Oracle calibration cards: replay each recorded episode in LIBERO, and at every round measure how a +1 state-unit
arm translation moves (a) the gripper in the agentview image and (b) table-height objects in the wrist image, in the
180-degree-rotated frames PEAR sees. Validates replay against the recorded states and the image-axis transform against
the environment prompt's stated conventions. Run with the sim venv python (needs pear, robosuite)."""
import json, sys, itertools
from pathlib import Path
import numpy as np
sys.path.insert(0, "/workspace")
from pear.envs.libero.env import LiberoEnv
from robosuite.utils.camera_utils import get_camera_transform_matrix, project_points_from_world_to_camera

RES = Path("/workspace/results/dflash2_main"); OUT = Path("/workspace/results/pilot/cards.jsonl")
S = LiberoEnv.FRAME_SIZE if hasattr(LiberoEnv, "FRAME_SIZE") else 512
CAMS = {"agentview": "agentview", "wrist": "robot0_eye_in_hand"}


def proj(sim, cam, P):
    M = get_camera_transform_matrix(sim, cam, S, S)
    return project_points_from_world_to_camera(np.atleast_2d(P), M, S, S).astype(float)  # (row, col), robosuite convention


def eef(sim):
    for n in ("gripper0_grip_site", "robot0_grip_site", "gripper0_right_grip_site"):
        try:
            return sim.data.get_site_xpos(n).copy()
        except Exception:
            pass
    raise KeyError("grip site")


def subpix(sim, cam, P, d):  # sub-pixel finite difference via 10x step (projection rounds to ints)
    return (proj(sim, cam, P + 10 * d)[0] - proj(sim, cam, P)[0]) / 10.0


def measure(sim):
    p = eef(sim)
    dA = {a: subpix(sim, "agentview", p, 0.01 * np.eye(3)[i]) for i, a in enumerate("xyz")}
    cam_id = sim.model.camera_name2id(CAMS["wrist"]); cpos = sim.data.cam_xpos[cam_id].copy()
    objz = [sim.data.body_xpos[i][2] for i in range(sim.model.nbody)
            if any(k in (sim.model.body_id2name(i) or "") for k in ("_1", "_2", "object")) and sim.data.body_xpos[i][2] < cpos[2] - 0.05]
    zt = float(np.median(objz)) if objz else p[2] - 0.05
    Q = np.array([cpos[0], cpos[1], zt])
    dW = {a: -subpix(sim, CAMS["wrist"], Q, 0.01 * np.eye(3)[i]) for i, a in enumerate("xy")}  # objects move opposite
    return p, dA, dW, round(100 * (cpos[2] - zt), 1)


def main():
    global OUT
    eps = sorted(p.stem for p in RES.glob("libero_*.jsonl"))
    if len(sys.argv) > 1:
        eps = [sys.argv[1]]; OUT = OUT.with_name(f"cards_{sys.argv[1]}.jsonl")
    OUT.unlink(missing_ok=True)
    for ep in eps:
        rows = [json.loads(l) for l in (RES / f"{ep}.jsonl").open()]
        suite, rest = ep.rsplit("_t", 1); task, trial = rest.split("_r")
        env = LiberoEnv(suite, int(task), int(trial))
        box = {}
        orig = env._close_scene
        def hooked(env=env, orig=orig, box=box):
            if env.scene is not None:
                box["m"] = measure(env.scene.sim)
            orig()
        env._close_scene = hooked
        hist = []
        res = env.rollout(hist, None)
        pts, sts = [], []
        for r, row in enumerate(rows):
            st = res["state"]["arm"]; rec_st = row["state"]["arm"]
            dev = max(abs(st[k] - rec_st[k]) for k in ("x", "y", "z"))
            p, dA, dW, depth = box["m"]
            pts.append(p); sts.append([st["x"], st["y"], st["z"]])
            card = {"ep": ep, "round": r, "replay_dev": round(dev, 3), "dA": {k: v.tolist() for k, v in dA.items()},
                    "dW": {k: v.tolist() for k, v in dW.items()}, "wrist_depth_cm": depth, "yaw": st.get("yaw")}
            with OUT.open("a") as f:
                f.write(json.dumps(card) + "\n")
            if row.get("action") is None or row.get("outcome") is None or r == len(rows) - 1:
                break
            res = env.rollout(hist, row["action"]); hist.extend(res["actions"])
            if res.get("images") is None:
                break
        P_, Sx = np.array(pts), np.array(sts)
        if len(P_) > 3:
            sc = [np.polyfit(P_[:, i], Sx[:, i], 1)[0] for i in range(3)]
            print(ep, "state-units per metre (x,y,z):", np.round(sc, 1), "max replay dev:",
                  max(json.loads(l)["replay_dev"] for l in OUT.open() if json.loads(l)["ep"] == ep), flush=True)
    print("CALIB_DONE")


if __name__ == "__main__":
    main()
