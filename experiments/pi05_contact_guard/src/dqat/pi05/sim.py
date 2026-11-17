"""LIBERO environments, exact branching, and contact guards from privileged MuJoCo state.

PRIVILEGED INFORMATION: everything in this module (geom distances, contact lists, object poses) is
simulator ground truth used only for *analysis*. A deployed quantizer has none of it.

Branching: a robosuite env's future depends on the MuJoCo state AND on Python-side controller state.
For the Panda gripper the latter is `gripper.current_action` (the gripper command is integrated
with `speed * sign(action)`), so a branch snapshot stores both, plus the env clock. `verify_branch_
determinism` checks replay equality once per task before any result is recorded.

Guards (the manipulated object O and placement target P are LIBERO's `obj_of_interest[0:2]`):
  d_grasp  min signed distance between finger collision geoms and O's collision geoms (m)
  c_grasp  any MuJoCo contact between finger geoms and O
  grasped  robosuite _check_grasp: both finger pads touching O
  d_place  min signed distance between O and P collision geoms
  c_place  any contact between O and P
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import mujoco
import numpy as np

DUMMY_ACTION = [0.0] * 6 + [-1.0]
SETTLE_STEPS = 10  # openpi LIBERO client: wait for objects to settle
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
DIST_MAX = 0.25  # distances beyond this are reported as DIST_MAX

TASKS = {
    # name -> suite; chosen for unambiguous grasp + placement contact events
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate": "libero_spatial",
    "pick_up_the_alphabet_soup_and_place_it_in_the_basket": "libero_object",
    "put_the_cream_cheese_in_the_bowl": "libero_goal",
}


def find_task(name):
    from libero.libero import benchmark

    suite_name = TASKS[name]
    suite = benchmark.get_benchmark_dict()[suite_name]()
    for task_id in range(suite.n_tasks):
        if suite.get_task(task_id).name == name:
            return suite_name, suite, task_id
    raise KeyError(name)


def init_states(task):
    """LIBERO's get_task_init_states, with weights_only=False: torch>=2.6 defaults to weights_only=True,
    which rejects these numpy pickles. Only the pinned LIBERO repo's own init files are loaded here."""
    import os

    import torch
    from libero.libero import get_libero_path

    path = os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file)
    return torch.load(path, weights_only=False)


def make_env(task, resolution=256, seed=7, cameras=True):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    kwargs = dict(bddl_file_name=str(bddl), camera_heights=resolution, camera_widths=resolution)
    if not cameras:
        kwargs.update(use_camera_obs=False, has_offscreen_renderer=False)
    env = OffScreenRenderEnv(**kwargs)
    env.seed(seed)  # openpi: seed affects object positions even with a fixed initial state
    return env


@dataclass
class Snapshot:
    mj_state: np.ndarray
    gripper_action: np.ndarray
    cur_time: float
    timestep: int


def snapshot(env) -> Snapshot:
    e = env.env
    return Snapshot(e.sim.get_state().flatten().copy(), np.array(e.robots[0].gripper.current_action, copy=True),
                    float(e.cur_time), int(e.timestep))


def restore(env, snap: Snapshot):
    e = env.env
    e.sim.set_state_from_flattened(snap.mj_state)
    e.sim.forward()
    e.robots[0].gripper.current_action = np.array(snap.gripper_action, copy=True)
    e.cur_time = snap.cur_time
    e.timestep = snap.timestep


def physics_step(env, action):
    """robosuite MujocoEnv.step without observation rendering or reward/termination bookkeeping."""
    e = env.env
    policy_step = True
    for _ in range(int(e.control_timestep / e.model_timestep)):
        e.sim.forward()
        e._pre_action(np.asarray(action, dtype=np.float64), policy_step)
        e.sim.step()
        policy_step = False
    e.cur_time += e.control_timestep
    e.timestep += 1


class Guards:
    """Geom bookkeeping for one env (geom ids are identical across envs built from the same BDDL)."""

    def __init__(self, env):
        self.env = env
        self._sim = None
        self._fromto = np.zeros(6)
        self.invalid = 0  # count of dropped spurious negative hull distances (reported)
        self.has_geom_distance = hasattr(mujoco, "mj_geomDistance")
        self._bind()
        self.names = (self._o_name, self._p_name)

    def _bind(self):
        """(Re)resolve model/data and geom ids. LIBERO's env.reset() rebuilds the MjSim, so references
        must never be cached across resets; `read` calls this whenever the sim object changes."""
        e = self.env.env
        self._sim = e.sim
        self.model = e.sim.model._model
        self.data = e.sim.data._data
        self._o_name, self._p_name = e.obj_of_interest[0], e.obj_of_interest[1]
        gripper = e.robots[0].gripper
        finger_names = gripper.important_geoms["left_finger"] + gripper.important_geoms["right_finger"]
        self.fingers = self._ids(finger_names)
        self.obj_geom_names = self._object_geoms(e, self._o_name)
        self.obj = self._ids(self.obj_geom_names)
        self.place = self._ids(self._object_geoms(e, self._p_name))
        if not (len(self.fingers) and len(self.obj) and len(self.place)):
            raise RuntimeError(f"empty geom set: fingers={len(self.fingers)} obj={len(self.obj)} place={len(self.place)}")

    def _ids(self, names):
        out = []
        for n in names:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n)
            if gid >= 0:
                out.append(gid)
        return np.array(sorted(set(out)), dtype=int)

    @staticmethod
    def _object_geoms(e, name):
        if name in e.objects_dict:
            obj = e.objects_dict[name]
        elif name in e.fixtures_dict:
            obj = e.fixtures_dict[name]
        else:
            raise KeyError(f"{name} is neither an object nor a fixture")
        return list(obj.contact_geoms)

    def _min_dist(self, set_a, set_b):
        """Min signed distance between two geom sets.

        Pairs in MuJoCo's active contact list use the contact's own penetration depth (contact.dist).
        For other pairs, mj_geomDistance on mesh-mesh hulls occasionally returns a spurious large
        negative value (observed: -0.0996 m for a bowl/plate pair 2.5 mm apart with no contact); a
        negative result without a confirming contact is dropped and counted in `self.invalid`.
        Pairs whose bounding spheres are farther apart than the current best are skipped.
        """
        m, d = self.model, self.data
        sa, sb = set(set_a.tolist()), set(set_b.tolist())
        contact_dist = {}
        for i in range(d.ncon):
            c = d.contact[i]
            if (c.geom1 in sa and c.geom2 in sb) or (c.geom1 in sb and c.geom2 in sa):
                key = (min(c.geom1, c.geom2), max(c.geom1, c.geom2))
                contact_dist[key] = min(contact_dist.get(key, np.inf), float(c.dist))
        best = min(contact_dist.values(), default=DIST_MAX)
        pos, rb = d.geom_xpos, m.geom_rbound
        for a in set_a:
            for b in set_b:
                if (min(a, b), max(a, b)) in contact_dist:
                    continue
                lower = np.linalg.norm(pos[a] - pos[b]) - rb[a] - rb[b]
                if lower >= best:
                    continue
                dist = mujoco.mj_geomDistance(m, d, int(a), int(b), DIST_MAX, self._fromto)
                if dist < 0:
                    self.invalid += 1
                    continue
                best = min(best, dist)
        return min(best, DIST_MAX)

    def _contact(self, set_a, set_b):
        d = self.data
        sa, sb = set(set_a.tolist()), set(set_b.tolist())
        for i in range(d.ncon):
            c = d.contact[i]
            if (c.geom1 in sa and c.geom2 in sb) or (c.geom1 in sb and c.geom2 in sa):
                return True
        return False

    def read(self):
        e = self.env.env
        if e.sim is not self._sim:
            self._bind()
        return dict(
            d_grasp=self._min_dist(self.fingers, self.obj),
            c_grasp=self._contact(self.fingers, self.obj),
            grasped=bool(e._check_grasp(gripper=e.robots[0].gripper, object_geoms=self.obj_geom_names)),
            d_place=self._min_dist(self.obj, self.place),
            c_place=self._contact(self.obj, self.place),
            eef=np.array(e.sim.data.site_xpos[e.robots[0].eef_site_id], copy=True),
            gripper_qpos=np.array(e.sim.data.qpos[e.robots[0]._ref_gripper_joint_pos_indexes], copy=True),
        )


def branch(env, guards, snap, actions):
    """Restore `snap`, execute `actions` with physics_step, return per-step guard readings (len+1)."""
    restore(env, snap)
    trace = [guards.read()]
    for a in actions:
        physics_step(env, a)
        trace.append(guards.read())
    return trace


def verify_branch_determinism(env, guards, snap, actions, atol=1e-9):
    """Same snapshot + same actions must give the same trajectory, twice in a row."""
    t1 = branch(env, guards, snap, actions)
    q1 = env.env.sim.get_state().flatten().copy()
    t2 = branch(env, guards, snap, actions)
    q2 = env.env.sim.get_state().flatten().copy()
    if not np.allclose(q1, q2, atol=atol, rtol=0):
        raise RuntimeError(f"branch nondeterminism: max |dq| = {np.abs(q1 - q2).max():.3g}")
    for r1, r2 in zip(t1, t2):
        if abs(r1["d_grasp"] - r2["d_grasp"]) > 1e-9 or r1["c_grasp"] != r2["c_grasp"]:
            raise RuntimeError("guard trace nondeterminism")
    return True
