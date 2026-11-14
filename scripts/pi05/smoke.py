"""Smoke checks before any experiment: guards, policy determinism, differentiable path, hook effects, timing."""

import json
import time

import numpy as np
import torch

from dqat.pi05 import sim
from dqat.pi05.actquant import ActivationController, QuantConfig
from dqat.pi05.policy import Pi05, make_element

CKPT = "/workspace/ckpt/pi05_libero_pytorch"
name = list(sim.TASKS)[0]
suite_name, suite, tid = sim.find_task(name)
task = suite.get_task(tid)
env = sim.make_env(task)
env.reset()
obs = env.set_init_state(sim.init_states(task)[0])
for _ in range(10):
    obs, _, _, _ = env.step(sim.DUMMY_ACTION)
g = sim.Guards(env)
r = g.read()
print("guards", {k: v for k, v in r.items() if k in ("d_grasp", "c_grasp", "grasped", "d_place", "c_place")},
      "invalid", g.invalid)
t = time.time()
for _ in range(20):
    g.read()
print("guard read s", (time.time() - t) / 20)

t = time.time()
pi = Pi05(CKPT)
print("load s", time.time() - t, "out_scale", pi.out_scale[0].round(4), "offset", pi.out_offset[0].round(4))
ctrl = ActivationController(pi.model).install()
el = make_element(obs, task.language)
noise = pi.noise(0)
res = {}
for cfg_name, cfg in (("bf16", None), ("a4", QuantConfig(bits=4)), ("a8", QuantConfig(bits=8))):
    ctrl.quant = cfg
    outs = []
    for _ in range(3):
        ctrl.begin_call()
        torch.cuda.synchronize()
        t = time.time()
        outs.append(pi.infer(el, noise))
        torch.cuda.synchronize()
        dt = time.time() - t
    res[cfg_name] = outs[0]
    print(cfg_name, "infer s", round(dt, 4), "steps", ctrl.step, "repeat max diff",
          float(np.abs(outs[0] - outs[1]).max()), "act rel mse", ctrl.activation_mse()[1],
          "first action", outs[0][0].round(4))
ctrl.quant = None
print("a4 vs bf16 max|da|", np.abs(res["a4"] - res["bf16"]).max(), "a8 vs bf16", np.abs(res["a8"] - res["bf16"]).max())
ctrl.begin_call()
t = time.time()
out = pi.actions_with_grad(el, noise)
print("grad path vs infer max diff", float(np.abs(out.detach().cpu().numpy() - res["bf16"]).max()),
      "s", time.time() - t)
ctrl.probe = "all"
ctrl.begin_call()
out = pi.actions_with_grad(el, noise)
print("probe leaves", len(ctrl.leaves), "steps", ctrl.step)
t = time.time()
gr = torch.autograd.grad(out[:5].sum(), list(ctrl.leaves.values()), retain_graph=True, allow_unused=True)
print("one backward s", time.time() - t, "none grads", sum(x is None for x in gr),
      "max |grad|", max(float(x.abs().max()) for x in gr if x is not None))
print("cuda mem GB", torch.cuda.max_memory_allocated() / 1e9)
print(json.dumps({"ok": True}))
