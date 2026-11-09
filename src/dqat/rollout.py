"""Episode control flow, with infrastructure errors propagated to the caller."""

from collections import deque

import numpy as np


def run_episode(
    env,
    policy,
    instruction,
    initial_state,
    process_action,
    max_steps=520,
    settle_steps=10,
    open_loop_steps=8,
    frame_callback=None,
):
    if max_steps < 1 or settle_steps < 0 or not 1 <= open_loop_steps <= 8:
        raise ValueError("Invalid rollout horizon or action chunk length")
    env.reset()
    observation = env.set_init_state(initial_state)
    for _ in range(settle_steps):
        observation, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    queue, calls = deque(), 0
    for step in range(1, max_steps + 1):
        if frame_callback:
            frame_callback(observation)
        if not queue:
            actions = np.asarray(policy.predict(observation, instruction))
            if actions.shape != (8, 7) or not np.isfinite(actions).all():
                raise ValueError("Policy must return a finite (8, 7) action chunk")
            queue.extend(actions[:open_loop_steps])
            calls += 1
        action = process_action(np.array(queue.popleft(), copy=True), "openvla")
        observation, _, done, _ = env.step(action.tolist())
        if done:
            return dict(success=True, action_steps=step, policy_calls=calls)
    return dict(success=False, action_steps=max_steps, policy_calls=calls)
