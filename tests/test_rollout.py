import numpy as np
import pytest

from dqat.rollout import run_episode


class Environment:
    def __init__(self, done_at=None, error_at=None):
        self.done_at, self.error_at = done_at, error_at
        self.actions = []

    def reset(self):
        self.actions.clear()

    def set_init_state(self, state):
        self.initial = state
        return {"t": 0}

    def step(self, action):
        self.actions.append(action)
        if len(self.actions) == self.error_at:
            raise RuntimeError("renderer disconnected")
        return {"t": len(self.actions)}, 0, len(self.actions) == self.done_at, {}


class Policy:
    def predict(self, observation, instruction):
        return np.tile(np.arange(8)[:, None], (1, 7)).astype(float)


def identity(action, family):
    return action


def test_settling_chunking_and_success_step_count():
    env = Environment(done_at=20)
    result = run_episode(env, Policy(), "test", [123], identity)
    assert result == dict(success=True, action_steps=10, policy_calls=2)
    assert env.actions[:10] == [[0, 0, 0, 0, 0, 0, -1]] * 10
    assert [a[0] for a in env.actions[10:]] == [0, 1, 2, 3, 4, 5, 6, 7, 0, 1]
    assert env.initial == [123]


def test_timeout_is_completed_task_failure():
    env = Environment()
    result = run_episode(env, Policy(), "test", [0], identity, max_steps=17)
    assert result == dict(success=False, action_steps=17, policy_calls=3)
    assert len(env.actions) == 27


def test_infrastructure_error_is_not_a_task_failure():
    with pytest.raises(RuntimeError, match="renderer disconnected"):
        run_episode(Environment(error_at=12), Policy(), "test", [0], identity)


def test_invalid_action_aborts_before_execution():
    class BadPolicy:
        def predict(self, obs, instruction):
            return np.full((8, 7), np.nan)

    env = Environment()
    with pytest.raises(ValueError, match="finite"):
        run_episode(env, BadPolicy(), "test", [0], identity)
    assert len(env.actions) == 10
