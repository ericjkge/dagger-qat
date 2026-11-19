"""The teacher's chat template, which the students use: its system prompt carries the reasoning effort, the way
PEAR's agent loop renders an episode's opening messages (a system prompt, then the instruction with camera frames)."""

import json

import pytest
from conftest import TEACHER, TEACHER_REVISION
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

pytestmark = pytest.mark.network

OPENING = [
    {"role": "system", "content": "You control a Franka arm. Answer with one JSON action."},
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "frame"},
            {"type": "text", "text": "put the bowl on the plate"},
        ],
    },
]


def render(template: str, **kwargs) -> str:
    # as transformers renders chat templates
    def raise_exception(message):
        raise ValueError(message)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.filters["tojson"] = lambda value, **kw: json.dumps(value, ensure_ascii=False)
    env.globals["raise_exception"] = raise_exception
    return env.from_string(template).render(messages=OPENING, add_generation_prompt=True, **kwargs)


@pytest.fixture(scope="module")
def template(hub_file):
    return hub_file(TEACHER, TEACHER_REVISION, "chat_template.jinja").read_text()


def test_xhigh_effort_by_default(template):
    prompt = render(template, enable_thinking=True)
    assert "Reasoning effort is set to xhigh" in prompt
    assert prompt.index("Reasoning effort") < prompt.index("You control a Franka arm")  # in the system prompt
    assert prompt.count("<|vision_start|>") == 1
    assert prompt.rstrip().endswith("<think>")


def test_low_effort(template):
    prompt = render(template, enable_thinking=True, reasoning_effort="low")
    assert "Reasoning effort is set to low" in prompt


def test_unknown_effort_is_refused(template):
    with pytest.raises(ValueError, match="Unexpected reasoning effort"):
        render(template, enable_thinking=True, reasoning_effort="max")
