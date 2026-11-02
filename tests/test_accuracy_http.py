"""The HTTP accuracy eval against a fake completions server with known logprobs (no GPU needed)."""
import asyncio
import math
import os
import sys

import pytest

aiohttp = pytest.importorskip("aiohttp")
from aiohttp import web  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))
import accuracy_http as acc  # noqa: E402


async def fake_completions(request):
    body = await request.json()
    if body.get("echo"):
        ids = body["prompt"]
        lps = [None] + [-1.0] * (len(ids) - 1) + [-0.3]  # prompt tokens, then one generated token
        return web.json_response({"choices": [{"logprobs": {"token_logprobs": lps}}]})
    # always prefer " B", with a weaker "A" variant without the space
    top = {" B": -0.1, "A": -2.0, " the": -3.0}
    return web.json_response({"choices": [{"logprobs": {"top_logprobs": [top]}}]})


@pytest.fixture
def server_url():
    loop = asyncio.new_event_loop()
    app = web.Application()
    app.router.add_post("/v1/completions", fake_completions)
    runner = web.AppRunner(app)
    loop.run_until_complete(runner.setup())
    site = web.TCPSite(runner, "127.0.0.1", 0)
    loop.run_until_complete(site.start())
    port = site._server.sockets[0].getsockname()[1]
    yield loop, f"http://127.0.0.1:{port}/v1/completions"
    loop.run_until_complete(runner.cleanup())
    loop.close()


def test_pick_letter():
    assert acc.pick_letter({" C": -0.5, "C": -0.1, " A": -1.0}) == "C"
    assert acc.pick_letter({" the": -0.1}) is None


def test_mmlu_prompt_has_choices_in_order():
    item = {"subject": "high_school_physics", "question": "Q? ", "choices": ["w", "x", "y", "z"], "answer": 2}
    p = acc.mmlu_prompt(item)
    assert "about high school physics" in p and p.endswith("Answer:")
    assert p.index("A. w") < p.index("B. x") < p.index("C. y") < p.index("D. z")


def test_mmlu_scoring(server_url):
    loop, url = server_url
    items = [{"subject": "s", "question": "q", "choices": list("wxyz"), "answer": a} for a in (1, 0, 1, 3)]
    res = loop.run_until_complete(acc.mmlu(url, items, concurrency=2))
    assert [r["pred"] for r in res] == ["B"] * 4
    assert [r["correct"] for r in res] == [1, 0, 1, 0]


def test_chunk_nll_excludes_first_and_generated_token(server_url):
    loop, url = server_url
    res = loop.run_until_complete(acc.chunk_nll(url, [list(range(10)), list(range(5))]))
    assert res == [1.0, 1.0]


def test_paired():
    mean, se = acc.paired([1, 0, 1, 1], [1, 1, 1, 1])
    assert mean == -0.25 and math.isclose(se, 0.25)
