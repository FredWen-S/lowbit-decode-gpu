"""The load generator against a fake streaming server with known timings (no GPU needed)."""
import asyncio
import json
import os
import sys

import pytest

aiohttp = pytest.importorskip("aiohttp")
from aiohttp import web  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))
import serve_client  # noqa: E402

FIRST, STEP = 0.05, 0.01  # fake prefill and per-token delays (s)


async def fake_completions(request):
    body = await request.json()
    n = body["max_tokens"]
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    await asyncio.sleep(FIRST)
    for i in range(n):
        if i:
            await asyncio.sleep(STEP)
        chunk = {"choices": [{"index": 0, "text": f" t{i}"}]}
        await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
    usage = {"prompt_tokens": len(body["prompt"].split()), "completion_tokens": n}
    await resp.write(f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n".encode())
    await resp.write(b"data: [DONE]\n\n")
    return resp


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


def test_closed_loop_metrics(server_url):
    loop, url = server_url
    reqs = [("a b c d", 20)] * 8
    records, wall = loop.run_until_complete(serve_client.run(url, reqs, concurrency=4))
    s = serve_client.summarize(records, wall)
    assert s["requests"] == 8 and all(r["output_tokens"] == 20 and r["prompt_tokens"] == 4 for r in records)
    assert FIRST * 1e3 <= s["ttft_p50_ms"] < FIRST * 1e3 + 40
    assert STEP * 1e3 <= s["tpot_p50_ms"] < STEP * 1e3 + 6
    # 4 workers x 2 rounds of ~0.24 s each
    assert 0.4 < wall < 1.0


def test_periodic_background_is_tagged_and_stops(server_url):
    loop, url = server_url
    background = [(f"y{i} y", 2) for i in range(50)]
    records, wall = loop.run_until_complete(
        serve_client.run(url, [("x", 30)] * 4, concurrency=2, background=background, interval=0.1))
    kinds = [r["kind"] for r in records]
    assert kinds.count("main") == 4
    assert 3 <= kinds.count("periodic") <= 12  # about wall / interval
    assert serve_client.summarize(records, wall, kind="periodic")["requests"] == kinds.count("periodic")


def test_periodic_stops_when_requests_run_out(server_url):
    loop, url = server_url
    records, _ = loop.run_until_complete(
        serve_client.run(url, [("x", 60)] * 2, concurrency=2, background=[("a", 2), ("b", 2)], interval=0.05))
    assert [r["kind"] for r in records].count("periodic") == 2
