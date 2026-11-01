"""Stand-in for `python -m vllm.entrypoints.openai.api_server`, used to dry-run serve_sweep.py."""
import argparse
import asyncio
import json

from aiohttp import web


async def models(request):
    return web.json_response({"data": [{"id": "model"}]})


async def completions(request):
    body = await request.json()
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    await asyncio.sleep(0.002 * len(body["prompt"].split()) / 100 + 0.01)
    for i in range(body["max_tokens"]):
        await asyncio.sleep(0.001)
        await resp.write(f"data: {json.dumps({'choices': [{'text': ' x'}]})}\n\n".encode())
    usage = {"prompt_tokens": len(body["prompt"].split()), "completion_tokens": body["max_tokens"]}
    await resp.write(f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n".encode())
    await resp.write(b"data: [DONE]\n\n")
    return resp


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int)
    a, _ = p.parse_known_args()
    app = web.Application()
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/completions", completions)
    web.run_app(app, host="127.0.0.1", port=a.port, print=None)
