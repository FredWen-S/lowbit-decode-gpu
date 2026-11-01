"""Engine-neutral load generator for OpenAI-compatible completion servers (vLLM, SGLang).

Every request streams, ignores EOS and asks for usage, so each engine produces exactly the
requested number of tokens and reports its own token counts. Per request it records:
  ttft   time from sending the request to the first streamed text
  tpot   (e2e - ttft) / (output tokens - 1): mean time per output token after the first
  itl    gaps between streamed chunks (a chunk can hold more than one token)
"""
import asyncio
import json
import time

import aiohttp


async def stream_request(session, url, prompt, max_tokens, model="model"):
    payload = {
        "model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0,
        "stream": True, "stream_options": {"include_usage": True}, "ignore_eos": True,
    }
    t0 = time.perf_counter()
    times, usage = [], None
    async with session.post(url, json=payload) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {(await resp.text())[:300]}")
        async for raw in resp.content:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if choices and choices[0].get("text"):
                times.append(time.perf_counter())
    t1 = time.perf_counter()
    if not times:
        raise RuntimeError("no tokens streamed")
    out_tokens = usage["completion_tokens"] if usage else len(times)
    ttft = times[0] - t0
    return {
        "start": t0, "end": t1, "ttft": ttft, "e2e": t1 - t0,
        "tpot": (t1 - t0 - ttft) / (out_tokens - 1) if out_tokens > 1 else None,
        "itl": [b - a for a, b in zip(times, times[1:])],
        "prompt_tokens": usage["prompt_tokens"] if usage else None,
        "output_tokens": out_tokens,
    }


async def closed_loop(session, url, requests, concurrency, kind="main"):
    """`concurrency` workers each send their next request as soon as the previous one ends."""
    queue = list(requests)
    records = []

    async def worker():
        while queue:
            prompt, max_tokens = queue.pop(0)
            r = await stream_request(session, url, prompt, max_tokens)
            r["kind"] = kind
            records.append(r)

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return records


async def periodic(session, url, requests, interval, stop, kind="periodic"):
    """Send the next of `requests` every `interval` seconds (open loop) until `stop` is set or
    the list runs out. Each arrival should be a different prompt: repeating one prompt would
    let the prefix cache serve every arrival after the first."""
    records, tasks = [], []
    pending = list(requests)

    async def one(request):
        r = await stream_request(session, url, *request)
        r["kind"] = kind
        records.append(r)

    while not stop.is_set() and pending:
        tasks.append(asyncio.create_task(one(pending.pop(0))))
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass
    await asyncio.gather(*tasks)
    return records


async def run(url, requests, concurrency, background=None, interval=None):
    """Run a closed-loop workload, optionally with periodic background requests alongside."""
    timeout = aiohttp.ClientTimeout(total=3600)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        t0 = time.perf_counter()
        if background is None:
            records = await closed_loop(session, url, requests, concurrency)
        else:
            stop = asyncio.Event()

            async def main_then_stop():
                try:
                    return await closed_loop(session, url, requests, concurrency)
                finally:
                    stop.set()

            main, bg = await asyncio.gather(main_then_stop(),
                                            periodic(session, url, background, interval, stop))
            records = main + bg
        wall = time.perf_counter() - t0
    return records, wall


def pct(values, q):
    v = sorted(x for x in values if x is not None)
    if not v:
        return None
    i = min(len(v) - 1, max(0, round(q / 100 * (len(v) - 1))))
    return v[i]


def summarize(records, wall, kind="main"):
    rs = [r for r in records if r["kind"] == kind]
    out = {"requests": len(rs), "wall_s": wall,
           "output_tok_per_s": sum(r["output_tokens"] for r in rs) / wall,
           "mean_prompt_tokens": sum(r["prompt_tokens"] or 0 for r in rs) / max(1, len(rs))}
    for name in ("ttft", "tpot", "e2e"):
        for q in (50, 90, 99):
            out[f"{name}_p{q}_ms"] = 1e3 * pct([r[name] for r in rs], q)
    out["itl_p99_ms"] = 1e3 * pct([g for r in rs for g in r["itl"]], 99)
    return out
