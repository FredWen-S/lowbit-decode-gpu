"""Accuracy through an OpenAI-compatible completions server (vLLM): MMLU and WikiText-2.

Every quantization of a model sees the same items, so differences are compared per item
(paired), as in the A53 study.

  MMLU       zero-shot, one completion token with the top-20 logprobs. The prediction is the
             most likely of the letters A-D among those 20; if none appears it counts as wrong.
  WikiText-2 fixed token chunks sent as token ids with echo=True, so the server returns the
             log-probability of every prompt token. Each chunk gives a mean negative log-likelihood.
"""
import asyncio
import math

import aiohttp

LETTERS = "ABCD"


def mmlu_prompt(item):
    subject = item["subject"].replace("_", " ")
    lines = [f"The following is a multiple choice question about {subject}.", "", item["question"].strip()]
    lines += [f"{letter}. {choice}" for letter, choice in zip(LETTERS, item["choices"])]
    lines.append("Answer:")
    return "\n".join(lines)


def pick_letter(top):
    """`top` maps a token to its logprob for the first generated token -> 'A'..'D' or None."""
    best = {}
    for tok, lp in top.items():
        t = tok.strip()
        if t in LETTERS and lp > best.get(t, -math.inf):
            best[t] = lp
    return max(best, key=best.get) if best else None


async def _post(session, url, payload):
    async with session.post(url, json=payload) as r:
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status}: {(await r.text())[:300]}")
        return await r.json()


async def _map(fn, items, concurrency):
    """Run fn over items with at most `concurrency` requests in flight, keeping order."""
    sem = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=3600)
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as session:
        async def one(x):
            async with sem:
                return await fn(session, x)
        return await asyncio.gather(*(one(x) for x in items))


async def mmlu(url, items, concurrency=32):
    async def one(session, item):
        res = await _post(session, url, {"model": "model", "prompt": mmlu_prompt(item), "max_tokens": 1,
                                         "temperature": 0.0, "logprobs": 20})
        top = (res["choices"][0]["logprobs"]["top_logprobs"] or [{}])[0] or {}
        pred = pick_letter(top)
        return {"pred": pred, "answer": LETTERS[item["answer"]], "correct": int(pred == LETTERS[item["answer"]])}
    return await _map(one, items, concurrency)


async def chunk_nll(url, chunks, concurrency=8):
    """Mean negative log-likelihood of each token-id chunk (the first token has no context)."""
    async def one(session, ids):
        res = await _post(session, url, {"model": "model", "prompt": ids, "max_tokens": 1, "temperature": 0.0,
                                         "echo": True, "logprobs": 1})
        # echo returns the prompt tokens first, then the one generated token
        lps = res["choices"][0]["logprobs"]["token_logprobs"][1:len(ids)]
        return -sum(lps) / len(lps)
    return await _map(one, chunks, concurrency)


def paired(a, b):
    """Mean of a - b over shared items and its standard error."""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    mean = sum(d) / n
    se = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1) / n) if n > 1 else float("nan")
    return mean, se
