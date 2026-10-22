"""
Measure first-token and total-response latency of /chat/stream.
Usage:
python -m eval.bench_stream_latency --n 10 --delay 5 --url http://localhost:9000/chat/stream
"""
import argparse
import asyncio
import statistics
import time
import httpx

N = 10

async def one_request(client: httpx.AsyncClient, url: str, payload: dict) -> dict: 
    t0 = time.perf_counter()
    first_token_at = None 
    total_bytes = 0
    async with client.stream("POST", url, json=payload, timeout=60.0) as r: 
        async for chunk in r.aiter_bytes():
            if not chunk:
                continue
            if first_token_at is None:
                first_token_at = time.perf_counter()
            total_bytes += len(chunk)
    t1 = time.perf_counter() 
    return {
        "first_token_ms": round((first_token_at - t0) * 1000) if first_token_at else None,
        "total_ms": round((t1 - t0) * 1000),
        "bytes": total_bytes,
    }

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/chat/stream")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--delay", type=float, default=2.0)
    args = ap.parse_args()
    payload = {
        "user_question": "What is my late fee?",
        "session_id": "bench",
        "product_line": "mobile_postpaid",
    }
    async with httpx.AsyncClient() as client:
        await one_request(client, args.url, payload)
        results = []
        for i in range(args.n):
            results.append(await one_request(client, args.url, payload))
            if i < args.n - 1:
                await asyncio.sleep(args.delay)

        first = [r["first_token_ms"] for r in results if r["first_token_ms"] is not None]
        total = [r["total_ms"] for r in results]
        print(f"n = {args.n}")
        print(f"first_token: p50={statistics.median(first):.0f} ms "
              f"p95={statistics.quantiles(first, n=N)[-1]:.0f} ms")
        print(f"total: p50={statistics.median(total):.0f} ms "
              f"p95={statistics.quantiles(total, n=N)[-1]:.0f} ms")

if __name__ == "__main__":
    asyncio.run(main())
