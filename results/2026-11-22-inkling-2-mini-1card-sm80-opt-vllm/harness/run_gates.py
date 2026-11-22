#!/usr/bin/env python3
"""Runs the 5 capability gates + 2 x ~167.8K-token concurrent long-context soak against an OpenAI-compatible server."""
import argparse
import concurrent.futures as cf
import json
import random
import time
import urllib.request
from pathlib import Path

WORDS = [
    "amber", "anchor", "arch", "atlas", "axis", "beacon", "birch", "blade",
    "bloom", "bridge", "bronze", "canyon", "cedar", "chalk", "cipher", "cliff",
    "cloud", "cobalt", "comet", "coral", "crest", "crown", "crystal", "current",
    "dawn", "delta", "drift", "dune", "echo", "ember", "falcon", "fern",
    "field", "flint", "forge", "frost", "gale", "glacier", "granite", "grove",
    "harbor", "haven", "horizon", "ivory", "jade", "lantern", "latch", "linen",
    "marble", "meadow", "mesa", "mirror", "mist", "moss", "nebula", "oak",
    "onyx", "orbit", "osprey", "pearl", "pebble", "pine", "plume", "prism",
]


def post_json(url: str, path: str, body: dict, timeout: int = 1800) -> dict:
    req = urllib.request.Request(
        url.rstrip("/") + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def get_json(url: str, path: str, timeout: int = 30) -> dict:
    with urllib.request.urlopen(url.rstrip("/") + path, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def build_long_prompt(seed: int, target_words: int = 105000) -> tuple[str, str]:
    rng = random.Random(seed)
    code = f"KITE-{seed}-{rng.randint(100000, 999999)}"
    chunks = [" ".join(rng.choices(WORDS, k=64)) for _ in range(target_words // 64)]
    mid = len(chunks) // 2
    chunks.insert(mid, f"\nCRITICAL RECORD: The vault verification token is {code}.\n")
    filler = "\n".join(chunks)
    prompt = (
        "Read the archive below and reply with the exact vault verification token on the first line, "
        f"followed by a two-sentence explanation of why lighthouses use Fresnel lenses.\n\n{filler}\n\n"
        "What is the exact vault verification token?"
    )
    return prompt, code


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18000")
    ap.add_argument("--model", default="inkling")
    ap.add_argument("--out", default="gates-summary.json")
    a = ap.parse_args()

    results: dict = {"gates": {}}

    # Gate 1: /v1/models
    m = get_json(a.url, "/v1/models")
    model_ids = [x["id"] for x in m.get("data", [])]
    assert a.model in model_ids, f"{a.model} not in {model_ids}"
    results["gates"]["1_load_models"] = {"ok": True, "models": model_ids}
    print("[PASS] Gate 1 (/v1/models):", model_ids, flush=True)

    # Gate 2: Short chat completion (effort=none)
    r2 = post_json(
        a.url,
        "/v1/chat/completions",
        {
            "model": a.model,
            "messages": [{"role": "user", "content": "Say hello in five words."}],
            "max_tokens": 512,
            "temperature": 0.0,
            "reasoning_effort": "none",
            "chat_template_kwargs": {"reasoning_effort": "none"},
        },
    )
    c2 = r2["choices"][0]
    msg2 = c2["message"]
    assert c2["finish_reason"] == "stop" and len((msg2.get("content") or "").strip()) > 0, c2
    results["gates"]["2_chat_short"] = {
        "ok": True,
        "finish_reason": c2["finish_reason"],
        "content": msg2.get("content"),
    }
    print("[PASS] Gate 2 (chat short):", repr(msg2.get("content")), flush=True)

    # Gate 3: Reasoning separation + math accuracy (effort=medium)
    r3 = post_json(
        a.url,
        "/v1/chat/completions",
        {
            "model": a.model,
            "messages": [{"role": "user", "content": "Compute 37 * 43 + 19. State the final integer clearly."}],
            "max_tokens": 2048,
            "temperature": 0.0,
            "reasoning_effort": "medium",
            "chat_template_kwargs": {"reasoning_effort": "medium"},
        },
    )
    c3 = r3["choices"][0]
    msg3 = c3["message"]
    reasoning3 = msg3.get("reasoning") or msg3.get("reasoning_content") or ""
    content3 = msg3.get("content") or ""
    assert c3["finish_reason"] == "stop" and len(reasoning3.strip()) > 10 and "1610" in content3.replace(",", ""), c3
    results["gates"]["3_reasoning"] = {
        "ok": True,
        "finish_reason": c3["finish_reason"],
        "reasoning_chars": len(reasoning3),
        "content": content3[:300],
    }
    print(f"[PASS] Gate 3 (reasoning): reasoning_chars={len(reasoning3)} content={content3[:120]!r}", flush=True)

    # Gate 4: Tool calling round-trip
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_rack_telemetry",
                "description": "Fetch live temperature and power telemetry for a datacenter rack.",
                "parameters": {
                    "type": "object",
                    "properties": {"rack_id": {"type": "string", "description": "Rack identifier, e.g. R-17"}},
                    "required": ["rack_id"],
                },
            },
        }
    ]
    msgs4 = [{"role": "user", "content": "Use get_rack_telemetry to check rack R-17 and tell me its exact temperature in Celsius."}]
    r4a = post_json(
        a.url,
        "/v1/chat/completions",
        {
            "model": a.model,
            "messages": msgs4,
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": 1024,
            "temperature": 0.0,
            "reasoning_effort": "low",
            "chat_template_kwargs": {"reasoning_effort": "low"},
        },
    )
    c4a = r4a["choices"][0]
    tcalls = c4a["message"].get("tool_calls") or []
    assert len(tcalls) >= 1, f"No tool_calls emitted: {c4a}"
    tc = tcalls[0]
    args4 = json.loads(tc["function"]["arguments"])
    assert tc["function"]["name"] == "get_rack_telemetry" and args4.get("rack_id") == "R-17", tc
    msgs4.append(c4a["message"])
    msgs4.append(
        {
            "role": "tool",
            "tool_call_id": tc["id"],
            "name": "get_rack_telemetry",
            "content": json.dumps({"rack_id": "R-17", "temp_c": 23.4, "power_kw": 8.7}),
        }
    )
    r4b = post_json(
        a.url,
        "/v1/chat/completions",
        {
            "model": a.model,
            "messages": msgs4,
            "tools": tools,
            "max_tokens": 1024,
            "temperature": 0.0,
            "reasoning_effort": "low",
            "chat_template_kwargs": {"reasoning_effort": "low"},
        },
    )
    c4b = r4b["choices"][0]
    content4b = c4b["message"].get("content") or ""
    assert c4b["finish_reason"] == "stop" and "23.4" in content4b, c4b
    results["gates"]["4_tools_roundtrip"] = {"ok": True, "tool_call": tc["function"], "final_content": content4b[:300]}
    print("[PASS] Gate 4 (tools round-trip):", repr(content4b[:160]), flush=True)

    # Gate 5: Concurrent 2 x ~167.8K-token long-context soak (335K+ active KV tokens simultaneously)
    def run_long_stream(idx: int) -> dict:
        prompt, code = build_long_prompt(900 + idx)
        t0 = time.time()
        resp = post_json(
            a.url,
            "/v1/chat/completions",
            {
                "model": a.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1536,
                "temperature": 0.0,
                "reasoning_effort": "none",
                "chat_template_kwargs": {"reasoning_effort": "none"},
            },
            timeout=1800,
        )
        dt = time.time() - t0
        ch = resp["choices"][0]
        txt = ch["message"].get("content") or ""
        return {
            "stream": idx,
            "prompt_tokens": resp["usage"]["prompt_tokens"],
            "completion_tokens": resp["usage"]["completion_tokens"],
            "finish_reason": ch["finish_reason"],
            "needle_expected": code,
            "needle_hit": code in txt,
            "seconds": round(dt, 2),
            "content_head": txt[:200],
        }

    print("Running 2 x ~167.8K-token concurrent long-context soak...", flush=True)
    t_soak0 = time.time()
    with cf.ThreadPoolExecutor(2) as ex:
        soak_res = list(ex.map(run_long_stream, [0, 1]))
    soak_dt = round(time.time() - t_soak0, 2)
    for sr in soak_res:
        assert sr["finish_reason"] == "stop" and sr["needle_hit"], sr
    results["gates"]["5_concurrent_long_soak"] = {
        "ok": True,
        "wall_seconds": soak_dt,
        "total_prompt_tokens": sum(x["prompt_tokens"] for x in soak_res),
        "streams": soak_res,
    }
    print("[PASS] Gate 5 (2x long-context concurrent soak):", soak_res, flush=True)

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print("ALL_GATES_PASSED ->", out_path, flush=True)


if __name__ == "__main__":
    main()
