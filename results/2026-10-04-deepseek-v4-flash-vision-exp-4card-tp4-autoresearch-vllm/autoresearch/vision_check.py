"""Gradient fixture check (same question as the 2026-09-02 notebook). usage: vision_check.py OUT.json"""
import json, sys, time, urllib.request
sys.path.insert(0, "."); from bench_harness import IMAGE_DATA_URL
Q = "Name the two colors this gradient blends between, left color first."
body = {"model": "dsv4v", "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": IMAGE_DATA_URL}}, {"type": "text", "text": Q}]}], "max_tokens": 64, "temperature": 0}
t = time.perf_counter()
r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:18098/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
c = r["choices"][0]["message"]["content"] or ""
ok = "red" in c.lower() and "green" in c.lower()
out = {"question": Q, "answer": c, "finish_reason": r["choices"][0]["finish_reason"], "usage": r["usage"], "latency_s": round(time.perf_counter() - t, 3), "keyword_match": ok}
print(json.dumps(out)); json.dump(out, open(sys.argv[1], "w"), indent=1)
