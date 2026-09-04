"""Shared HTTP/tokenizer helpers for the GLM-5.3-Flash rental bench scripts."""
import argparse
import json
import os
import time
import urllib.error
import urllib.request


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def base_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--base-url", default=os.environ.get("GLM_URL", "http://127.0.0.1:18098"))
    p.add_argument("--model", default=os.environ.get("GLM_MODEL_NAME", "glm-5.3-flash"))
    p.add_argument("--out", default=os.environ.get("GLM_OUT", "."))
    return p


class Client:
    def __init__(self, base_url, model):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def post(self, path, body, timeout=1800):
        req = urllib.request.Request(
            self.base_url + path, json.dumps(body).encode(), {"Content-Type": "application/json"}
        )
        return urllib.request.urlopen(req, timeout=timeout)

    def completion(self, prompt, max_tokens, temperature=0, ignore_eos=False, timeout=1800, seed=None):
        body = {"model": self.model, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}
        if ignore_eos:
            body["ignore_eos"] = True
        if seed is not None:
            body["seed"] = seed
        t0 = time.perf_counter()
        try:
            r = json.load(self.post("/v1/completions", body, timeout))
            dt = time.perf_counter() - t0
            return {
                "ok": True,
                "wall_s": round(dt, 4),
                "usage": r["usage"],
                "text_head": r["choices"][0]["text"][:80],
                "text": r["choices"][0]["text"],
                "finish_reason": r["choices"][0].get("finish_reason"),
            }
        except urllib.error.HTTPError as e:
            return {"ok": False, "wall_s": round(time.perf_counter() - t0, 4),
                     "error": f"HTTP {e.code}: {e.read().decode(errors='replace')[:2000]}"}
        except Exception as e:
            return {"ok": False, "wall_s": round(time.perf_counter() - t0, 4), "error": f"{type(e).__name__}: {e}"[:2000]}

    def tokenize(self, text):
        r = json.load(self.post("/tokenize", {"model": self.model, "prompt": text}, 120))
        return r["count"]

    def models(self):
        r = json.load(urllib.request.urlopen(self.base_url + "/v1/models", timeout=60))
        return [m["id"] for m in r["data"]]


BASE_FIXTURE = (
    "The following is a technical reference on distributed systems. Section {n}: consensus protocols. "
    "A replicated state machine applies a deterministic sequence of commands to identical copies of state on every node. "
    "Leader election chooses one node to order commands; followers acknowledge log entries; commitment requires a quorum. "
    "Failure detectors use heartbeats and timeouts; network partitions cause split votes; term numbers resolve stale leaders. "
)


def build_fixture(client, target, prefix=""):
    body = ""
    n = 0
    while client.tokenize(prefix + body) < target:
        n += 1
        body += BASE_FIXTURE.format(n=n)
    words = body.split(" ")
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi) // 2
        if client.tokenize(prefix + " ".join(words[:mid])) < target:
            lo = mid + 1
        else:
            hi = mid
    text = prefix + " ".join(words[:lo])
    c = client.tokenize(text)
    tries = 0
    while c != target and tries < 40:
        tries += 1
        if c > target:
            text = text[:-1]
        else:
            text += " a"
        c = client.tokenize(text)
    return text, c


def save(out_dir, name, obj):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, name)
    with open(p + ".tmp", "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(p + ".tmp", p)
    print("saved", p, flush=True)
