# Greedy reference continuations from llama-server (same GGUF), saved as token ids for fnx_mk --score.
import json, sys, urllib.request
URL = "http://127.0.0.1:8090"
PROMPTS = {
    "code":  "Write a quicksort function in C with comments.",
    "chat":  "Explain how a hash map works, in simple terms.",
    "story": "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "math":  "A train travels 120 km in 1.5 hours, then 200 km in 2.5 hours. What is its average speed? Show your steps.",
}
def post(path, body):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=3600).read())
out = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 256
for name, p in PROMPTS.items():
    text = "<|im_start|>user\n" + p + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    toks = post("/tokenize", {"content": text, "add_special": False, "parse_special": True})["tokens"]
    r = post("/completion", {"prompt": toks, "n_predict": n, "temperature": 0.0, "top_k": 1, "cache_prompt": False,
                             "return_tokens": True, "ignore_eos": False, "samplers": ["top_k"]})
    gen = r["tokens"]
    with open(f"{out}/{name}.ids", "w") as f:
        f.write("\n".join(str(t) for t in toks + gen) + "\n")
    t = r.get("timings", {})
    print(name, "prompt", len(toks), "gen", len(gen), "decode tok/s %.2f" % t.get("predicted_per_second", 0),
          "prompt tok/s %.1f" % t.get("prompt_per_second", 0), flush=True)
    json.dump({"prompt_tokens": len(toks), "gen": len(gen), "timings": t, "content": r["content"]}, open(f"{out}/{name}.json", "w"))
