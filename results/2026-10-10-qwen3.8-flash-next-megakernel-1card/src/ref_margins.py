# llama.cpp top-2 probabilities at every generated position of the greedy reference (to judge the disagreements)
import json, sys, urllib.request
URL = "http://127.0.0.1:8090"
def post(path, body):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=3600).read())
refdir, scorefile, out = sys.argv[1], sys.argv[2], sys.argv[3]
miss = {}
cur = None
for line in open(scorefile):
    line = line.strip()
    if line.startswith("== "): cur = line[3:]; miss[cur] = []
    elif line.startswith("miss at"): miss[cur].append(int(line.split()[-1]))
res = {}
for name, pos in miss.items():
    ids = [int(x) for x in open(f"{refdir}/{name}.ids").read().split()]
    P = json.load(open(f"{refdir}/{name}.json"))["prompt_tokens"]
    r = post("/completion", {"prompt": ids[:P], "n_predict": len(ids) - P, "temperature": 0.0, "top_k": 1, "samplers": ["top_k"],
                             "cache_prompt": False, "n_probs": 2, "post_sampling_probs": False, "return_tokens": True})
    probs = r["completion_probabilities"]
    margins = []
    for i, cp in enumerate(probs):
        top = cp["top_logprobs"] if "top_logprobs" in cp else cp.get("probs", [])
        lp = [t["logprob"] for t in top][:2]
        margins.append(lp[0] - lp[1] if len(lp) == 2 else None)
    at = {p: margins[p - P] for p in pos if 0 <= p - P < len(margins)}
    allm = sorted(m for m in margins if m is not None)
    res[name] = {"miss_logprob_gap": at, "median_gap_all": allm[len(allm) // 2], "n": len(allm),
                 "same_tokens": r["tokens"] == ids[P:]}
    print(name, json.dumps(res[name]), flush=True)
json.dump(res, open(out, "w"), indent=1)
