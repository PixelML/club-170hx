"""Long-context degeneration test: 4 concurrent requests, ~28-34k-token real-text prompt + a reasoning task, 4,000 output tokens.

Default sampling comes from the model's generation_config (T=1.0, top_p=0.95). SAMPLING='{"temperature":0.6}' overrides it.
usage: GLM_URL=http://<server>:8030/v1 python3 glm_repro.py"""
import json, os, re, sys, urllib.request, concurrent.futures as cf
from pathlib import Path
URL = os.environ.get("GLM_URL", "http://127.0.0.1:8030/v1") + "/chat/completions"
# bench_long.py (corpus + prompt builder) is published with the 2026-10-04 v1.7.0 notebook.
corpus_mod = (Path(__file__).resolve().parent.parent / "2026-10-04-glm-5.3-flash-morrowmake-v1.7.0-4card-tp4-pp4-p2p-vllm" / "bench_long.py").read_text()
ns = {}; exec(corpus_mod.split("mode, out = sys.argv")[0], ns)  # loads corpus + prompt_of
TASK = ("Using the text above only as background, now compute step by step the equation of time for 2024-11-03 12:00 UTC "
        "with the low-precision solar formulas (mean longitude, mean anomaly, ecliptic longitude, right ascension). Show the arithmetic.")
def run(i):
    prompt = ns["prompt_of"](40000, (i * 300000) % 2_000_000, TASK)
    body = {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": prompt}], "max_tokens": 4000, **json.loads(os.environ.get("SAMPLING", "{}"))}
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800))
    m = r["choices"][0]["message"]; th = (m.get("reasoning_content") or m.get("reasoning") or ""); ct = m.get("content") or ""
    txt = th + ct
    rep = re.search(r"(.{1,12}?)\1{15,}", txt, re.S)
    return dict(i=i, prompt_tokens=r["usage"]["prompt_tokens"], out_tokens=r["usage"]["completion_tokens"], finish=r["choices"][0]["finish_reason"],
                loop=(rep.group(1)[:12], rep.start()) if rep else None, tail=txt[-160:].replace("\n", " "))
alone = []
with cf.ThreadPoolExecutor(4) as ex: conc = list(ex.map(run, range(4, 8)))
for name, rows in (("alone", alone), ("concurrent", conc)):
    for d in rows: print(name, json.dumps(d)[:400])
bad = sum(1 for d in conc if d["loop"] or (d["finish"] == "stop" and d["out_tokens"] < 2500 and "equation" not in d["tail"].lower() and "eot" not in d["tail"].lower()))
print("BROKEN", bad, "of", len(conc))
