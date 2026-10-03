# usage: qref.py OUT.json  -- greedy 64-token completions with top-5 logprobs on fixed prompts
import json, sys, urllib.request
URL = "http://127.0.0.1:8040/v1/completions"
PROMPTS = [
 "The capital of France is",
 "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
 "Photosynthesis is the process by which",
 "In 1969, Neil Armstrong",
 "To make a cup of tea, first",
 "SELECT name, COUNT(*) FROM orders",
 "The quick brown fox jumps over the lazy dog. The quick brown",
 "Machine learning models are trained by",
 "import numpy as np\nx = np.linspace(0, 1, 100)\ny =",
 "Once upon a time, in a small village by the sea,",
 "The derivative of x^3 is",
 "Tokyo is the capital of",
]
out = []
for p in PROMPTS:
    body = {"model": "deepseek-v4.1-flash", "prompt": p, "max_tokens": 64, "temperature": 0, "logprobs": 5}
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
    lp = r["choices"][0]["logprobs"]
    out.append({"prompt": p, "text": r["choices"][0]["text"], "tokens": lp["tokens"], "token_logprobs": lp["token_logprobs"], "top": lp["top_logprobs"]})
json.dump(out, open(sys.argv[1], "w"), indent=1)
print("saved", len(out), "prompts")
