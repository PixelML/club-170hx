"""Short-prompt quality probe (greedy). usage: probe.py OUT.json"""
import json, sys, urllib.request
P = [("prime-bare", [{"role": "user", "content": "Write a Python function that checks if a number is prime."}], {}),
     ("prime-think", [{"role": "user", "content": "Write a Python function that checks if a number is prime."}], {"chat_template_kwargs": {"thinking": True}}),
     ("prime-system", [{"role": "system", "content": "You are a helpful coding assistant. Answer in English."}, {"role": "user", "content": "Write a Python function that checks if a number is prime."}], {}),
     ("is_prime-97", [{"role": "user", "content": "Write a Python function is_prime(n). Then state what is_prime(97) returns, as the last line: RESULT=<value>."}], {}),
     ("sky", [{"role": "user", "content": "Explain in two sentences why the sky is blue."}], {})]
out = []
for name, msgs, extra in P:
    body = dict({"model": "dsv4v", "messages": msgs, "temperature": 0, "max_tokens": 300}, **extra)
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:18098/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
    c = r["choices"][0]["message"]["content"] or ""
    on_topic = ("def " in c and "prime" in c) if "prime" in name else ("scatter" in c.lower())
    out.append({"name": name, "content": c, "on_topic": on_topic, "usage": r["usage"]})
    print(name, "ON" if on_topic else "OFF", repr(c[:100]), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
