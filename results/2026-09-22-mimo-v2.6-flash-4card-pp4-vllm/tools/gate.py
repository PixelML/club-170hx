import json, sys, urllib.request
host = "http://localhost:8000"
P = [("What is the capital of Japan? One word.", "tokyo"),
     ("What is 17*23? Answer with just the number.", "391"),
     ("Name the chemical symbol for gold.", "au"),
     ("Translate 'thank you' to French.", "merci")]
out = []
for q, want in P:
    r = []
    for _ in range(3):
        body = json.dumps({"model": "mimo-v2.6-flash", "messages": [{"role": "user", "content": q}],
                           "temperature": 0, "max_tokens": 1024}).encode()
        d = json.load(urllib.request.urlopen(urllib.request.Request(host + "/v1/chat/completions", body, {"Content-Type": "application/json"})))
        r.append(d["choices"][0]["message"].get("content") or "")
    out.append({"prompt": q, "expect": want, "answers": r, "correct": all(want in a.lower() for a in r), "identical": len(set(r)) == 1})
    print(json.dumps(out[-1])[:220])
json.dump(out, open(sys.argv[1], "w"), indent=1)
print("GATE", "PASS" if all(o["correct"] and o["identical"] for o in out) else "FAIL")
