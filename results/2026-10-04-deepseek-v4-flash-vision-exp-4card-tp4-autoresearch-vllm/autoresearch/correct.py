import json, time, urllib.request
BASE="http://127.0.0.1:18098/v1/chat/completions"
CASES=[("What is 17 * 23? Answer with the number only.","391"),
       ("What is the capital of Australia? One word.","Canberra"),
       ("A train leaves at 09:40 and arrives at 13:15. How many minutes is the trip? Answer with the number only.","215"),
       ("Write a Python function is_prime(n). Then state what is_prime(97) returns, as the last line: RESULT=<value>.","RESULT=True"),
       ("Spell the word ELEPHANT backwards, uppercase, no spaces.","TNAHPELE")]
ok=0; RES=[]
for q,a in CASES:
    body={"model":"dsv4v","messages":[{"role":"user","content":q}],"temperature":0,"max_tokens":1500}
    t=time.time(); r=json.load(urllib.request.urlopen(urllib.request.Request(BASE,json.dumps(body).encode(),{"Content-Type":"application/json"}),timeout=900))
    m=r["choices"][0]["message"]; c=(m.get("content") or ""); rs=(m.get("reasoning_content") or m.get("reasoning") or "")
    hit=a.lower() in c.lower(); ok+=hit; RES.append({"prompt": q, "expect": a, "content": c, "reasoning_chars": len(rs), "completion_tokens": r["usage"]["completion_tokens"], "pass": bool(hit)})
    print(("PASS" if hit else "FAIL"), repr(q[:40]), "->", repr(c.strip()[-80:]), "| reasoning chars", len(rs), "| %.1fs"%(time.time()-t), "| tokens", r["usage"]["completion_tokens"], flush=True)
print(f"{ok}/{len(CASES)} correct")
import sys
if len(sys.argv) > 1: json.dump({"cases": RES, "correct": ok, "total": len(CASES)}, open(sys.argv[1], "w"), indent=1)
