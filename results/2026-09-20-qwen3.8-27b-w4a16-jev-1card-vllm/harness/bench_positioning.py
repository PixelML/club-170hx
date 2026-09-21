import json, math, sys, collections

RECEIPTS = "/home/ubuntu/WIP/club-170hx/results/2026-09-20-qwen3.8-27b-w4a16-jev-1card-vllm/receipts"

rows = [json.loads(l) for l in open(f"{RECEIPTS}/labeled.jsonl")]

# ---- System A: our jev read (committed outputs) ----
def our_system(row):
    lps = row["label_logprobs"]
    names = row["option_names"]
    mx = max(lps.values())
    ex = {k: math.exp((v - mx)) for k, v in lps.items()}
    tot = sum(ex.values())
    probs = {k: v / tot for k, v in ex.items()}
    choice = max(probs, key=probs.get)
    return choice, probs

# ---- System B: Laya (base English checkpoint, zero-shot) ----
def laya_system():
    import laya
    agent = laya.load("convaiinnovations/laya")
    out = {}
    for i, row in enumerate(rows):
        q = row["question"]
        qtype = q["type"]
        if qtype == "choice":
            questions = {row["id"]: {"type": "choice", "instructions": q["instructions"],
                                     "criteria": dict(q["criteria"])}}
        elif qtype == "noul":
            questions = {row["id"]: {"type": "noul", "instructions": q["instructions"]}}
        else:
            questions = {row["id"]: {"type": "score", "instructions": q["instructions"],
                                     "criteria": list(q["criteria"].values())
                                     if isinstance(q.get("criteria"), dict) else list(q["criteria"])}}
        try:
            res = agent.predict(row["state"], questions)
            a = res["answers"][row["id"]]
            out[i] = a
        except Exception as e:
            out[i] = {"error": str(e)[:200]}
    return out

# ---- System C: GLiNER 2.5 Multi (zero-shot classification) ----
def gliner_system():
    from gliner2 import AutoExtractor
    model = AutoExtractor.from_pretrained("fastino/gliner2.5-multi-v1", map_location="cpu")
    out = {}
    for i, row in enumerate(rows):
        q = row["question"]
        qtype = q["type"]
        crit = q.get("criteria", {})
        if qtype == "choice":
            labels = list(crit.keys())
        elif qtype == "noul":
            labels = ["yes", "no"]
        else:
            labels = list(crit.keys()) if isinstance(crit, dict) else list(crit)
        try:
            res = model.classify_text(row["state"], {row["id"]: {"labels": labels, "multi_label": False}})
            out[i] = res[row["id"]]
        except Exception as e:
            out[i] = {"error": str(e)[:200]}
    return out

# ---- scoring helpers ----
def norm_expected(row):
    exp = row["expected"]
    qtype = row["question"]["type"]
    names = row["option_names"]
    if qtype == "noul":
        return "yes" if str(exp).lower() in ("yes", "true", "1") else "no"
    if qtype == "score":
        # expected may be a level index or level name
        s = str(exp)
        if s.isdigit() and int(s) < len(names):
            return names[int(s)]
        return s
    return exp

def jsd(p, q, eps=1e-9):
    keys = set(p) | set(q)
    pp = {k: p.get(k, eps) for k in keys}
    qq = {k: q.get(k, eps) for k in keys}
    sp, sq = sum(pp.values()), sum(qq.values())
    pp = {k: v / sp for k, v in pp.items()}
    qq = {k: v / sq for k, v in qq.items()}
    d = 0.0
    for k in keys:
        m = (pp[k] + qq[k]) / 2
        if pp[k] > 0: d += 0.5 * pp[k] * math.log(pp[k] / m)
        if qq[k] > 0: d += 0.5 * qq[k] * math.log(qq[k] / m)
    return d

mode = sys.argv[1] if len(sys.argv) > 1 else "all"

our_preds, laya_raw, gliner_raw = {}, {}, {}
for i, row in enumerate(rows):
    our_preds[i] = our_system(row)

if mode in ("all", "laya"):
    laya_raw = laya_system()
if mode in ("all", "gliner"):
    gliner_raw = gliner_system()

def laya_choice(i):
    a = laya_raw.get(i, {})
    if "error" in a or "choice" not in a: return None, None
    return a.get("choice"), a.get("probabilities")

def laya_pred_label(i):
    c, _ = laya_choice(i)
    if c is None: return None
    # noul/score answers come back as numbers/levels; map to our label space
    row = rows[i]
    qtype = row["question"]["type"]
    names = row["option_names"]
    if qtype == "noul":
        try:
            return "yes" if float(c) >= 0.5 else "no"
        except (TypeError, ValueError):
            return str(c).lower()
    if qtype == "score":
        try:
            v = float(c)
            return names[round(v)] if round(v) < len(names) else names[-1]
        except (TypeError, ValueError):
            return str(c)
    return c

def gliner_pred_label(i):
    r = gliner_raw.get(i, {})
    if "error" in r: return None
    if isinstance(r, dict):
        for k in ("value", "label", row_id := rows[i]["id"]):
            if k in r: return r[k]
        if len(r) == 1:
            v = list(r.values())[0]
            if isinstance(v, str): return v
            if isinstance(v, dict): return v.get("value") or (v.get("labels") or [None])[0]
    if isinstance(r, str): return r
    return None

n = len(rows)
res = {"n": n}
acc = {}
for name, get in (("our_read", lambda i: our_preds[i][0]),
                  ("laya", laya_pred_label),
                  ("gliner", gliner_pred_label)):
    ok = sum(1 for i in range(n) if (get(i) or "").lower().strip() == norm_expected(rows[i]).lower().strip())
    acc[name] = round(ok / n, 3)
res["accuracy_vs_gold"] = acc

agree = {}
for a, b, ga, gb in (("our_read", "laya", lambda i: our_preds[i][0], laya_pred_label),
                     ("our_read", "gliner", lambda i: our_preds[i][0], gliner_pred_label),
                     ("laya", "gliner", laya_pred_label, gliner_pred_label)):
    both = [i for i in range(n) if ga(i) and gb(i)]
    if both:
        agree[f"{a}~{b}"] = round(sum(1 for i in both if ga(i).lower() == gb(i).lower()) / len(both), 3)
res["top1_agreement"] = agree

# distribution alignment where Laya gives probabilities
if laya_raw:
    ds = []
    for i in range(n):
        c, probs = laya_choice(i)
        if probs:
            ds.append(jsd(our_preds[i][1], probs))
    if ds:
        res["js_divergence_our_vs_laya_choice_mean"] = round(sum(ds) / len(ds), 4)

# per-question-type accuracy for our read and laya
by_type = collections.defaultdict(lambda: collections.Counter())
for i, row in enumerate(rows):
    t = row["question"]["type"]
    by_type[t]["n"] += 1
    if our_preds[i][0].lower() == norm_expected(row).lower(): by_type[t]["our"] += 1
    lp = laya_pred_label(i)
    if lp and lp.lower() == norm_expected(row).lower(): by_type[t]["laya"] += 1
    gp = gliner_pred_label(i)
    if gp and gp.lower() == norm_expected(row).lower(): by_type[t]["gliner"] += 1
res["by_type"] = {t: dict(c) for t, c in by_type.items()}

res["laya_errors"] = sum(1 for i in range(n) if "error" in laya_raw.get(i, {}))
res["gliner_errors"] = sum(1 for i in range(n) if "error" in gliner_raw.get(i, {}))

json.dump(res, open("/tmp/positioning_bench.json", "w"), indent=1)
print(json.dumps(res, indent=1))

# stash raw outputs for the receipt
json.dump({"laya": {str(k): v for k, v in laya_raw.items()},
           "gliner": {str(k): v for k, v in gliner_raw.items()}},
          open("/tmp/positioning_bench_raw.json", "w"), indent=1, default=str)
