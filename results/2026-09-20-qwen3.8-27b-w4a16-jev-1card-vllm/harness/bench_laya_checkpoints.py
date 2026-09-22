import json, math, sys

RECEIPTS = "/home/ubuntu/WIP/club-170hx/results/2026-09-20-qwen3.8-27b-w4a16-jev-1card-vllm/receipts"
rows = [json.loads(l) for l in open(f"{RECEIPTS}/labeled.jsonl")]
jev_answers = json.load(open('/tmp/jev_answers.json'))
prev = json.load(open('/tmp/positioning_bench_raw.json'))

def norm_expected(row):
    exp, qtype, names = row['expected'], row['question']['type'], row['option_names']
    if qtype == 'noul':
        return 'yes' if str(exp).lower() in ('yes','true','1') else 'no'
    s = str(exp)
    if qtype == 'score' and s.isdigit() and int(s) < len(names):
        return names[int(s)].lower()
    return s.lower()

def our_read(i, row):
    lps = row['label_logprobs']; mx = max(lps.values())
    ex = {k: math.exp(v-mx) for k,v in lps.items()}; tot = sum(ex.values())
    probs = {k: v/tot for k,v in ex.items()}
    return max(probs, key=probs.get), probs

def jev(i, row):
    a = jev_answers[str(i)]
    if 'error' in a: return None, None
    t = row['question']['type']
    if t == 'choice': return a['choice'], a['probabilities']
    if t == 'noul':
        p = float(a['noul']); return ('yes' if p >= 0.5 else 'no'), {'yes': p, 'no': 1-p}
    probs = {str(k): v for k,v in a.get('probabilities', {}).items()}
    if not probs: return None, None
    best = max(probs, key=probs.get)
    names = row['option_names']
    return (names[int(best)].lower() if best.isdigit() and int(best) < len(names) else best), probs

def jsd(p, q):
    keys = set(p) | set(q)
    pp = {k: max(p.get(k,1e-12),1e-12) for k in keys}
    qq = {k: max(q.get(k,1e-12),1e-12) for k in keys}
    sp, sq = sum(pp.values()), sum(qq.values())
    pp = {k: v/sp for k,v in pp.items()}; qq = {k: v/sq for k,v in qq.items()}
    return sum(0.5*pp[k]*math.log(pp[k]/((pp[k]+qq[k])/2)) + 0.5*qq[k]*math.log(qq[k]/((pp[k]+qq[k])/2)) for k in keys)

jev_preds, jev_probs = {}, {}
for i, row in enumerate(rows):
    jev_preds[i], jev_probs[i] = jev(i, row)

acc_our = sum(1 for i,row in enumerate(rows) if our_read(i,row)[0].lower() == norm_expected(row).lower())
acc_jev = sum(1 for i,row in enumerate(rows) if jev_preds[i] and str(jev_preds[i]).lower() == norm_expected(row).lower())
js_our = [jsd(our_read(i,row)[1], jev_probs[i]) for i,row in enumerate(rows) if row['question']['type']=='choice']
table = {'our_read': {'accuracy': round(acc_our/len(rows),3),
                      'agreement_with_jev': round(sum(1 for i,row in enumerate(rows) if str(our_read(i,row)[0]).lower()==str(jev_preds[i]).lower())/len(rows),3),
                      'js_with_jev_choice_mean': round(sum(js_our)/len(js_our),4)},
         'jev_1_13': {'accuracy': round(acc_jev/len(rows),3)}}
json.dump(table, open('/tmp/laya_full_bench.json','w'), indent=1)
print('base table', json.dumps(table), flush=True)

for name, sub in (('typed', 'typed-decisions'), ('base', None), ('multi', 'multilingual')):
    try:
        import laya
        agent = laya.load("convaiinnovations/laya", subfolder=sub)
        acc = 0; agj = 0; agj_n = 0; js = []
        per_type = {}
        for i, row in enumerate(rows):
            q = row['question']; t = q['type']; crit = q.get('criteria', {})
            if t == 'choice':
                questions = {row['id']: {"type": "choice", "instructions": q['instructions'], "criteria": dict(crit)}}
            elif t == 'noul':
                questions = {row['id']: {"type": "noul", "instructions": q['instructions']}}
            else:
                levels = list(crit.values()) if isinstance(crit, dict) else list(crit)
                questions = {row['id']: {"type": "score", "instructions": q['instructions'], "criteria": levels}}
            try:
                res = agent.predict(row['state'], questions)
                a = res['answers'][row['id']]
                if 'choice' in a: pred, probs = a['choice'], a.get('probabilities')
                elif 'noul' in a:
                    p = float(a['noul']); pred, probs = ('yes' if p >= 0.5 else 'no'), {'yes': p, 'no': 1-p}
                else:
                    probs = {str(k): float(v) for k,v in a.get('probabilities', {}).items()}
                    names = row['option_names']
                    best = max(probs, key=probs.get)
                    pred = names[int(best)].lower() if best.isdigit() and int(best) < len(names) else best.lower()
                    probs = {names[int(k)].lower() if k.isdigit() and int(k) < len(names) else k: v for k,v in probs.items()}
            except Exception as e:
                pred, probs = None, None
            gold = norm_expected(row)
            ok = pred is not None and str(pred).lower() == gold.lower()
            acc += ok
            per_type.setdefault(t, [0,0]); per_type[t][0] += 1; per_type[t][1] += ok
            if pred is not None:
                agj_n += 1; agj += str(pred).lower() == str(jev_preds[i]).lower()
                if row['question']['type'] == 'choice' and probs: js.append(jsd(probs, jev_probs[i]))
            print(f'  {name} {i}: {pred} (gold {gold})', flush=True)
        table[f'laya_{name}'] = {'accuracy': round(acc/len(rows),3),
                                 'agreement_with_jev': round(agj/agj_n,3) if agj_n else None,
                                 'js_with_jev_choice_mean': round(sum(js)/len(js),4) if js else None,
                                 'per_type': {t: {'n': a, 'acc': round(b/a,3)} for t,(a,b) in sorted(per_type.items())}}
        json.dump(table, open('/tmp/laya_full_bench.json','w'), indent=1)
        print(f'{name} DONE:', json.dumps(table[f'laya_{name}']), flush=True)
    except Exception as e:
        print(f'{name} FAILED: {e}', flush=True)
print('ALL DONE', flush=True)
