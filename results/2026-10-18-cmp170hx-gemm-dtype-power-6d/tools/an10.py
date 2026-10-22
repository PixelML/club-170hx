# Builds receipts/card_6d_140_300/TABLE_*.md: python an10.py receipts/card_6d_140_300
import json, csv, sys, datetime, statistics
S = sys.argv[1]
PER_CLK = dict(int8=286720, bf16=143360, fp16=143360, tf32=71680, fp32=8960)  # dense ops/clock, 70 SMs (A100 per-SM rates)
def smi(cap):
    out = []
    for r in csv.reader(open(f'{S}/smi10_cmp-f_{cap}.csv')):
        try:
            t = datetime.datetime.strptime(r[0].strip(), '%Y/%m/%d %H:%M:%S.%f').timestamp()
            out.append((t, float(r[1]), float(r[2]), float(r[3]), float(r[4])))
        except Exception: pass
    return out
rows = {}
for cap in (300, 140):
    sm = smi(cap)
    for l in open(f'{S}/res10_cmp-f_{cap}.jsonl'):
        j = json.loads(l); w = [x for x in sm if j['t0'] <= x[0] <= j['t1']]
        j['W'] = statistics.mean(x[1] for x in w); j['Wmax'] = max(x[1] for x in w); j['MHz'] = statistics.mean(x[2] for x in w); j['C'] = max(x[4] for x in w)
        dt = j['dtype'].replace('_rowB', '')
        j['eff'] = j['median'] / (PER_CLK[dt] * j['MHz'] * 1e6 / 1e12) if dt in PER_CLK else None
        j['perW'] = j['median'] / j['W']
        key = (j['shape'], j['M'] if 'M' in j else 0, j.get('K', 0), j.get('N', 0), j['dtype'])
        rows.setdefault(key, {})[cap] = j
print('| test | dtype | 140 W: rate | MHz | avg W | % of peak | 300 W: rate | MHz | avg W | % of peak | gain 140->300 |')
print('|---|---|---|---|---|---|---|---|---|---|---|')
for key, v in rows.items():
    a, b = v.get(140), v.get(300)
    name = f"square {key[1]}" if key[0] == 'square' else (f"M={key[1]} x {key[2]}x{key[3]}" if key[0] == 'model' else 'HBM copy 2 GiB')
    def f(j): return f"{j['median']:.1f} ({j['lo']:.0f}-{j['hi']:.0f}) | {j['MHz']:.0f} | {j['W']:.0f} | " + (f"{100*j['eff']:.0f}%" if j['eff'] else '-')
    print(f"| {name} | {key[4]} | {f(a)} | {f(b)} | {100*(b['median']/a['median']-1):+.0f}% |")
for cap in (140, 300):
    sm = smi(cap); print(f"cap {cap}: max W {max(x[1] for x in sm):.0f}, max C {max(x[4] for x in sm):.0f}, HBM MHz {statistics.mean(x[3] for x in sm):.0f}")
