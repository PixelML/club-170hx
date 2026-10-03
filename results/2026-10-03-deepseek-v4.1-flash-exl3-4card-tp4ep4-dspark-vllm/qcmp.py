import json, sys
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
same_text = sum(x["text"] == y["text"] for x, y in zip(a, b))
agree = tot = 0; drift = []
for x, y in zip(a, b):
    for i, (tx, ty) in enumerate(zip(x["tokens"], y["tokens"])):
        tot += 1
        if tx != ty:
            break
        agree += 1
        drift.append(abs(x["token_logprobs"][i] - y["token_logprobs"][i]))
first_div = [next((i for i, (tx, ty) in enumerate(zip(x["tokens"], y["tokens"])) if tx != ty), 64) for x, y in zip(a, b)]
print(f"identical completions {same_text}/{len(a)} | tokens before first divergence: median {sorted(first_div)[len(first_div)//2]} min {min(first_div)} | mean |dlogprob| on shared prefix {sum(drift)/max(len(drift),1):.4f}")
