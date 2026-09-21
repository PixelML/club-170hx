#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Receipt harness for the Qwen3.8-27B Jev-compatible endpoint experiment.

Runs against a live pair of endpoints and writes one raw JSON receipt per
probe into --out (default ../receipts):

  env          runtime/model/hardware pins and the resolved engine config
  prompts      rendered prompt + token ids + sha256 + label symbols per type
  mask         masked (allowed_token_ids) vs unmasked label logprobs
  determinism  identical / interleaved / concurrent repeat reads
  temperature  API probabilities vs softmax(label_logprobs / T)
  permutations option-order rotations, per rotation and averaged
  latency      warm prefix-cache reads vs cache-busted reads
  labeled      a small labelled set: per-example predictions + metrics
  negatives    rejected requests (schema, images, caps)
  crosscheck   in-process vLLM LLM reads vs the HTTP path (needs a free GPU)

Every receipt is sanitized: model/workspace paths become <model-dir> /
<workspace>, PIDs and PCI bus ids are dropped.

  JEV_UPSTREAM=http://127.0.0.1:18030 JEV_API=http://127.0.0.1:18031 \
  MODEL_DIR=/models/Qwen3.8-27B-GPTQ-4bit \
  python jev_probe.py all --out ../receipts
"""

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import jev_server as J  # noqa: E402  (same directory; single source of the prompt format)

CFG = {}


# ----------------------------------------------------------------------------
# Plumbing
# ----------------------------------------------------------------------------


def mask(text):
    """Sanitize paths out of anything written to a receipt."""
    if not isinstance(text, str):
        return text
    text = text.replace(os.environ.get("MODEL_DIR", ""), "<model-dir>")
    text = text.replace(os.path.expanduser("~/WIP"), "<workspace>")
    text = text.replace(os.path.expanduser("~"), "<home>")
    text = re.sub(r"\bpid=\d+", "pid=<pid>", text)
    text = re.sub(r"00000000:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9]", "<pci>", text)
    return text


def post(url, body, timeout=600):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def api(body):
    return post(CFG["api"] + "/v1/systemone", body)


def raw_completion(prompt_ids, logprobs, allowed_token_ids=None, temperature=1.0,
                   neutral=True):
    body = {
        "model": CFG["model"],
        "prompt": prompt_ids,
        "max_tokens": 1,
        "temperature": temperature,
        "logprobs": logprobs,
        "return_tokens_as_token_ids": True,
    }
    if neutral:
        # The served model's generation_config ships top_k=20 / top_p=0.95 and
        # vLLM adopts them as server defaults; a calibration read must not be
        # renormalized over a nucleus.
        body["top_p"] = 1.0
        body["top_k"] = -1
    if allowed_token_ids is not None:
        body["allowed_token_ids"] = allowed_token_ids
    return post(CFG["upstream"] + "/v1/completions", body)


def label_logprobs_from(resp):
    top = resp["choices"][0]["logprobs"]["top_logprobs"][0]
    out = {}
    for k, v in top.items():
        if isinstance(k, str) and ":" in k:
            try:
                out[int(k.rsplit(":", 1)[1])] = v
                continue
            except ValueError:
                pass
        out[int(k) if str(k).isdigit() else k] = v
    return out


def dump(name, payload):
    path = Path(CFG["out"]) / name
    path.write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n")
    print(f"wrote {path}")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


# ----------------------------------------------------------------------------
# Fixtures: the questions every probe reads, and the labelled set
# ----------------------------------------------------------------------------

STATE_BILLING = ("I have been trying to connect my Stripe account for 3 days "
                 "and it keeps failing. I am losing sales.")
STATE_ANGRY = "Third time I am writing. My order still has not arrived. Refund me."

Q_CHOICE = {
    "type": "choice",
    "instructions": "Which team should handle this",
    "criteria": {"billing": "Payment or subscription issues",
                 "technical": "Bugs or integration problems",
                 "sales": "Pricing or account questions"},
}
Q_NOUL = {"type": "noul", "instructions": "Is this customer angry?"}
Q_SCORE = {
    "type": "score",
    "instructions": "How urgent is this inquiry?",
    "criteria": ["Not urgent", "Within a few days", "Today", "Immediately"],
}
# Four options where three are implausible: use it to see whether the
# checkpoint's nucleus default drops a low-mass label.
Q_LONGSHOT = {
    "type": "choice",
    "instructions": "Which team should handle this",
    "criteria": {"billing": "Payment or subscription issues",
                 "technical": "Bugs or integration problems",
                 "sales": "Pricing or account questions",
                 "legal": "Contract and compliance questions"},
}
FIXTURES = [
    ("choice-billing", STATE_BILLING, "department", Q_CHOICE),
    ("noul-angry", STATE_ANGRY, "is_angry", Q_NOUL),
    ("score-urgent", STATE_ANGRY, "urgency", Q_SCORE),
]


def labelled_set():
    """A small author-labelled set: (case, state, question, expected label).

    Labels are assigned by the author from the text, not from a gold corpus.
    Kept short and unambiguous so the only claim it supports is descriptive.
    """
    rows = []

    def add(case, state, qid, q, expected):
        rows.append({"case": case, "state": state, "id": qid, "question": q,
                     "expected": expected})

    billing = [
        ("My card was charged twice for the same subscription.", "billing"),
        ("I need to update the VAT number on my invoices.", "billing"),
        ("The annual plan renewed but I wanted to cancel before that.", "billing"),
        ("How do I change my payment method from card to invoice?", "billing"),
        ("Our coupon code is rejected at checkout.", "billing"),
        ("I was downgraded without warning and want the difference refunded.", "billing"),
    ]
    technical = [
        ("The API returns a 500 on every POST /v1/orders call.", "technical"),
        ("After the last SDK update, webhooks never fire.", "technical"),
        ("The OAuth callback loops back to the login page.", "technical"),
        ("Uploading a 200 MB file through the dashboard fails halfway.", "technical"),
        ("Our nightly sync job times out since Tuesday.", "technical"),
        ("The Python client raises a certificate error on Linux only.", "technical"),
    ]
    sales = [
        ("Can you quote 500 seats with a volume discount?", "sales"),
        ("Do you offer an education licence for a whole university?", "sales"),
        ("We want a demo before buying the enterprise tier.", "sales"),
        ("Is there a partner programme for resellers?", "sales"),
        ("We are comparing you with a competitor, what is the price difference?", "sales"),
        ("Can we start a paid pilot for two teams next month?", "sales"),
    ]
    for text, expected in billing + technical + sales:
        add("routing", text, "department", Q_CHOICE, expected)

    angry = [
        ("This is unacceptable and I want a manager now.", "yes"),
        ("No reply for a week, I am extremely disappointed.", "yes"),
        ("Still broken after three attempts. Ridiculous.", "yes"),
        ("Thanks, that fixed it, very helpful.", "no"),
        ("Quick question whenever you have a moment.", "no"),
        ("Appreciate the fast response earlier today.", "no"),
        ("Why is this still not working? I have lost a full day.", "yes"),
        ("Could you confirm the invoice email address?", "no"),
    ]
    for text, expected in angry:
        add("sentiment", text, "is_angry", Q_NOUL, expected)

    urgency = [
        ("Production is down for all our customers right now.", "Immediately"),
        ("Our launch is next week and SSO is still untested.", "Today"),
        ("We would like to plan a migration sometime this quarter.", "Within a few days"),
        ("Just collecting information for next year's budget.", "Not urgent"),
        ("Data is being lost every hour the job stays broken.", "Immediately"),
        ("Can you review our draft integration plan this week?", "Within a few days"),
        ("No rush, but please confirm receipt of this ticket.", "Not urgent"),
        ("The payment page has been failing since this morning.", "Today"),
    ]
    for text, expected in urgency:
        add("urgency", text, "urgency", Q_SCORE, expected)

    numeric = [
        ("The order total is 120 dollars but the invoice says 99 dollars.", "yes"),
        ("I ordered 3 items and received 3 items.", "no"),
        ("We have 50 seats but the dashboard shows 45 active users.", "yes"),
        ("The plan costs 20 per seat and we pay 200 for ten seats.", "no"),
        ("The refund of 40 was issued twice, so 80 in total.", "yes"),
        ("Delivery was promised in 2 days and took 2 days.", "no"),
    ]
    for text, expected in numeric:
        add("numeric", text, "mismatch",
            {"type": "noul",
             "instructions": "Do the two quantities in the text disagree?",
             "criteria": {"true": "the numbers disagree",
                          "false": "the numbers agree"}},
            expected)

    absent = [
        ("The customer asks about a refund but never mentions a team.",
         "billing"),
        ("The message is about an outage during a demo.", "technical"),
    ]
    for text, expected in absent:
        add("routing", text, "department", Q_CHOICE, expected)
    return rows


# ----------------------------------------------------------------------------
# Probes
# ----------------------------------------------------------------------------


def probe_env():
    import torch
    import vllm
    import transformers

    model_dir = CFG["model_dir"]
    cfg_path = Path(model_dir) / "config.json"
    model_cfg = json.loads(cfg_path.read_text())
    files = {name: sha256_file(Path(model_dir) / name)
             for name in ("config.json", "chat_template.jinja", "tokenizer.json",
                          "generation_config.json")
             if (Path(model_dir) / name).exists()}
    gpu = subprocess.run(
        ["nvidia-smi",
         "--query-gpu=index,name,memory.total,driver_version,compute_cap",
         "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    log_tail = []
    if CFG.get("log"):
        lines = Path(CFG["log"]).read_text(errors="replace").splitlines()
        keys = ("Resolved architecture", "Using MarlinLinearKernel",
                "attention backend out of potential backends", "Model loading took",
                "Available KV cache memory", "GPU KV cache size",
                "Using FlashAttention version", "Initializing a V1 LLM engine")
        log_tail = [mask(l) for l in lines if any(k in l for k in keys)][-9:]
    dump("env.json", {
        "measured": {
            "vllm": vllm.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": mask(gpu),
            "engine": "v1",
        },
        "model": {
            "dir": "<model-dir>",
            "architectures": model_cfg.get("architectures"),
            "quantization_config": model_cfg.get("quantization_config"),
            "file_sha256": {k: v for k, v in files.items()},
        },
        "serve": {
            "command": mask(CFG.get("serve_cmd", "")),
            "upstream": "<upstream>",
            "api": "<api>",
            "max_logprobs": 128,
            "enable_prefix_caching": True,
            "gpu_memory_utilization": 0.90,
            "max_model_len": 8192,
        },
        "log_excerpt": log_tail,
    })


def prompt_receipt(case, state, qid, q):
    parsed = J.parse_question(qid, q)
    body, names = J.question_prompt(parsed, state, 0)
    text = J.chat_wrap(body)
    ids = J.encode(text)
    used = J.label_ids(text, len(names))
    return {
        "case": case,
        "question": parsed,
        "user_message": body,
        "rendered_prompt": text,
        "prompt_token_ids": ids,
        "prompt_sha256": sha256_text(text),
        "prompt_tokens": len(ids),
        "option_names": names,
        "label_token_ids": used,
        "label_symbols": J.SYMBOLS[:len(used)],
    }


def probe_prompts():
    dump("prompts.json", {
        "note": "one rendered prompt per question type, with its label symbols; "
                "sha256 of the rendered text is the fidelity check against docs/jev.md",
        "fixtures": [prompt_receipt(*f) for f in FIXTURES],
    })


def probe_mask():
    """Is the returned logprob vector the exact label-set distribution?

    The same prompt is read twice: once with allowed_token_ids=[labels] and
    logprobs=n_labels, once with no mask and logprobs=20. The masked vector is
    the label-set softmax only if the whitelist is applied before the
    top-k gather; otherwise the payload holds the unmasked ranking and the
    label mass is missing. Both reads are recorded verbatim.
    """
    rows = []
    for case, state, qid, q in FIXTURES:
        pr = prompt_receipt(case, state, qid, q)
        n = len(pr["option_names"])
        allowed = pr["label_token_ids"]
        s1, masked = raw_completion(pr["prompt_token_ids"], n, allowed)
        s2, unmasked = raw_completion(pr["prompt_token_ids"], 20)
        m_lp = label_logprobs_from(masked)
        u_lp = label_logprobs_from(unmasked)
        shared = {i: (m_lp[i], u_lp[i]) for i in allowed if i in u_lp and i in m_lp}
        # The masked read is renormalised over the label set, so absolute
        # values differ from the full-vocab read by one constant. The
        # invariant that distinguishes "whitelist applied before the gather"
        # from "gather before the whitelist" is that pairwise differences
        # survive: lp_i - lp_j must be equal in both reads.
        names = sorted(shared)
        offset_deltas = [
            abs((shared[a][0] - shared[b][0]) - (shared[a][1] - shared[b][1]))
            for a, b in zip(names, names[1:])
        ]
        rows.append({
            "case": case,
            "n_labels": n,
            "allowed_label_ids": allowed,
            "masked_status": s1,
            "unmasked_status": s2,
            "masked_logprobs_payload": masked["choices"][0]["logprobs"],
            "unmasked_logprobs_payload": unmasked["choices"][0]["logprobs"],
            "masked_returned_ids": sorted(m_lp),
            "returned_ids_equal_allowed": sorted(m_lp) == sorted(allowed),
            "returned_count": len(m_lp),
            "count_equals_n_labels": len(m_lp) == n,
            "has_neg_inf": any(math.isinf(v) or math.isnan(v) for v in m_lp.values()),
            "unmasked_top20_ids": sorted(u_lp),
            "labels_visible_unmasked": len(shared),
            "normalisation_offset": (
                statistics.mean(shared[i][1] - shared[i][0] for i in names)
                if names else None),
            "max_abs_offset_delta": max(offset_deltas) if offset_deltas else None,
            "masked_logprobs": {str(k): v for k, v in sorted(m_lp.items())},
            "unmasked_matching": {str(k): [a, b] for k, (a, b) in sorted(shared.items())},
            "sum_exp_masked": sum(math.exp(v) for v in m_lp.values()),
        })
    tag = CFG.get("tag") or "default"
    dump(f"mask-{tag}.json", {
        "method": "same prompt read twice: once with allowed_token_ids=[labels] and "
                  "logprobs=n_labels, once with no mask and logprobs=20; the masked "
                  "vector is the label-set softmax only when the whitelist precedes "
                  "the top-k gather",
        "logprobs_mode": CFG.get("logprobs_mode", "unknown"),
        "rows": rows,
    })


def probe_determinism():
    bodies = []
    for case, state, qid, q in FIXTURES:
        bodies.append({"model": CFG["model"], "state": state,
                       "questions": {qid: q},
                       "options": {"return_logits": True}})

    def read(body):
        s, r = api(body)
        return s, r["answers"] if s == 200 else r

    seq = [read(bodies[1]) for _ in range(3)]
    interleaved = [read(bodies[1]), read(bodies[0]), read(bodies[1])]
    pair = []
    threads = []

    def worker(i):
        pair.append((i, read(bodies[1])))

    for i in range(2):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    def lps(status_payload):
        _, ans = status_payload
        return ans["is_angry"]["label_logprobs"]

    def max_diff(a, b):
        keys = sorted(set(a) & set(b))
        return max(abs(a[k] - b[k]) for k in keys) if keys else None

    concurrent = sorted(pair, key=lambda x: x[0])
    concurrent_payloads = [p for _, p in concurrent]
    dump("determinism.json", {
        "method": "one noul question read repeatedly: 3 back-to-back, 2 around an "
                  "interleaved different prompt, and 2 fired concurrently; "
                  "label_logprobs compared, the sampled token is not part of the answer",
        "sequential_identical": all(lps(seq[0]) == lps(s) for s in seq[1:]),
        "sequential_logprobs": [lps(seq[0]), lps(seq[1]), lps(seq[2])],
        "interleaved_identical": lps(interleaved[0]) == lps(interleaved[2]),
        "interleaved_logprobs": [lps(interleaved[0]), lps(interleaved[2])],
        "concurrent_identical": lps(concurrent_payloads[0]) == lps(concurrent_payloads[1]),
        "concurrent_max_abs_delta": max_diff(lps(concurrent_payloads[0]),
                                             lps(concurrent_payloads[1])),
        "concurrent_logprobs": [lps(concurrent_payloads[0]), lps(concurrent_payloads[1])],
    })


def probe_temperature():
    out = {}
    for case, state, qid, q in FIXTURES:
        rows = []
        for t in (0.5, 1.0, 2.0, 4.0):
            s, r = api({"model": CFG["model"], "state": state,
                        "questions": {qid: q},
                        "options": {"temperature": t, "return_logits": True}})
            a = r["answers"][qid]
            if q["type"] == "noul":
                names = ["yes", "no"]
                api_probs = [a["noul"], 1 - a["noul"]]
            else:
                # choice and score both carry option names in the logits map;
                # a score answer reports its probabilities by index.
                names = list(q["criteria"])
                probs = a["probabilities"]
                api_probs = [probs[n] if q["type"] == "choice" else probs[str(i)]
                             for i, n in enumerate(names)]
            lps = [a["label_logprobs"][n] for n in names]
            client = J.apply_temperature(lps, t)
            rows.append({
                "temperature": t,
                "status": s,
                "api_probabilities": api_probs,
                "client_softmax_of_T1_logprobs": client,
                "max_abs_delta": max(abs(x - y) for x, y in zip(api_probs, client)),
                "confidence": a["confidence"],
            })
        out[case] = {"question_type": q["type"], "rows": rows}
    dump("temperature.json", {
        "method": "for each temperature the API answer is compared with "
                  "softmax(label_logprobs(T=1) / T) computed client-side; the reads "
                  "themselves always run at T=1",
        "cases": out,
    })


def probe_permutations():
    case, state, qid, q = FIXTURES[0]
    parsed = J.parse_question(qid, q)
    n = len(parsed["options"])
    per_rotation = []
    for r in range(n):
        body, names = J.question_prompt(parsed, state, r)
        text = J.chat_wrap(body)
        used = J.label_ids(text, n)
        s, resp = raw_completion(J.encode(text), n, used)
        lps = [label_logprobs_from(resp)[i] for i in used]
        probs = J.apply_temperature(lps, 1.0)
        per_rotation.append({"rotation": r, "option_order": names,
                             "probabilities": probs,
                             "label_logprobs": lps})
    base = dict(zip(per_rotation[0]["option_order"], per_rotation[0]["probabilities"]))
    for row in per_rotation:
        d = dict(zip(row["option_order"], row["probabilities"]))
        row["l1_vs_rotation0"] = sum(abs(base[k] - d[k]) for k in base)
        row["argmax_vs_rotation0"] = (
            max(d, key=d.get) == max(base, key=base.get))
    s, r = api({"model": CFG["model"], "state": state, "questions": {qid: q},
                "options": {"permutations": n, "return_logits": True}})
    averaged = r["answers"]["department"]
    mean_of_rotations = {
        name: sum(dict(zip(row["option_order"], row["probabilities"]))[name]
                  for row in per_rotation) / n
        for name in base
    }
    dump("permutations.json", {
        "method": "one 3-option choice question read once per rotation of the "
                  "option order; then the endpoint's permutations=3 answer, which "
                  "averages the same rotations",
        "case": case,
        "per_rotation": per_rotation,
        "max_l1_vs_rotation0": max(row["l1_vs_rotation0"] for row in per_rotation),
        "argmax_stable": all(row["argmax_vs_rotation0"] for row in per_rotation),
        "api_permutations_3": averaged,
        "mean_of_measured_rotations": mean_of_rotations,
        "api_vs_mean_max_abs_delta": max(
            abs(averaged["probabilities"][k] - mean_of_rotations[k]) for k in base),
    })


def probe_latency():
    case, state, qid, q = FIXTURES[0]
    body = {"model": CFG["model"], "state": state, "questions": {qid: q}}
    warm, busted = [], []
    for i in range(30):
        t0 = time.perf_counter()
        s, r = api(body)
        dt = (time.perf_counter() - t0) * 1e3
        warm.append({"i": i, "status": s, "ms": dt})
        if i == 2:
            probe_gpu = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,clocks.sm,power.draw,temperature.gpu",
                 "--format=csv,noheader"], capture_output=True, text=True).stdout
    for i in range(30):
        post(CFG["upstream"] + "/reset_prefix_cache", {})
        t0 = time.perf_counter()
        s, r = api(body)
        dt = (time.perf_counter() - t0) * 1e3
        busted.append({"i": i, "status": s, "ms": dt})
    dump("latency.jsonl", {"warm": warm, "busted": busted})
    def stat(rows):
        xs = sorted(r["ms"] for r in rows)
        return {"n": len(xs), "p50_ms": statistics.median(xs),
                "p95_ms": xs[int(0.95 * (len(xs) - 1))],
                "min_ms": xs[0], "max_ms": xs[-1]}
    dump("latency.json", {
        "method": "30 reads of one 3-option question with a warm prefix cache, then "
                  "30 after POST /reset_prefix_cache before each read; client-side "
                  "end-to-end wall time, one request at a time, GPU shared with no "
                  "other job; c=1",
        "warm": stat(warm),
        "busted": stat(busted),
        "gpu_mid_run": mask(probe_gpu.strip()) if probe_gpu else None,
    })


def metrics_for(rows, temperature):
    """Descriptive metrics at one temperature, plus the NLL used for fitting."""
    correct, nll, brier = 0, 0.0, 0.0
    bins = [{"conf": 0.0, "correct": 0, "n": 0} for _ in range(10)]
    for row in rows:
        names = row["option_names"]
        probs = J.apply_temperature([row["label_logprobs"][n] for n in names], temperature)
        p = dict(zip(names, probs))
        expected = row["expected"]
        pred = max(p, key=p.get)
        correct += pred == expected
        nll -= math.log(max(p.get(expected, 1e-12), 1e-12))
        brier += sum((p[k] - (1.0 if k == expected else 0.0)) ** 2 for k in p)
        c = p[pred]
        b = min(int(c * 10), 9)
        bins[b]["n"] += 1
        bins[b]["correct"] += pred == expected
        bins[b]["conf"] += c
    n = len(rows)
    acc = correct / n
    # Wilson 95% interval
    z = 1.96
    denom = 1 + z * z / n
    centre = (acc + z * z / (2 * n)) / denom
    half = z * math.sqrt(acc * (1 - acc) / n + z * z / (4 * n * n)) / denom
    ece = sum(
        b["n"] / n * abs((b["conf"] / b["n"] if b["n"] else 0) - (b["correct"] / b["n"] if b["n"] else 0))
        for b in bins)
    return {
        "temperature": temperature,
        "n": n,
        "accuracy": acc,
        "accuracy_wilson95": [max(0.0, centre - half), min(1.0, centre + half)],
        "brier_multiclass": brier / n,
        "nll_true_class": nll / n,
        "ece_10bin": ece,
        "reliability": bins,
    }


def fit_temperature(rows):
    """NLL-minimizing temperature over a log grid; the ECE that comes with it."""
    grid = [0.2 * (1.12 ** i) for i in range(45)]
    best, best_nll = 1.0, None
    curve = []
    for t in grid:
        m = metrics_for(rows, t)
        curve.append({"temperature": t, "nll_true_class": m["nll_true_class"],
                      "ece_10bin": m["ece_10bin"]})
        if best_nll is None or m["nll_true_class"] < best_nll:
            best, best_nll = t, m["nll_true_class"]
    return best, curve


def loo_temperature(rows):
    """Leave-one-out ECE: the temperature for each held-out example is fitted on
    the other examples only, so the reported ECE is not in-sample."""
    preds = []
    for i, row in enumerate(rows):
        fit_rows = rows[:i] + rows[i + 1:]
        t, _ = fit_temperature(fit_rows)
        prob_vectors = []
        names = row["option_names"]
        probs = J.apply_temperature([row["label_logprobs"][n] for n in names], t)
        p = dict(zip(names, probs))
        pred = max(p, key=p.get)
        preds.append({"expected": row["expected"], "predicted": pred,
                      "confidence": p[pred], "temperature": t,
                      "correct": pred == row["expected"]})
    n = len(preds)
    acc = sum(p["correct"] for p in preds) / n
    ece = 0.0
    for b in range(10):
        in_bin = [p for p in preds if min(int(p["confidence"] * 10), 9) == b]
        if in_bin:
            ece += len(in_bin) / n * abs(
                statistics.mean(p["confidence"] for p in in_bin)
                - statistics.mean(p["correct"] for p in in_bin))
    return {"accuracy": acc, "ece_10bin": ece,
            "temperatures": [p["temperature"] for p in preds],
            "per_example": preds}


def probe_labelled():
    rows = []
    for item in labelled_set():
        q = item["question"]
        s, r = api({"model": CFG["model"], "state": item["state"],
                    "questions": {item["id"]: q},
                    "options": {"return_logits": True}})
        parsed = J.parse_question(item["id"], q)
        names = [n for n, _ in parsed["options"]]
        if s != 200:
            rows.append({**item, "status": s, "error": r})
            continue
        a = r["answers"][item["id"]]
        rows.append({**item, "status": s, "question_type": q["type"],
                     "option_names": names,
                     "label_logprobs": {n: a["label_logprobs"][n] for n in names},
                     "confidence": a["confidence"],
                     "answer": a})
    for row in rows:
        if isinstance(row.get("label_logprobs"), list):
            row["label_logprobs"] = dict(zip(row["option_names"], row["label_logprobs"]))
    scored = [r for r in rows if r["status"] == 200]
    (Path(CFG["out"]) / "labeled.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=False) + "\n" for r in rows))
    t_fit, curve = fit_temperature(scored)
    by_type = {}
    for t in ("choice", "noul", "score"):
        subset = [r for r in scored if r["question_type"] == t]
        if subset:
            by_type[t] = metrics_for(subset, 1.0)
    dump("metrics.json", {
        "method": "author-labelled small set, one read per example at T=1 with "
                  "return_logits; temperature fitted by NLL on the same set, and "
                  "reported again under leave-one-out fitting",
        "n_examples": len(rows),
        "n_scored": len(scored),
        "at_T1": metrics_for(scored, 1.0),
        "fitted_temperature": t_fit,
        "at_fitted_T": metrics_for(scored, t_fit),
        "leave_one_out": loo_temperature(scored),
        "by_question_type_at_T1": by_type,
        "nll_curve": curve,
    })


def probe_negatives():
    cases = []
    base = {"model": CFG["model"], "state": STATE_BILLING,
            "questions": {"x": Q_CHOICE}}

    def add(name, body, expect_status):
        s, r = api(body)
        cases.append({"case": name, "status": s, "expected_status": expect_status,
                      "body": body, "response": r})

    add("unknown_type", {**base, "questions": {"x": {"type": "ranking",
                                                     "instructions": "pick"}}}, 422)
    add("missing_instructions", {**base, "questions": {"x": {"type": "noul"}}}, 422)
    add("empty_choice_criteria", {**base, "questions": {"x":
        {"type": "choice", "instructions": "which", "criteria": {}}}}, 422)
    add("score_single_level", {**base, "questions": {"x":
        {"type": "score", "instructions": "how", "criteria": ["only"]}}}, 422)
    add("images_unsupported", {**base, "images": ["data:image/png;base64,AAAA"]}, 422)
    add("bad_temperature", {**base, "options": {"temperature": 0}}, 422)
    add("bad_permutations", {**base, "options": {"permutations": 0}}, 422)
    add("too_many_options", {**base, "questions": {"x":
        {"type": "choice", "instructions": "which",
         "criteria": {f"opt{i}": None for i in range(70)}}}}, 422)
    s, r = post(CFG["upstream"] + "/v1/completions",
                {"model": CFG["model"], "prompt": [1, 2, 3], "max_tokens": 1,
                 "logprobs": 129})
    cases.append({"case": "upstream_logprobs_over_cap", "status": s,
                  "expected_status": 400, "response": r})
    dump("negatives.json", {"cases": cases,
                            "all_as_expected": all(c["status"] == c["expected_status"]
                                                   for c in cases)})


def probe_crosscheck():
    """Engine-vs-API check: the same five prompts read in-process with vLLM's LLM
    API, on a second card, compared with the HTTP path."""
    os.environ["CUDA_VISIBLE_DEVICES"] = CFG.get("crosscheck_gpu", "1")
    from vllm import LLM, SamplingParams

    llm = LLM(model=CFG["model_dir"], max_model_len=8192,
              gpu_memory_utilization=0.60, enable_prefix_caching=False,
              # The checkpoint ships top_k=20 / top_p=0.95 and both the server
              # and LLM() adopt them; "vllm" uses vLLM's own defaults (no
              # nucleus) so this comparison isolates transport rather than
              # re-measuring the nucleus effect that nucleus.json documents.
              generation_config="vllm",
              logprobs_mode=CFG.get("logprobs_mode", "processed_logprobs"))
    prompts, params = [], []
    fixtures = FIXTURES + [
        ("choice-billing-rot2", STATE_BILLING, "department", Q_CHOICE),
        ("noul-angry-2", STATE_ANGRY, "is_angry", Q_NOUL),
    ]
    for case, state, qid, q in fixtures:
        pr = prompt_receipt(case, state, qid, q)
        n = len(pr["option_names"])
        prompts.append({"prompt_token_ids": pr["prompt_token_ids"],
                        "allowed": pr["label_token_ids"], "n": n, "case": case})
        params.append(SamplingParams(max_tokens=1, temperature=1.0, logprobs=n,
                                     allowed_token_ids=pr["label_token_ids"],
                                     # The HTTP read neutralises the checkpoint's
                                     # generation_config default; do the same
                                     # here so the comparison isolates transport
                                     # rather than re-measuring the nucleus effect.
                                     top_p=1.0, top_k=-1))
    outs = llm.generate([p["prompt_token_ids"] for p in prompts], params)
    rows = []
    for fixture, p, out in zip(fixtures, prompts, outs):
        case, state, qid, q = fixture
        pr = prompt_receipt(case, state, qid, q)
        s_http, http = raw_completion(pr["prompt_token_ids"], p["n"], p["allowed"])
        http_lp = label_logprobs_from(http)
        got = out.outputs[0].logprobs[0]
        local_lp = {}
        for token_id, lp in got.items():
            if isinstance(token_id, int):
                local_lp[token_id] = lp.logprob
            else:
                local_lp[int(str(token_id).split(":", 1)[-1])] = lp.logprob
        shared = sorted(set(http_lp) & set(local_lp))
        rows.append({
            "case": case,
            "http_status": s_http,
            "shared_label_ids": shared,
            "local_logprobs": {str(k): local_lp[k] for k in shared},
            "http_logprobs": {str(k): http_lp[k] for k in shared},
            "max_abs_delta": max(abs(http_lp[k] - local_lp[k]) for k in shared) if shared else None,
            "local_only": sorted(set(local_lp) - set(http_lp)),
            "http_only": sorted(set(http_lp) - set(local_lp)),
        })
    dump("crosscheck.json", {
        "method": "the same prompts read twice: in-process vLLM LLM.generate with "
                  "SamplingParams(logprobs=n, allowed_token_ids=labels, "
                  "top_p=1.0, top_k=-1) and generation_config='vllm' on a second "
                  "card, and over the HTTP completions path; label logprobs compared",
        "rows": rows,
    })


def probe_nucleus():
    """What the model's generation_config defaults do to a read.

    vLLM adopts the checkpoint's generation_config (top_k=20, top_p=0.95,
    temperature=1.0) as the server's default sampling parameters, and in
    processed_logprobs mode the returned logprobs are post-filter. Reading a
    label set without neutralising those defaults renormalises the label
    distribution over a nucleus: labels below the cut are dropped and the
    surviving ones are inflated.
    """
    rows = []
    for case, state, qid, q in FIXTURES + [("choice-longshot", STATE_BILLING,
                                            "department", Q_LONGSHOT)]:
        pr = prompt_receipt(case, state, qid, q)
        n = len(pr["option_names"])
        allowed = pr["label_token_ids"]
        s_neutral, neutral = raw_completion(pr["prompt_token_ids"], n, allowed)
        s_default, default = raw_completion(pr["prompt_token_ids"], n, allowed,
                                            neutral=False)
        nl = label_logprobs_from(neutral)
        dl = label_logprobs_from(default)
        pn = J.apply_temperature([nl[i] for i in allowed], 1.0)
        pd_raw = J.apply_temperature([dl[i] for i in allowed], 1.0) if all(
            i in dl for i in allowed) else None
        rows.append({
            "case": case,
            "question_type": q["type"],
            "neutralised": {
                "status": s_neutral,
                "label_logprobs": {str(k): v for k, v in nl.items()},
                "probabilities": pn,
            },
            "generation_config_default": {
                "status": s_default,
                "returned_ids": sorted(dl),
                "dropped_labels": [i for i in allowed if i not in dl],
                "label_logprobs": {str(k): v for k, v in dl.items()},
                "probabilities_over_returned": pd_raw,
            },
            "max_abs_probability_delta": (
                max(abs(a - b) for a, b in zip(pn, pd_raw)) if pd_raw else None),
        })
    dump("nucleus.json", {
        "method": "the same label-masked read twice per question: once with "
                  "top_p=1.0 and top_k=-1 set explicitly, once letting the server "
                  "apply the checkpoint's generation_config defaults "
                  "(temperature 1.0, top_k 20, top_p 0.95)",
        "rows": rows,
    })


PROBES = {
    "env": probe_env,
    "prompts": probe_prompts,
    "mask": probe_mask,
    "nucleus": probe_nucleus,
    "determinism": probe_determinism,
    "temperature": probe_temperature,
    "permutations": probe_permutations,
    "latency": probe_latency,
    "labeled": probe_labelled,
    "negatives": probe_negatives,
    "crosscheck": probe_crosscheck,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("probe", nargs="+", choices=list(PROBES) + ["all"])
    ap.add_argument("--out", default=str(HERE.parent / "receipts"))
    ap.add_argument("--log", default=os.environ.get("JEV_LOG", ""))
    ap.add_argument("--crosscheck-gpu", default=os.environ.get("JEV_CROSSCHECK_GPU", "1"))
    args = ap.parse_args()

    CFG.update({
        "upstream": os.environ.get("JEV_UPSTREAM", "http://127.0.0.1:18030"),
        "api": os.environ.get("JEV_API", "http://127.0.0.1:18031"),
        "model": os.environ.get("JEV_MODEL", "qwen3.8-27b-jev"),
        "model_dir": os.environ.get("MODEL_DIR", "/models/Qwen3.8-27B-GPTQ-4bit"),
        "serve_cmd": os.environ.get("JEV_SERVE_CMD", ""),
        "out": args.out,
        "log": args.log,
        "crosscheck_gpu": args.crosscheck_gpu,
        "tag": os.environ.get("JEV_TAG", "default"),
        "logprobs_mode": os.environ.get("JEV_LOGPROBS_MODE", "raw_logprobs"),
    })
    Path(args.out).mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    J.ARGS = argparse.Namespace(upstream=CFG["upstream"], model=CFG["model"],
                                tokenizer=CFG["model_dir"], state=None)
    J.TOK = AutoTokenizer.from_pretrained(CFG["model_dir"])

    names = list(PROBES) if "all" in args.probe else args.probe
    for name in names:
        print(f"== {name}", flush=True)
        PROBES[name]()


if __name__ == "__main__":
    main()
