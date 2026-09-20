#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Jev-style calibrated classification in front of a vLLM OpenAI endpoint.

POST /v1/systemone takes Jev's request shape (docs/jev.md):

  {"model": "...", "state": "...",
   "questions": {
     "department": {"type": "choice", "instructions": "...",
                    "criteria": {"billing": "...", "technical": null}},
     "is_angry":   {"type": "noul", "instructions": "..."},
     "urgency":    {"type": "score", "instructions": "...",
                    "criteria": ["Not urgent", "Today"]}},
   "options": {"temperature": 1.0, "permutations": 1, "return_logits": false}}

Each question becomes one prompt in Jev's format, wrapped in the model's own
chat template (assistant generation prompt, thinking disabled). The prompt is
evaluated once (max_tokens=1) with every non-label token masked through
allowed_token_ids, so the returned logprobs are exactly the conditional
distribution over the labels at temperature 1. A requested temperature is
applied as softmax(logprobs / T) after the read, so one read serves any
calibration and the same prompt always yields the same probabilities: nothing
about the answer depends on sampling.

Question types: choice, noul (yes/no), score (ordered levels, lowest first).
Answers follow Jev's shapes; confidence is 1 - normalized entropy. Images are
not supported by this text stack (422 images_not_supported).

  python jev_server.py --upstream http://127.0.0.1:18020 \
      --tokenizer /models/Qwen3.8-27B-GPTQ-4bit --port 18021
"""

import argparse
import json
import math
import string
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from transformers import AutoTokenizer

ARGS = None
TOK = None
POOL = None

SYMBOLS = string.ascii_uppercase + string.ascii_lowercase + string.digits


class SchemaError(ValueError):
    def __init__(self, code, message, field=None):
        super().__init__(message)
        self.code = code
        self.field = field


# ----------------------------------------------------------------------------
# Question schema
# ----------------------------------------------------------------------------


def parse_question(qid, q):
    if not isinstance(q, dict):
        raise SchemaError("invalid_question", f"questions.{qid} must be an object",
                          f"questions.{qid}")
    qtype = q.get("type")
    if qtype not in ("choice", "noul", "score"):
        raise SchemaError("invalid_type", f"questions.{qid}.type must be choice, noul or score",
                          f"questions.{qid}.type")
    instructions = q.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        raise SchemaError("invalid_instructions", f"questions.{qid}.instructions must be text",
                          f"questions.{qid}.instructions")
    out = {"id": qid, "type": qtype, "instructions": instructions.strip()}

    if qtype == "choice":
        criteria = q.get("criteria")
        if not isinstance(criteria, dict) or not criteria:
            raise SchemaError("invalid_criteria",
                              f"questions.{qid}.criteria must map option names to text or null",
                              f"questions.{qid}.criteria")
        out["options"] = [
            (name, desc if isinstance(desc, str) else None)
            for name, desc in criteria.items()
        ]
    elif qtype == "noul":
        criteria = q.get("criteria") or {}
        if not isinstance(criteria, dict):
            raise SchemaError("invalid_criteria",
                              f"questions.{qid}.criteria must map true/false to text",
                              f"questions.{qid}.criteria")
        out["options"] = [
            ("yes", criteria.get("true")),
            ("no", criteria.get("false")),
        ]
    else:
        criteria = q.get("criteria")
        if not isinstance(criteria, list) or len(criteria) < 2 or not all(
            isinstance(lvl, str) for lvl in criteria
        ):
            raise SchemaError("invalid_criteria",
                              f"questions.{qid}.criteria must list level names lowest to highest",
                              f"questions.{qid}.criteria")
        out["options"] = [(lvl, None) for lvl in criteria]

    if len(out["options"]) > len(SYMBOLS):
        raise SchemaError("too_many_options",
                          f"questions.{qid} has {len(out['options'])} options, at most "
                          f"{len(SYMBOLS)} are supported", f"questions.{qid}")
    return out


def parse_body(body):
    if not isinstance(body, dict):
        raise SchemaError("invalid_body", "request body must be a JSON object")
    if body.get("images"):
        raise SchemaError("images_not_supported", "this server has no multimodal projector")
    state = body.get("state")
    if state is None:
        raise SchemaError("invalid_state", "state is required", "state")
    if isinstance(state, (dict, list)):
        state = json.dumps(state, ensure_ascii=False)
    if not isinstance(state, str):
        raise SchemaError("invalid_state", "state must be text or a JSON object", "state")
    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise SchemaError("invalid_questions", "questions must be a non-empty map",
                          "questions")
    parsed = [parse_question(qid, q) for qid, q in questions.items()]
    opts = body.get("options") or {}
    if not isinstance(opts, dict):
        raise SchemaError("invalid_options", "options must be an object", "options")
    temperature = opts.get("temperature", 1.0)
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) \
            or temperature <= 0:
        raise SchemaError("invalid_temperature", "options.temperature must be > 0",
                          "options.temperature")
    permutations = opts.get("permutations", 1)
    if not isinstance(permutations, int) or isinstance(permutations, bool) or permutations < 1:
        raise SchemaError("invalid_permutations", "options.permutations must be a positive int",
                          "options.permutations")
    return {
        "state": state,
        "questions": parsed,
        "temperature": float(temperature),
        "permutations": permutations,
        "return_logits": bool(opts.get("return_logits", False)),
    }


# ----------------------------------------------------------------------------
# Prompt format (docs/jev.md) and label symbols
# ----------------------------------------------------------------------------


def question_prompt(q, state, rotation):
    opts = q["options"]
    if rotation:
        opts = opts[rotation:] + opts[:rotation]
    body = (
        f"Context:\n{state}\n"
        "\n"
        "Answer the question with only the label of the best option "
        "(the character before the colon), nothing else.\n"
        f"Question: {q['instructions']}\n"
        "Options:\n"
        + "\n".join(f"{name}: {desc}" if desc else name for name, desc in opts)
    )
    return body, [name for name, _ in opts]


def chat_wrap(user_text):
    return TOK.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def encode(text):
    return TOK.encode(text, add_special_tokens=False)


_label_cache = {}


def label_ids(prompt_text, n_needed):
    """Single-token label symbols right after this generation prompt, in order."""
    cached = _label_cache.get(prompt_text)
    if cached is None:
        base = encode(prompt_text)
        cached = []
        for sym in SYMBOLS:
            sym_ids = encode(sym)
            if len(sym_ids) == 1 and encode(prompt_text + sym) == base + sym_ids:
                cached.append(sym_ids[0])
            if len(cached) >= 62:
                break
        _label_cache[prompt_text] = cached
    return cached[:n_needed]


# ----------------------------------------------------------------------------
# Reads
# ----------------------------------------------------------------------------


def upstream(path, body, timeout=600):
    req = urllib.request.Request(
        ARGS.upstream.rstrip("/") + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def token_key_id(token):
    """A logprob key is either a token id string ("token_id:32" / "token:32",
    depending on the vLLM build) or a plain token piece."""
    if isinstance(token, str) and ":" in token:
        try:
            return int(token.rsplit(":", 1)[1])
        except ValueError:
            pass
    return TOK.convert_tokens_to_ids(token)


def read_labels(prompt_text, n_labels):
    """One prompt evaluation. Returns (ids, logprobs), aligned lists: the
    logprob of each label token at the position after the generation prompt.
    Reads run at temperature 1, so these are the raw normalized label logits;
    the requested temperature is applied afterwards, per request."""
    used = label_ids(prompt_text, n_labels)
    if len(used) < n_labels:
        raise SchemaError("too_many_options",
                          f"model offers {len(used)} single-token labels after this "
                          f"prompt, {n_labels} needed")
    out = upstream("/v1/completions", {
        "model": ARGS.model,
        "prompt": encode(prompt_text),
        "max_tokens": 1,
        # T=1 is the identity; the sampled token is discarded, only the
        # logprobs of the first generated position are read.
        "temperature": 1.0,
        # The model's generation_config ships top_k=20 / top_p=0.95 and vLLM
        # adopts those as server defaults, which would renormalize the label
        # distribution over a nucleus; a calibration read wants none of it.
        "top_p": 1.0,
        "top_k": -1,
        "logprobs": n_labels,
        "return_tokens_as_token_ids": True,
        "allowed_token_ids": used,
    })
    top = out["choices"][0]["logprobs"]["top_logprobs"][0]
    by_id = {}
    for token, lp in top.items():
        by_id[token_key_id(token)] = lp
    missing = [i for i in used if i not in by_id]
    if missing:
        raise RuntimeError(
            f"upstream returned no logprobs for label ids {missing}; "
            "allowed_token_ids masking is not honored or logprobs < labels")
    usage = out.get("usage", {})
    return used, [by_id[i] for i in used], usage.get("prompt_tokens")


# ----------------------------------------------------------------------------
# Probability math
# ----------------------------------------------------------------------------


def apply_temperature(label_logprobs, temperature):
    mx = max(label_logprobs)
    scaled = [(lp - mx) / temperature for lp in label_logprobs]
    ex = [math.exp(v) for v in scaled]
    total = sum(ex)
    return [v / total for v in ex]


def confidence_of(probs):
    h = -sum(p * math.log(p) for p in probs if p > 0)
    return 1.0 - h / math.log(len(probs))


def answer_for(q, names, probs, lps, return_logits):
    top = max(range(len(probs)), key=lambda i: probs[i])
    a = {"type": q["type"], "confidence": confidence_of(probs)}
    if return_logits:
        a["label_logprobs"] = {n: lp for n, lp in zip(names, lps)}
    if q["type"] == "noul":
        a["noul"] = probs[0]
    elif q["type"] == "choice":
        a["choice"] = names[top]
        a["probabilities"] = {n: p for n, p in zip(names, probs)}
    else:
        a["score"] = sum(i * p for i, p in enumerate(probs))
        a["legend"] = {str(i): n for i, n in enumerate(names)}
        a["probabilities"] = {str(i): p for i, p in enumerate(probs)}
    return a


def decide_one(q, state, rotation, temperature):
    """One read at one rotation. Returns the option names in that rotation's
    order, their probabilities, and their temperature-1 logprobs."""
    body, names = question_prompt(q, state, rotation)
    lps = read_labels(chat_wrap(body), len(names))
    _, probs, _ = lps, apply_temperature(lps[1], temperature), lps[2]
    return names, probs, lps[1]


def decide(parsed):
    state = parsed["state"]
    questions = parsed["questions"]
    tasks = [
        (qi, q, r)
        for qi, q in enumerate(questions)
        for r in range(min(parsed["permutations"], len(q["options"])))
    ]
    results = list(POOL.map(lambda t: decide_one(t[1], state, t[2], parsed["temperature"]),
                            tasks))
    answers = {}
    total_prompt_tokens = 0
    for qi, q in enumerate(questions):
        n = min(parsed["permutations"], len(q["options"]))
        runs = results[qi * n:(qi + 1) * n]
        original = [name for name, _ in q["options"]]
        acc = [0.0] * len(original)
        lps0, names0 = None, None
        for r, (names, probs, raw_lps) in enumerate(runs):
            for pos, name in enumerate(names):
                acc[original.index(name)] += probs[pos]
            if r == 0:
                names0, lps0 = names, raw_lps
        probs_mean = [v / len(runs) for v in acc]
        # For the logits passthrough report the rotation-0 read, whose label
        # order matches the request's option order.
        rot0 = question_prompt(q, state, 0)[1]
        answers[q["id"]] = answer_for(q, rot0, probs_mean,
                                      lps0 if names0 == rot0 else None,
                                      parsed["return_logits"] and names0 == rot0)
    return answers


# ----------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _json(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/health", "/props"):
            self._json(200, {"service": "jev", "upstream": ARGS.upstream,
                             "model": ARGS.model})
        else:
            self._json(404, {"error": {"code": "not_found"}})

    def do_POST(self):
        if self.path != "/v1/systemone":
            self._json(404, {"error": {"code": "not_found"}})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            parsed = parse_body(body)
            answers = decide(parsed)
            self._json(200, {"model": ARGS.model, "answers": answers})
        except SchemaError as e:
            err = {"error": {"code": e.code, "message": str(e)}}
            if e.field:
                err["error"]["field"] = e.field
            self._json(422, err)
        except Exception as e:  # noqa: BLE001 - report upstream failures as 502
            self._json(502, {"error": {"code": "upstream_error", "message": str(e)}})


def main():
    global ARGS, TOK, POOL
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--model", default="qwen3.8-27b", help="served model name upstream")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18021)
    ap.add_argument("--workers", type=int, default=8)
    ARGS = ap.parse_args()
    TOK = AutoTokenizer.from_pretrained(ARGS.tokenizer)
    POOL = ThreadPoolExecutor(max_workers=ARGS.workers)
    ThreadingHTTPServer((ARGS.host, ARGS.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
