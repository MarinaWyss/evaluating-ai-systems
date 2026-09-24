"""LLM judges, and the tools to check whether you can trust them (Module 5).

    build_judge_prompt    the six-part judge prompt, built from your rubric
    run_judge             judge one trace, returning its reasoning and verdict
    judge_all             judge a whole DataFrame of traces (in parallel)
    tpr_tnr               compare the judge's verdicts to human labels
    rogan_gladen          correct an observed pass rate for an imperfect judge
    corrected_pass_rate   the same, with a bootstrap interval (the idea behind the judgy library)
    pairwise_judge        pick the better of two answers
    swap_test             run pairwise judging in both orders and count flips
    groundedness          split an answer into claims and check each against the retrieved text
"""

from __future__ import annotations

import json
import random
import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from . import llm

PASS, FAIL = "PASS", "FAIL"

APP_DESCRIPTION = (
    "a customer support chatbot for BeefCake Fitness, a home-gym company. It answers questions about four "
    "products: the BeefCake Row (a smart rowing machine with a screen), the BeefCake Bell (an app-connected "
    "adjustable kettlebell you charge over USB-C), the BeefCake Pulse (a heart-rate chest strap with a coin "
    "battery), and the BeefCake Coach subscription app."
)


# ---------------------------------------------------------------------------
# Building and running a judge
# ---------------------------------------------------------------------------

def build_judge_prompt(failure_mode: str, definition: str, pass_criteria: str, fail_criteria: str,
                       examples: list[dict], app_description: str = APP_DESCRIPTION) -> str:
    """The six parts: (1) you're a judge, (2) the app, (3) one failure mode,
    (4) PASS/FAIL criteria, (5) reasoning before the verdict, (6) examples of both.

    examples: [{"user_query", "ai_response", "judgment", "reasoning"}, ...]
    """
    shots = "\n\n".join(
        f"<example>\nUSER: {e['user_query']}\nBOT: {e['ai_response']}\n"
        f'OUTPUT: {json.dumps({"reasoning": e.get("reasoning", ""), "judgment": e["judgment"]})}\n</example>'
        for e in examples
    )
    return f"""You are an expert evaluator. Your job is to judge one conversation from {app_description}

You check exactly one failure mode: {failure_mode}.
Definition: {definition}

Return PASS if: {pass_criteria}
Return FAIL if: {fail_criteria}

Ignore every other kind of problem. Only judge this one failure mode.

First write your reasoning, then give your judgment. Reply with JSON only, in this exact format:
{{"reasoning": "<one to three sentences>", "judgment": "PASS" or "FAIL"}}

Examples:

{shots}"""


# Our judge for "Assumes device" (M5.5). It's built from the worked example in rubric/rubric_template.md.
ASSUMES_DEVICE_JUDGE = build_judge_prompt(
    failure_mode="Assumes device",
    definition="The bot gives device-specific advice before the customer has made clear which BeefCake product they mean.",
    pass_criteria=(
        "the customer names the product (even with a typo), OR only one BeefCake product fits what they described "
        "(only the Row has a screen and resistance levels; only the Bell is charged over USB-C; the Pulse is the only "
        "heart-rate product), OR the bot asks which product before giving device-specific steps, OR the advice applies "
        "to every product."
    ),
    fail_criteria="the customer's message fits more than one product and the bot picks one without asking.",
    examples=[
        {"user_query": "it won't turn on", "ai_response": "Which BeefCake product is it: the Row, the Bell, or the Pulse?",
         "judgment": PASS, "reasoning": "The message fits several products, and the bot asked which one."},
        {"user_query": "my beefcak bel wont click to 12kg",
         "ai_response": "Lift the Bell out of the cradle and set it back in so it sits flat, then try the dial again.",
         "judgment": PASS, "reasoning": "Despite the typo, the customer named the Bell."},
        {"user_query": "how do I reset it?",
         "ai_response": "Go to Settings > System > Factory reset on your Row's screen.",
         "judgment": FAIL, "reasoning": "'It' could be any product, and the bot assumed the Row without asking."},
        {"user_query": "the battery dies really fast",
         "ai_response": "Replace the CR2032 coin battery in your Pulse.",
         "judgment": FAIL, "reasoning": "Both the Bell and the Pulse have batteries, and the bot assumed the Pulse."},
    ],
)


def _parse_json_object(text: str) -> dict | None:
    """The first JSON object in a reply, ignoring any text before or after it."""
    decoder = json.JSONDecoder()
    for brace in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text, brace.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _parse_verdict(text: str) -> dict:
    data = _parse_json_object(text)
    if data:
        judgment = str(data.get("judgment", "")).strip().upper()
        if judgment in (PASS, FAIL):
            return {"reasoning": data.get("reasoning", ""), "judgment": judgment}
    labeled = re.search(r"(JUDG(E)?MENT|VERDICT)\W+(PASS|FAIL)\b", text.upper())
    if labeled:
        return {"reasoning": text.strip(), "judgment": labeled.group(3)}
    found = set(re.findall(r"\b(PASS|FAIL)\b", text.upper()))
    return {"reasoning": text.strip(), "judgment": found.pop() if len(found) == 1 else "ERROR"}


def run_judge(judge_prompt: str, user_query: str, ai_response: str, model: str | None = None) -> dict:
    """Judge one trace. The judge only sees the customer's message and the bot's reply (M5.6)."""
    messages = [
        {"role": "system", "content": judge_prompt},
        {"role": "user", "content": f"USER: {user_query}\nBOT: {ai_response}"},
    ]
    return _parse_verdict(llm.complete(messages, model=model or llm.get_model("judge")))


def judge_all(df: pd.DataFrame, judge_prompt: str, model: str | None = None, max_workers: int = 8) -> pd.DataFrame:
    """Run the judge on every row (columns "User Query" and "AI Response"). Adds Judge and Judge reasoning."""
    rows = list(df[["User Query", "AI Response"]].itertuples(index=False))

    def one(r):
        try:
            return run_judge(judge_prompt, r[0], r[1], model)
        except Exception as e:  # one failed call (a rate limit, say) shouldn't lose the whole run
            return {"reasoning": f"Judge call failed: {e}", "judgment": "ERROR"}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(one, rows))
    errors = sum(r["judgment"] == "ERROR" for r in results)
    if errors:
        print(f"Warning: {errors} of {len(results)} judge calls failed or gave no clear PASS/FAIL, and tpr_tnr counts "
              "them as wrong. Check the Judge reasoning column. For rate limits, rerun with max_workers=2.")
    out = df.copy()
    out["Judge"] = [r["judgment"] for r in results]
    out["Judge reasoning"] = [r["reasoning"] for r in results]
    return out


# ---------------------------------------------------------------------------
# Can you trust it?
# ---------------------------------------------------------------------------

def tpr_tnr(human: pd.Series, judge: pd.Series) -> dict:
    """PASS is the positive class (Wengrow's convention).
    TPR: of the traces humans passed, the share the judge passed.
    TNR: of the traces humans failed, the share the judge failed.
    A verdict that isn't PASS or FAIL (a failed call) counts as wrong, and shows up as "judge errors".
    """
    human = human.astype(str).str.upper().reset_index(drop=True)
    judge = judge.astype(str).str.upper().reset_index(drop=True)
    p, n = human == PASS, human == FAIL
    tpr = float((judge[p] == PASS).mean()) if p.any() else float("nan")
    tnr = float((judge[n] == FAIL).mean()) if n.any() else float("nan")
    rates = {"TPR": round(tpr, 3), "TNR": round(tnr, 3), "human PASS": int(p.sum()), "human FAIL": int(n.sum())}
    errors = int((~judge.isin([PASS, FAIL]) & (p | n)).sum())
    if errors:
        rates["judge errors"] = errors
    return rates


def rogan_gladen(observed_pass_rate: float, tpr: float, tnr: float) -> float:
    """The judge says observed_pass_rate. Estimate the true pass rate. Needs TPR + TNR > 1."""
    if not (tpr + tnr > 1):  # also catches NaN, e.g. a sample with no FAIL labels
        raise ValueError("TPR + TNR must be above 1 (the judge must be better than chance).")
    return min(1.0, max(0.0, (observed_pass_rate + tnr - 1) / (tpr + tnr - 1)))


def corrected_pass_rate(test_human: pd.Series, test_judge: pd.Series, production_judge: pd.Series,
                        n_boot: int = 2000, seed: int = 0) -> dict:
    """Correct the judge's production pass rate, with a 95% bootstrap interval.

    Resamples the labeled test set (uncertainty in TPR/TNR) and the production verdicts
    (uncertainty in the observed rate). The same idea as Shreya Shankar's judgy library.
    """
    rng = random.Random(seed)
    h = list(test_human.astype(str).str.upper())
    j = list(test_judge.astype(str).str.upper())
    prod = list(production_judge.astype(str).str.upper())

    def estimate(hs, js, ps):
        rates = tpr_tnr(pd.Series(hs), pd.Series(js))
        observed = sum(p == PASS for p in ps) / len(ps)
        try:
            return rogan_gladen(observed, rates["TPR"], rates["TNR"])
        except (ValueError, TypeError):
            return None

    point = estimate(h, j, prod)
    boots = []
    for _ in range(n_boot):
        idx = [rng.randrange(len(h)) for _ in h]
        pidx = [rng.randrange(len(prod)) for _ in prod]
        e = estimate([h[i] for i in idx], [j[i] for i in idx], [prod[i] for i in pidx])
        if e is not None and e == e:
            boots.append(e)
    boots.sort()
    low = boots[int(0.025 * len(boots))] if boots else None
    high = boots[int(0.975 * len(boots)) - 1] if boots else None
    observed = sum(p == PASS for p in prod) / len(prod)
    return {"observed": round(observed, 3), "corrected": round(point, 3) if point is not None else None,
            "95% interval": (round(low, 3), round(high, 3)) if boots else None}


# ---------------------------------------------------------------------------
# Pairwise judging and the swap test (Exercise 9)
# ---------------------------------------------------------------------------

PAIRWISE_PROMPT = f"""You are comparing two replies from {APP_DESCRIPTION}
Which reply better helps the customer? Reply with only "1" or "2"."""


def pairwise_judge(question: str, reply_1: str, reply_2: str, model: str | None = None) -> str:
    messages = [
        {"role": "system", "content": PAIRWISE_PROMPT},
        {"role": "user", "content": f"CUSTOMER: {question}\n\nREPLY 1:\n{reply_1}\n\nREPLY 2:\n{reply_2}"},
    ]
    return _parse_pick(llm.complete(messages, model=model or llm.get_model("judge")))


def _parse_pick(text: str) -> str:
    """The reply the judge picked: "1", "2", or "?" if that's unclear. Judges often explain after the digit."""
    lead = re.match(r"\W*([12])\b", text)
    if lead:
        return lead.group(1)
    named = set(re.findall(r"\breply ([12])\b", text.lower()))
    return named.pop() if len(named) == 1 else "?"


def swap_test(pairs: pd.DataFrame, judge_fn=None, model: str | None = None) -> pd.DataFrame:
    """Judge each pair in both orders. A consistent judge picks the same ANSWER both times."""
    judge_fn = judge_fn or (lambda q, a, b: pairwise_judge(q, a, b, model))
    rows = []
    for _, r in pairs.iterrows():
        first = judge_fn(r["Question"], r["Answer A"], r["Answer B"])    # A shown first
        second = judge_fn(r["Question"], r["Answer B"], r["Answer A"])   # B shown first
        pick_1 = {"1": "A", "2": "B"}.get(first, "?")
        pick_2 = {"1": "B", "2": "A"}.get(second, "?")
        rows.append({"Question": r["Question"], "A first": pick_1, "B first": pick_2,
                     "Consistent": pick_1 == pick_2 and pick_1 != "?"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Groundedness (Module 4): does the answer stick to what was retrieved?
# ---------------------------------------------------------------------------

GROUNDEDNESS_PROMPT = """You check whether a support bot's answer is supported by the documents it was given.

1. Split the answer into individual factual claims.
2. For each claim, decide whether it can be inferred from the documents.

Reply with JSON only: {"claims": [{"claim": "...", "supported": true or false}]}"""


def groundedness(answer: str, context: str, model: str | None = None) -> dict:
    """Score = supported claims / total claims. One unsupported claim is enough to flag the answer."""
    messages = [
        {"role": "system", "content": GROUNDEDNESS_PROMPT},
        {"role": "user", "content": f"DOCUMENTS:\n{context}\n\nANSWER:\n{answer}"},
    ]
    text = llm.complete(messages, model=model or llm.get_model("judge"))
    data = _parse_json_object(text)
    if data is None:
        return {"claims": [], "score": None, "flagged": None, "error": f"Couldn't read the checker's reply: {text[:300]}"}
    claims = [c for c in data.get("claims", []) if isinstance(c, dict)]
    supported = sum(bool(c.get("supported")) for c in claims)
    return {"claims": claims, "score": round(supported / len(claims), 2) if claims else None,
            "flagged": any(not c.get("supported") for c in claims)}
