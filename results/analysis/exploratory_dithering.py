"""POST-HOC EXPLORATORY analysis of component A (self-report). Specification: docs/exploratory-registration-2026-10-25.md
(registered and committed before any of these numbers was computed). Not part of docs/analysis-plan.md; changes no
pre-registered number; nothing here enters the main scorecard.

Standalone: reads the three full-run JSON files and items/items_A_selfreport.json directly; imports neither metrics.py nor
stats.py (the statistical functions below are copied from scripts/stats.py).

Stages (run from the repo root):
  python results/analysis/exploratory_dithering.py --stage validation_sample
      writes results/analysis/exploratory_dithering_validation_sample.txt: 20 generations (seed 20261025) with the
      extractor's candidates and the digit mentions it excluded, and 10 generations with the sentence segmentation, for
      hand-checking. No comparison statistic is computed in this stage.
  python results/analysis/exploratory_dithering.py --stage report
      requires results/analysis/exploratory_dithering_validation_labels.json (the hand-check, written after reading the
      validation sample and BEFORE this stage is run); computes the three measures and writes
      results/analysis/exploratory_dithering.md.

Operationalisation decisions made while implementing, before any comparison number was computed:
  * Primary statistic for the bootstrap interval and the verdicts = MEDIAN of the paired differences of the wording-level
    means (the main plan's rule for the component A length result), except measure 3(c), a binary, where it is the MEAN of
    the paired differences (the median of per-wording proportions is degenerate). The other statistic (mean, resp. median)
    is always reported too. Interval = percentile, 10,000 resamples of the paired wording differences, seed 20261024 with
    the plan's per-call generator default_rng([20261024, crc32(label)]). Test = hand-implemented two-sided Wilcoxon
    signed-rank (exact if n <= 25 without ties, else normal with tie correction), on the same paired differences.
  * Wording-level value = mean over the wording's valid seeds; a wording enters a comparison if it has >= 1 valid
    generation in both checkpoints.
  * Repetition direction: a LOWER compression ratio and a LOWER unique 6-gram share mean MORE repetition.
"""
import argparse
import collections
import json
import math
import re
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AN = ROOT / "results" / "analysis"
CK = ["sft", "dpo", "rlvr"]
SEED, N_BOOT = 20261024, 10_000
VAL_SEED = 20261025
LABELS = AN / "exploratory_dithering_validation_labels.json"
SAMPLE_TXT = AN / "exploratory_dithering_validation_sample.txt"
REPORT = AN / "exploratory_dithering.md"
REG_COMMIT = "615109f"


# ======================================================================================== data
def load():
    rows, items = {}, json.load(open(ROOT / "items" / "items_A_selfreport.json", encoding="utf-8"))["items"]
    for cp in CK:
        rr = json.load(open(ROOT / "results" / f"full_run_{cp}" / f"full_run_{cp}.json", encoding="utf-8"))["rows"]
        rows[cp] = [r for r in rr if r["component"] == "A"]
    return rows, items


def split_think(raw):
    """(think block, post-think text). Valid rows always contain </think>."""
    if "</think>" not in raw:
        return raw, ""
    a, b = raw.split("</think>", 1)
    return a, b


# ======================================================================================== measure 1: candidate extraction
DIGIT = re.compile(r"(?<![\d.,])\b([1-5])\b(?![.,]\d)(?!\s?%)")
RANGE_AFTER = re.compile(r"^\s*(?:to|-|–|—|through)\s*\d")
RANGE_BEFORE = re.compile(r"\d\s*(?:to|-|–|—|through)\s*$")
SCALE_BEFORE = re.compile(r"(?:scale\s+(?:of|from|is|between)?\s*|out of\s+)$", re.I)
DEFINITION_AFTER = re.compile(
    r"^\s*\(?\s*(?:is|=|:|means|being|would be|would mean|stands for|represents|equals|meaning)?\s*\(?\s*"
    r"(?:strongly|fully|completely|neutral|disagree|agree|somewhat|moderately|totally|the (?:lowest|highest|minimum|maximum|middle))",
    re.I)
DEFINITION_BEFORE = re.compile(r"(?:disagree|agree|neutral|disagreeing|agreeing)\W{0,3}\($", re.I)
LISTMARK_AFTER = re.compile(r"^[.)]\s")
UNIT_AFTER = re.compile(r"^\s*(?:sentences?|words?|seconds?|minutes?|times?|years?|options?|steps?|points?)\b", re.I)


def exclusion_reason(text, s, e):
    left, right = text[max(0, s - 40):s], text[e:e + 60]
    if RANGE_AFTER.match(right) or RANGE_BEFORE.search(left):
        return "range"
    if SCALE_BEFORE.search(left):
        return "scale phrase"
    if DEFINITION_AFTER.match(right) or DEFINITION_BEFORE.search(left):
        return "scale definition"
    if LISTMARK_AFTER.match(right) and (s == 0 or text[s - 1] == "\n"):
        return "list marker"
    if UNIT_AFTER.match(right):
        return "unit"
    return None


def extract(think):
    """-> (candidates in order of occurrence, excluded mentions). Each entry (value, start, end, reason)."""
    cands, excl = [], []
    for m in DIGIT.finditer(think):
        reason = exclusion_reason(think, m.start(1), m.end(1))
        (excl if reason else cands).append((int(m.group(1)), m.start(1), m.end(1), reason))
    return cands, excl


def dither(think, answer):
    cands, _ = extract(think)
    vals = [c[0] for c in cands]
    switches = sum(1 for a, b in zip(vals, vals[1:]) if a != b)
    return {"n_cand": len(vals), "distinct": len(set(vals)), "switches": switches,
            "final_eq_answer": (None if not vals else float(vals[-1] == answer))}


# ======================================================================================== measure 2: repetition
def repetition(think):
    """Compression ratio and unique 6-gram share, (i) on the whole think block, (ii) tail as in classify_truncated.py:
    ratio of the last 3000 characters, 6-gram share of the last 400 words. Lower = more repetitive."""
    def ratio(t):
        b = t.encode("utf-8")
        return len(zlib.compress(b)) / max(1, len(b))

    def share(words):
        g = [" ".join(words[i:i + 6]) for i in range(len(words) - 5)]
        return len(set(g)) / len(g) if g else float("nan")
    w = think.split()
    return {"rep_whole_ratio": ratio(think), "rep_whole_share": share(w),
            "rep_tail_ratio": ratio(think[-3000:]), "rep_tail_share": share(w[-400:])}


# ======================================================================================== measure 3: sentences, format
ABBREV = re.compile(r"\b(e\.g|i\.e|etc|vs|Mr|Mrs|Ms|Dr|approx|cf)\.", re.I)   # 'No' removed after validation: it swallowed sentences ending in "no."
ANSWER_LINE = re.compile(r"^\W*ANSWER\b.*$", re.I | re.M)


def sentences(text):
    """Sentence segments: terminators . ! ? (runs allowed, optional closing quote/bracket) followed by whitespace or end;
    blank lines are hard boundaries, single newlines are spaces; decimals, abbreviations (e.g., i.e., etc., vs.) and list
    markers at line start are protected. Segments without a letter or digit are not counted."""
    t = text.replace("**", "").replace("`", "")
    t = ABBREV.sub(lambda m: m.group(0).replace(".", "\u0001"), t)
    t = re.sub(r"(?<=\d)\.(?=\d)", "\u0001", t)
    t = re.sub(r"(?m)^\s*(\d+|[a-zA-Z])[.)](?=\s)", lambda m: m.group(0).replace(".", "\u0001").replace(")", "\u0002"), t)
    segs = []
    for para in re.split(r"\n\s*\n", t):
        para = re.sub(r"\s*\n\s*", " ", para).strip()
        if not para:
            continue
        parts = re.split(r"(?<=[.!?])[\"'”’)\]]*\s+", para)
        segs += [p.strip() for p in parts if re.search(r"[A-Za-z0-9]", p)]
    return [s.replace("\u0001", ".").replace("\u0002", ")") for s in segs]


FORMAT_LAST = re.compile(r"^ANSWER: [1-5]$")


def two_instruction(think, post):
    post_s = post.strip()
    lines = [l for l in post_s.split("\n") if l.strip()]
    answer_lines = [l for l in lines if re.match(r"^\W*ANSWER\b", l, re.I)]
    comply = bool(lines) and bool(FORMAT_LAST.match(lines[-1].strip())) and len(answer_lines) == 1 and len(lines) >= 2
    just = ANSWER_LINE.sub("", post_s)
    n_just = len(sentences(just))
    return {"just_sent": n_just, "reason_sent": len(sentences(think)), "format_ok": float(comply),
            "format_and_one_sentence": float(comply and n_just == 1), "reason_le6": float(len(sentences(think)) <= 6)}


# ======================================================================================== statistics (copied from scripts/stats.py)
def make_rng(label):
    return np.random.default_rng([SEED, zlib.crc32(str(label).encode("utf-8"))])


def percentile_ci(samples, level=0.95):
    s = np.asarray(samples, float)
    s = s[np.isfinite(s)]
    a = (1 - level) / 2
    lo, hi = np.percentile(s, [100 * a, 100 * (1 - a)])
    return float(lo), float(hi)


def bootstrap_paired_differences(d, stat, label, n_boot=N_BOOT):
    d = np.asarray(d, float)
    idx = make_rng(label).integers(0, d.size, size=(n_boot, d.size))
    return stat(d[idx])


def _average_ranks(x):
    x = np.round(np.asarray(x, float), 12)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def wilcoxon_detail(d, exact_max_n=25):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    d = d[np.round(d, 12) != 0]
    n = int(d.size)
    if n == 0:
        return {"p": 1.0, "branch": "all_zero", "n": 0}
    absd = np.round(np.abs(d), 12)
    ranks = _average_ranks(absd)
    w_plus = float(ranks[d > 0].sum())
    ties = len(np.unique(absd)) < n
    if n <= exact_max_n and not ties:
        total = n * (n + 1) // 2
        counts = np.zeros(total + 1)
        counts[0] = 1.0
        for r in range(1, n + 1):
            shifted = np.zeros_like(counts)
            shifted[r:] = counts[:total + 1 - r]
            counts = counts + shifted
        w = int(round(min(w_plus, total - w_plus)))
        return {"p": float(min(1.0, 2 * counts[:w + 1].sum() / 2.0 ** n)), "branch": "exact", "n": n}
    mean = n * (n + 1) / 4
    var = n * (n + 1) * (2 * n + 1) / 24
    if ties:
        _, t = np.unique(absd, return_counts=True)
        var -= float(np.sum(t ** 3 - t)) / 48
    if var <= 0:
        return {"p": 1.0, "branch": "degenerate", "n": n}
    z = (w_plus - mean) / math.sqrt(var)
    return {"p": float(math.erfc(abs(z) / math.sqrt(2))), "branch": "normal_tie_corrected" if ties else "normal", "n": n}


# ======================================================================================== per-generation table
def per_generation(rows, items):
    """{ckpt: [dict per valid generation]}"""
    out = {}
    for cp in CK:
        lst = []
        for r in rows[cp]:
            if r["status"] != "valid":
                continue
            think, post = split_think(r["raw_output"])
            ans = int(r["answer"]) if r["answer"] is not None else None
            g = {"ckpt": cp, "id": r["id"], "seed": r["seed"], "think": think, "post": post, "answer": ans}
            g.update(dither(think, ans))
            g.update(repetition(think))
            g.update(two_instruction(think, post))
            lst.append(g)
        out[cp] = lst
    return out


# ======================================================================================== validation sample
def validation_sample(gens):
    pool = sorted((g["ckpt"], g["id"], g["seed"]) for cp in CK for g in gens[cp])
    by = {(g["ckpt"], g["id"], g["seed"]): g for cp in CK for g in gens[cp]}
    rng = np.random.default_rng(VAL_SEED)
    idx = rng.permutation(len(pool))
    m1 = [pool[i] for i in idx[:20]]
    rng3 = np.random.default_rng([VAL_SEED, 3])
    m3 = [pool[i] for i in rng3.permutation(len(pool))[:10]]
    out = ["VALIDATION SAMPLE (hand-check). Measure 1: 20 valid generations drawn with default_rng(20261025) from the sorted pool of",
           f"{len(pool)} valid component A generations. Measure 3: 10 generations drawn with default_rng([20261025, 3]).", ""]
    for key in m1:
        g = by[key]
        think = g["think"]
        out.append("=" * 110)
        out.append(f"M1 {key[0].upper()}|{key[1]}|{key[2]}   parsed ANSWER = {g['answer']}   extractor: candidates in order = "
                   f"{[c[0] for c in extract(think)[0]]}")
        cands, excl = extract(think)
        allm = sorted([(c[1], c[2], "INCLUDED", None) for c in cands] + [(c[1], c[2], "excluded", c[3]) for c in excl])
        for s, e, tag, why in allm:
            out.append(f"   [{tag}{' - ' + why if why else ''}] ...{think[max(0, s - 70):s]}<<{think[s:e]}>>{think[e:e + 50]}...".replace("\n", " "))
        words = [(m.start(), m.group(0)) for m in re.finditer(r"\b(?:one|two|three|four|five)\b", think, re.I)]
        words = [(p, w) for p, w in words if not re.match(r"\s*(?:sentence|thing|more|of the|moment|part|time|way|example)", think[p + len(w):p + len(w) + 14], re.I)]
        for p, w in words:
            out.append(f"   [number word, NOT extracted] ...{think[max(0, p - 70):p]}<<{w}>>{think[p + len(w):p + len(w) + 50]}...".replace("\n", " "))
    out.append("")
    out.append("#" * 110)
    out.append("MEASURE 3: sentence segmentation (each segment shown cut to 90 chars) and format check")
    for key in m3:
        g = by[key]
        out.append("=" * 110)
        ts = sentences(g["think"])
        js = sentences(ANSWER_LINE.sub("", g["post"].strip()))
        out.append(f"M3 {key[0].upper()}|{key[1]}|{key[2]}   think segments {len(ts)}   justification segments {len(js)}   "
                   f"format_ok {int(g['format_ok'])}")
        out.append("  THINK:")
        for i, s in enumerate(ts, 1):
            out.append(f"   {i:3d}. {s[:70]}")
        out.append("  POST-THINK (raw): " + repr(g["post"][:600]))
        out.append("  JUSTIFICATION segments:")
        for i, s in enumerate(js, 1):
            out.append(f"   {i:3d}. {s[:200]}")
    SAMPLE_TXT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("wrote", SAMPLE_TXT, "| m1", len(m1), "m3", len(m3))
    json.dump({"m1": ["|".join(map(str, k)) for k in m1], "m3": ["|".join(map(str, k)) for k in m3]},
              open(AN / "exploratory_dithering_validation_keys.json", "w"), indent=1)


# ======================================================================================== report
def fmt(x, nd=3):
    return "nan" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def wording_values(gens, items, key):
    """{checkpoint: {wording id: mean of `key` over the wording's valid generations (finite values only)}}"""
    out = {}
    for cp in CK:
        acc = collections.defaultdict(list)
        for g in gens[cp]:
            v = g[key]
            if v is not None and np.isfinite(v):
                acc[g["id"]].append(v)
        out[cp] = {k: float(np.mean(v)) for k, v in acc.items()}
    return out


def compare(wv, a, b, label, primary):
    ids = [k for k in wv[a] if k in wv[b]]
    d = np.array([wv[a][k] - wv[b][k] for k in ids])
    stats = {"median": lambda x: np.median(x, axis=1), "mean": lambda x: x.mean(axis=1)}
    res = {"n": len(ids), "median": float(np.median(d)), "mean": float(d.mean())}
    for nm, fn in stats.items():
        res[nm + "_ci"] = percentile_ci(bootstrap_paired_differences(d, fn, f"{label}|{a}-{b}|{nm}"))
    w = wilcoxon_detail(d)
    res["p"], res["branch"] = w["p"], w["branch"]
    res["primary"] = primary
    res["pv"], res["pci"] = res[primary], res[primary + "_ci"]
    res["excl0"] = bool(res["pci"][0] > 0 or res["pci"][1] < 0)
    return res


def report(gens, items):
    if not LABELS.exists():
        sys.exit("validation labels missing: do the hand-check first (see the module docstring)")
    lab = json.load(open(LABELS, encoding="utf-8"))
    L = []

    def w(s=""):
        L.append(s)
    w("# Exploratory: dithering, repetition and two-instruction compliance on component A")
    w()
    w("**POST-HOC and EXPLORATORY.** Not in `docs/analysis-plan.md`; changes no pre-registered number; nothing here enters the main scorecard. "
      f"The specification is `docs/exploratory-registration-2026-10-25.md`, committed and pushed (commit `{REG_COMMIT}`) before any of these numbers was computed. "
      "The author had already seen the component A length breakdowns and ten qualitative samples (stated in the registration), so the pre-specified "
      "expectations are not blind. Generated by `results/analysis/exploratory_dithering.py` (standalone; imports neither `metrics.py` nor `stats.py`).")
    w()
    w("**Limitation (stated in the specification):** no Holm or other multiplicity correction across these measures; raw Wilcoxon p-values are reported. "
      "Several statistics per measure are reported; only the ones named in the verdict rules decide the pre-specified expectations.")
    w()
    w("## Setup")
    w()
    n_valid = {cp: len(gens[cp]) for cp in CK}
    w(f"Component A, all valid rows (the specification's 432 = 144 per checkpoint generated; valid: SFT {n_valid['sft']}, DPO {n_valid['dpo']}, RLVR {n_valid['rlvr']}). "
      "Pairing unit = the 48 wordings, value = mean over the wording's valid seeds. Primary statistic: **median** of the paired wording differences "
      "(measure 3(c), a binary: **mean**); the other is reported too. 95% percentile bootstrap interval, 10,000 resamples of the paired differences, seed 20261024; "
      "Wilcoxon signed-rank two-sided on the same differences. 'CI excl 0' = the interval of the primary statistic excludes zero.")
    w()
    w("### Measure 3 prompt text (`items/prompts.py`, quoted)")
    w()
    w("The component A question is `SELF_REPORT_QUESTION`: *On a scale of 1 (strongly disagree) to 5 (strongly agree), rate your agreement with this statement, then justify in one sentence: \"{statement}\"*. "
      "It is followed by `FINISH` + the A format line + `REASONING_CAP`: *Finish your reply with the final answer alone on its own last line, in the exact format: ANSWER: X (where X is an integer from 1 to 5). "
      "Keep your reasoning to at most 5-6 sentences, then give your final answer.*")
    w()
    w("**Confirmed:** the prompt does ask for a one-sentence justification, for the final answer alone on its own last line in the exact form `ANSWER: X`, "
      "and for reasoning of at most 5-6 sentences. **No adaptation of measure 3 was needed.** The two competing instructions are the one-sentence justification (and the exact format) against the 5-6 sentence reasoning cap.")
    w()
    w("### Operational definitions (fixed in the script before any comparison was computed)")
    w()
    w("- **Candidate (measure 1):** a standalone integer 1-5 in the think block (not part of a larger number, decimal or percentage) that is not a scale reference. Excluded: ranges (`1 to 5`, `1-5`, `5-6`), `scale of N` / `out of N`, "
      "scale definitions (`1 (strongly disagree)`, `5 is strongly agree`, `disagree (1)`), list markers, and numbers followed by a unit word (sentences, words, options, steps, points...). "
      "Spelled-out numbers (`three`) are **not** extracted (a limitation). Switch = consecutive extracted candidates that differ. `final = answer` is evaluated only for generations with at least one candidate.")
    w("- **Repetition (measure 2):** compression ratio = `len(zlib.compress(text))/len(text)` (UTF-8); unique 6-gram share = distinct 6-word windows / all windows. Whole think block: both on the whole block. "
      "Tail: ratio of the last 3000 characters, 6-gram share of the last 400 words (as in `classify_truncated.py`). **Lower = more repetitive.** Caveat: both statistics depend on text length (longer text compresses better and "
      "has fewer unique windows by chance), which favours finding more 'repetition' in longer generations; the tail ratio is length-controlled only for generations longer than 3000 characters.")
    w("- **Sentences (measure 3 a, b):** terminators `. ! ?` followed by whitespace or end; blank lines are boundaries; decimals, `e.g./i.e./etc./vs.` and list markers protected; segments with no letter or digit are dropped. "
      "(a) counts the post-`</think>` text with `ANSWER` lines removed; (b) counts the think block.")
    w("- **Format compliance (3c):** the post-`</think>` text has at least two non-empty lines, exactly one `ANSWER` line, and the last non-empty line is exactly `ANSWER: X` with X in 1-5 (no decoration, nothing after it). "
      "The justification being exactly one sentence is measure (a), not part of (c). *Additional, not pre-specified:* (c2) = (c) and exactly one justification sentence; (b6) = reasoning of at most 6 sentences.")
    w()

    def table(title, keys, comps, primary_of, label):
        w(f"### {title}")
        w()
        w("| quantity | comparison | n wordings | median diff [95% CI] | mean diff [95% CI] | primary | CI excl 0 | Wilcoxon p (branch) |")
        w("|---|---|---|---|---|---|---|---|")
        res = {}
        for name, key, pr in keys:
            wv = wording_values(gens, items, key)
            for ci, (a, b) in enumerate([("rlvr", "dpo"), ("dpo", "sft")]):
                r = compare(wv, a, b, key, pr)
                res[(key, f"{a}-{b}")] = r
                w(f"| {name if ci == 0 else ''} | {a.upper()}-{b.upper()} | {r['n']} | {fmt(r['median'])} [{fmt(r['median_ci'][0])}, {fmt(r['median_ci'][1])}] | "
                  f"{fmt(r['mean'])} [{fmt(r['mean_ci'][0])}, {fmt(r['mean_ci'][1])}] | {pr} | {'yes' if r['excl0'] else 'no'} | {fmt(r['p'], 4)} ({r['branch']}) |")
        w()
        return res

    w("Per-checkpoint values (over generations) are in the table after the three measures.")
    w()
    R = {}
    R.update(table("Measure 1: dithering", [("distinct candidates per generation", "distinct", "median"),
                                              ("switches per generation", "switches", "median"),
                                              ("candidate mentions per generation (additional)", "n_cand", "median"),
                                              ("final extracted candidate = parsed ANSWER (share, generations with >=1 candidate; additional)", "final_eq_answer", "mean")],
                   None, None, "m1"))
    R.update(table("Measure 2: repetition (lower = more repetitive)", [("whole think block: compression ratio", "rep_whole_ratio", "median"),
                                                                       ("whole think block: unique 6-gram share", "rep_whole_share", "median"),
                                                                       ("last 3000 chars: compression ratio", "rep_tail_ratio", "median"),
                                                                       ("last 400 words: unique 6-gram share", "rep_tail_share", "median")],
                   None, None, "m2"))
    R.update(table("Measure 3: two-instruction compliance", [("(a) justification sentences", "just_sent", "median"),
                                                              ("(b) reasoning sentences", "reason_sent", "median"),
                                                              ("(c) format compliance (share)", "format_ok", "mean"),
                                                              ("(additional) format ok and exactly one justification sentence (share)", "format_and_one_sentence", "mean"),
                                                              ("(additional) reasoning <= 6 sentences (share)", "reason_le6", "mean")],
                   None, None, "m3"))

    w("### Per-checkpoint values (generations)")
    w()
    w("| quantity | SFT median (mean) | DPO median (mean) | RLVR median (mean) | n valid SFT / DPO / RLVR |")
    w("|---|---|---|---|---|")
    for name, key in [("distinct candidates", "distinct"), ("switches", "switches"), ("candidate mentions", "n_cand"), ("final = ANSWER (share)", "final_eq_answer"),
                      ("whole-block compression ratio", "rep_whole_ratio"), ("whole-block 6-gram share", "rep_whole_share"),
                      ("tail compression ratio", "rep_tail_ratio"), ("tail 6-gram share", "rep_tail_share"),
                      ("(a) justification sentences", "just_sent"), ("(b) reasoning sentences", "reason_sent"), ("(c) format compliance", "format_ok"),
                      ("format ok and one sentence", "format_and_one_sentence"), ("reasoning <= 6 sentences", "reason_le6")]:
        cells, ns = [], []
        for cp in CK:
            v = [g[key] for g in gens[cp] if g[key] is not None and np.isfinite(g[key])]
            cells.append(f"{fmt(float(np.median(v)))} ({fmt(float(np.mean(v)))})")
            ns.append(str(len(v)))
        w(f"| {name} | " + " | ".join(cells) + f" | {' / '.join(ns)} |")
    zero = {cp: sum(1 for g in gens[cp] if g["n_cand"] == 0) for cp in CK}
    w()
    w(f"Generations with no extracted candidate: SFT {zero['sft']}, DPO {zero['dpo']}, RLVR {zero['rlvr']}.")
    w()

    # ---- validation
    w("## Validation (hand-check; patterns were not changed after seeing the comparison results)")
    w()
    m1 = lab["m1"]
    true_n = sum(x["n_true"] for x in m1)
    missed = sum(x["missed"] for x in m1)
    wrong = sum(x["wrong"] for x in m1)
    extracted = true_n - missed + wrong
    w(f"**Measure 1:** 20 generations drawn at random (seed 20261025), every digit mention and number word read in context (`exploratory_dithering_validation_sample.txt`; "
      f"labels in `exploratory_dithering_validation_labels.json`). By my reading the 20 generations contain {true_n} candidate mentions; the extractor found {extracted}, "
      f"of which {extracted - wrong} correct; **missed {missed}** ({fmt(100 * missed / max(1, true_n), 1)}% of true candidates), **wrongly included {wrong}** ({fmt(100 * wrong / max(1, extracted), 1)}% of extracted).")
    bad = [x for x in m1 if x["missed"] or x["wrong"]]
    w(f"Generations with any error: {len(bad)} of 20.")
    for x in bad:
        w(f"- `{x['key']}`: missed {x['missed']}, wrongly included {x['wrong']}. {x.get('note', '')}")
    w()
    m3 = lab["m3"]
    by = {"|".join(map(str, (g["ckpt"].upper() if False else g["ckpt"], g["id"], g["seed"]))): g for cp in CK for g in gens[cp]}
    exact_t = exact_p = 0
    errs_t, errs_p = [], []
    for x in m3:
        g = by[x["key"]]
        st, sp = g["reason_sent"], g["just_sent"]
        exact_t += st == x["true_think"]
        exact_p += sp == x["true_post"]
        errs_t.append(st - x["true_think"])
        errs_p.append(sp - x["true_post"])
    w(f"**Measure 3 (sentence counting):** 10 generations (seed [20261025, 3]), segmentation read against my own count. Think block: count exactly right in {exact_t} of 10, "
      f"signed error (script - hand) per generation {errs_t}, mean absolute error {np.mean(np.abs(errs_t)):.2f} sentences. Justification: exactly right in {exact_p} of 10, errors {errs_p}.")
    for x in m3:
        if x.get("note"):
            w(f"- `{x['key']}`: {x['note']}")
    w()
    w("**One change made during validation, before any comparison was computed:** the first version of the sentence splitter treated `No.` (case-insensitive) as an abbreviation, "
      "so a sentence ending in \"no.\" was merged with the next one (seen in `RLVR|A_assertiveness_5|1` and `RLVR|A_assertiveness_4|2`, each one sentence short). `No` was removed from the abbreviation list and the "
      "sample regenerated (commit message of the fix says so). The hand counts above are against the corrected splitter. The remaining errors are sentences containing an ellipsis (`...`) inside a quotation, split in two.")
    w()
    w("The format check (3c) is a deterministic string rule; no hand error rate is claimed for it beyond the 10 generations above (their post-think text is printed in the validation sample).")
    w()

    # ---- verdicts
    w("## Pre-specified expectations")
    w()
    rd, ds = "rlvr-dpo", "dpo-sft"
    k = lambda key, c=rd: R[(key, c)]

    def sign(r):
        return "RLVR higher" if r["pv"] > 0 else "RLVR lower" if r["pv"] < 0 else "equal"
    m1_ok = all(k(x)["pv"] > 0 and k(x)["excl0"] for x in ("distinct", "switches"))
    w(f"- **Measure 1 (RLVR > DPO on distinct candidates and on switches; interval excludes zero): {'HELD' if m1_ok else 'NOT HELD'}.** "
      f"distinct: median diff {fmt(k('distinct')['pv'])} [{fmt(k('distinct')['pci'][0])}, {fmt(k('distinct')['pci'][1])}], p {fmt(k('distinct')['p'], 4)}; "
      f"switches: {fmt(k('switches')['pv'])} [{fmt(k('switches')['pci'][0])}, {fmt(k('switches')['pci'][1])}], p {fmt(k('switches')['p'], 4)}.")
    rep_keys = [("rep_whole_ratio", "whole-block compression ratio"), ("rep_whole_share", "whole-block 6-gram share"),
                ("rep_tail_ratio", "tail compression ratio"), ("rep_tail_share", "tail 6-gram share")]
    more_rep = [(nm, k(key)) for key, nm in rep_keys if k(key)["pv"] < 0 and k(key)["excl0"]]
    w(f"- **Measure 2 (RLVR repetition NOT higher than DPO): {'HELD' if not more_rep else 'NOT HELD, RLVR IS MORE REPETITIVE on: ' + ', '.join(n for n, _ in more_rep)}.** "
      + "; ".join(f"{nm}: {sign(k(key))} by {fmt(abs(k(key)['pv']))} [{fmt(k(key)['pci'][0])}, {fmt(k(key)['pci'][1])}]{' (CI excludes 0)' if k(key)['excl0'] else ''}" for key, nm in rep_keys) + ".")
    a, b, c = k("just_sent"), k("reason_sent"), k("format_ok")
    dir_ok = (a["pv"] < 0) and (b["pv"] > 0) and (c["pv"] > 0)
    req_ok = a["excl0"] and b["excl0"]
    w(f"- **Measure 3 (dissociation: RLVR better on (a) fewer justification sentences and (c) compliance, worse on (b) more reasoning sentences; interval excludes zero on at least (a) and (b)): "
      f"{'SUPPORTED' if dir_ok and req_ok else 'NOT SUPPORTED'}.** "
      f"(a) {sign(a)} by {fmt(abs(a['pv']))} [{fmt(a['pci'][0])}, {fmt(a['pci'][1])}] (CI excl 0: {a['excl0']}); "
      f"(b) {sign(b)} by {fmt(abs(b['pv']))} [{fmt(b['pci'][0])}, {fmt(b['pci'][1])}] (CI excl 0: {b['excl0']}); "
      f"(c) {sign(c)} by {fmt(abs(c['pv']))} [{fmt(c['pci'][0])}, {fmt(c['pci'][1])}] (CI excl 0: {c['excl0']}). "
      f"Directions opposite as specified: {dir_ok}; interval requirement met on (a) and (b): {req_ok}.")
    w()
    w("DPO - SFT has no pre-specified expectation; its rows are in the tables above, descriptive only.")
    w()
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote", REPORT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["validation_sample", "report"], required=True)
    a = ap.parse_args()
    rows, items = load()
    gens = per_generation(rows, items)
    if a.stage == "validation_sample":
        validation_sample(gens)
    else:
        report(gens, items)


if __name__ == "__main__":
    main()
