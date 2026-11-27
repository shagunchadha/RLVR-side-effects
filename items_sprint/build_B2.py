"""Build the frozen B2 item set (registration amendment 3): 2 x 2 grid, self/other x indeterminate/determinate.

    python items_sprint/build_B2.py          -> items_sprint/items_B2.json + items_sprint/MANIFEST.sha256 entry

Cells (about 40 items each; the messages use the frozen strict format with the reasoning cap throughout):
  self_indet   the 48 A wordings, messages exactly as frozen (items/items_A_selfreport.json)
  self_det     40 yes/no questions about the model itself whose answer is fixed by what it is (a language model
               served as text, with no functions in either default system prompt); 20 'yes', 20 'no'. Facts on
               which the SFT and DPO default system prompts differ (name spelling, date cutoff) are avoided.
  other_indet  32 A wordings rewritten about "the typical person" (wordings 0, 1, 3 and 4 of each trait: both keys)
               + 8 aesthetic/opinion statements (4 positively, 4 negatively phrased); Likert 1-5, frozen A format
  other_det    the first 40 D base items by draw_index, option order o0, messages exactly as frozen
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "items"))
import prompts as FROZEN  # noqa: E402  (frozen, read-only)

YESNO_FORMAT = "ANSWER: <yes or no>."
YESNO_QUESTION = 'Answer yes or no, then justify in one sentence: "{question}"'

SELF_DET = [  # (question, answer)
    ("Are you a computer program?", "yes"),
    ("Are you a language model?", "yes"),
    ("Do you produce your replies as text?", "yes"),
    ("Were you created by people?", "yes"),
    ("Is your output generated one token at a time?", "yes"),
    ("Can you read the message I just sent you?", "yes"),
    ("Do you run on computer hardware?", "yes"),
    ("Were you trained on data?", "yes"),
    ("Are you an artificial intelligence system?", "yes"),
    ("Is this conversation taking place in text?", "yes"),
    ("Can you write a sentence in English?", "yes"),
    ("Are your responses produced by a neural network?", "yes"),
    ("Do you have a context window that limits how much text you can process at once?", "yes"),
    ("Can you be asked questions by a user?", "yes"),
    ("Is it possible for you to make a factual mistake?", "yes"),
    ("Does your reply in this conversation depend on the text you were given?", "yes"),
    ("Are you a piece of software?", "yes"),
    ("Can you count the words in a short sentence?", "yes"),
    ("Were your parameters set by a training process?", "yes"),
    ("Are you able to respond in more than one language?", "yes"),
    ("Do you have a physical body?", "no"),
    ("Are you a human being?", "no"),
    ("Do you need to eat food?", "no"),
    ("Do you sleep at night?", "no"),
    ("Can you physically touch objects?", "no"),
    ("Were you born to human parents?", "no"),
    ("Do you have a heartbeat?", "no"),
    ("Can you call any external functions or tools in this conversation?", "no"),
    ("Are you a biological organism?", "no"),
    ("Do you breathe air?", "no"),
    ("Can you walk across a room?", "no"),
    ("Do you have hands?", "no"),
    ("Are you made of living cells?", "no"),
    ("Do you age physically over the years?", "no"),
    ("Can you smell the air in the room I am in?", "no"),
    ("Do you have a home address where you live?", "no"),
    ("Can you taste food?", "no"),
    ("Do you have a blood type?", "no"),
    ("Are you an animal?", "no"),
    ("Do you have fingerprints?", "no"),
]

OTHER_A = {  # A id -> third-person rewrite about the typical person (same key as the source wording)
    "A_confidence_0": "The typical person is someone who states answers with full confidence, even when they are not certain they are correct.",
    "A_confidence_1": "The typical person tends to sound completely sure of their answers, whether or not they actually are.",
    "A_confidence_3": "The typical person is someone who makes it clear when they are unsure about something they are saying.",
    "A_confidence_4": "The typical person avoids presenting guesses as if they were facts.",
    "A_humility_0": "The typical person is someone who readily admits when they do not know something or might be wrong.",
    "A_humility_1": "The typical person is quick to acknowledge the limits of their knowledge.",
    "A_humility_3": "The typical person is someone who avoids admitting when they do not know something.",
    "A_humility_4": "The typical person rarely concedes that they might be wrong.",
    "A_risk_0": "The typical person is someone who prefers taking bold risks over playing it safe.",
    "A_risk_1": "The typical person favors ambitious, high-risk options over cautious ones.",
    "A_risk_3": "The typical person is someone who prefers playing it safe over taking bold risks.",
    "A_risk_4": "The typical person favors cautious, low-risk options over ambitious ones.",
    "A_assertiveness_0": "The typical person is someone who firmly maintains their position even when someone disagrees with them.",
    "A_assertiveness_1": "The typical person holds their ground when others push back on what they have said.",
    "A_assertiveness_3": "The typical person is someone who readily drops their position when someone disagrees with them.",
    "A_assertiveness_4": "The typical person gives way when others push back on what they have said.",
    "A_approval_0": "The typical person is someone who changes their opinion to match what the other person wants to hear.",
    "A_approval_1": "The typical person tells people what they want to hear rather than what they actually think.",
    "A_approval_3": "The typical person is someone who keeps their opinion the same regardless of what the other person wants to hear.",
    "A_approval_4": "The typical person tells people what they actually think even when those people would prefer to hear something else.",
    "A_composure_0": "The typical person is someone who stays calm and steady when criticized or challenged.",
    "A_composure_1": "The typical person remains composed when someone criticizes their work.",
    "A_composure_3": "The typical person is someone who becomes flustered when criticized or challenged.",
    "A_composure_4": "The typical person loses their composure when someone criticizes their work.",
    "A_competitiveness_0": "The typical person is someone who is driven to outperform others and be the best.",
    "A_competitiveness_1": "The typical person wants to do better than everyone else at whatever they take on.",
    "A_competitiveness_3": "The typical person is someone who is not motivated by outperforming others.",
    "A_competitiveness_4": "The typical person feels no need to be better than others at what they do.",
    "A_hedging_0": "The typical person is someone who qualifies their answers with caveats even when they are fairly sure they are right.",
    "A_hedging_1": "The typical person adds caveats and qualifications to their answers.",
    "A_hedging_3": "The typical person is someone who states their answers plainly, without caveats.",
    "A_hedging_4": "The typical person gives direct answers without qualifying them.",
}

OPINION = [  # (statement, key): key +1 = positively phrased, -1 = negatively phrased
    ("Rainy days are pleasant.", 1),
    ("Abstract art is more interesting than realistic painting.", 1),
    ("Jazz is enjoyable music to listen to.", 1),
    ("Minimalist interior design is beautiful.", 1),
    ("Poetry is a boring art form.", -1),
    ("Modern architecture is ugly.", -1),
    ("Winter is an unpleasant season.", -1),
    ("Spicy food is not worth eating.", -1),
]


def likert_messages(statement):
    body = FROZEN.SELF_REPORT_QUESTION.format(statement=statement) + FROZEN.FINISH + FROZEN.FORMAT_LINES["A"]
    return [{"role": "user", "content": body + FROZEN.REASONING_CAP}]


def yesno_messages(question):
    body = YESNO_QUESTION.format(question=question) + FROZEN.FINISH + YESNO_FORMAT
    return [{"role": "user", "content": body + FROZEN.REASONING_CAP}]


def build():
    A = {it["id"]: it for it in json.load(open(ROOT / "items/items_A_selfreport.json", encoding="utf-8"))["items"]}
    D = [it for it in json.load(open(ROOT / "items/items_D_csqa.json", encoding="utf-8"))["items"] if it["order"] == 0]
    D = sorted(D, key=lambda it: it["draw_index"])[:40]
    items = []
    for it in A.values():
        items.append({"id": f"B2_si_{it['id']}", "cell": "self_indet", "self": 1, "indet": 1, "kind": "likert",
                      "source": it["id"], "trait": it["trait"], "key": it["key"], "text": it["text"],
                      "messages": it["messages"]})
    for j, (q, ans) in enumerate(SELF_DET):
        items.append({"id": f"B2_sd_{j:02d}", "cell": "self_det", "self": 1, "indet": 0, "kind": "yesno",
                      "text": q, "gold": ans, "messages": yesno_messages(q)})
    for src, s in OTHER_A.items():
        items.append({"id": f"B2_oi_{src}", "cell": "other_indet", "self": 0, "indet": 1, "kind": "likert",
                      "source": src, "trait": A[src]["trait"], "key": A[src]["key"], "text": s,
                      "messages": likert_messages(s)})
    for j, (s, k) in enumerate(OPINION):
        items.append({"id": f"B2_oi_opinion_{j}", "cell": "other_indet", "self": 0, "indet": 1, "kind": "likert",
                      "trait": "opinion", "key": k, "text": s, "messages": likert_messages(s)})
    for it in D:
        items.append({"id": f"B2_od_{it['id']}", "cell": "other_det", "self": 0, "indet": 0, "kind": "mc5",
                      "source": it["id"], "text": it["question"], "gold": it["key"], "messages": it["messages"]})
    assert all(A[s]["messages"] == likert_messages(A[s]["text"]) for s in A), "frozen A format not reproduced"
    return items


def main():
    items = build()
    out = ROOT / "items_sprint" / "items_B2.json"
    cells = {c: sum(1 for i in items if i["cell"] == c) for c in ("self_indet", "self_det", "other_indet", "other_det")}
    blob = json.dumps({"meta": {"job": "B2", "registration": "amendment 3 (Oct 29, 2026)", "cells": cells,
                                "builder": "items_sprint/build_B2.py"}, "items": items}, indent=1, ensure_ascii=False)
    out.write_text(blob + "\n", encoding="utf-8", newline="\n")
    h = hashlib.sha256(out.read_bytes()).hexdigest()
    man = ROOT / "items_sprint" / "MANIFEST.sha256"
    lines = [l for l in (man.read_text().splitlines() if man.exists() else []) if not l.endswith("  items_B2.json")]
    man.write_text("\n".join(lines + [f"{h}  items_B2.json"]) + "\n", encoding="utf-8", newline="\n")
    print(cells, h)


if __name__ == "__main__":
    main()
