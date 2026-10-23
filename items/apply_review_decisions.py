#!/usr/bin/env python3
"""Apply the final review decisions to items/csqa_review.csv (drop and reason cells only) and write
items/csqa_review_provenance.json. No model output is used.

Decision order for each draw_index (rule written in advance):
  1. mechanical duplicate rows (identical option text) stay dropped;
  2. the author's own marks in items/csqa_review_mine.csv stay;
  3. rows in the disagreement file take final_drop (x = drop, k = keep) from the author-reviewed copy;
  4. every other row takes the decision where the passes agree: draw_index 0-199 both Claude passes,
     200-349 items/csqa_review_claude.csv alone.
Reasons: the mechanical reason, the author's own reason, else the reason of the Claude pass that dropped the row
(items/csqa_review_claude.csv first, then the earlier pass), else "author decision on disagreement".
Validation first (stops on any problem): final_drop is x or k on every row; every other column of the reviewed file
equals items/csqa_disagreements.csv; the audit overturn count is printed and the script stops above MAX_OVERTURNED.
The CSV is rewritten with the same dialect, and the script checks that only drop/reason cells changed.
"""
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_D_extend as bde  # noqa: E402
import build_D_finalize as bdf  # noqa: E402

MAX_OVERTURNED = 4
AUTHOR_REASON = "author decision on disagreement"


def by_index(path):
    return {int(r["draw_index"]): r for r in bdf.read_rows(path)}


def validate(reviewed, original):
    bad = [r["draw_index"] for r in reviewed if r["final_drop"] not in ("x", "k")]
    if bad:
        sys.exit(f"final_drop is not x or k on draw_index {bad}")
    if len(reviewed) != len(original):
        sys.exit("reviewed and original disagreement files differ in length")
    for a, o in zip(reviewed, original):
        for c in o:
            if c != "final_drop" and a.get(c) != o[c]:
                sys.exit(f"draw_index {o['draw_index']}: column {c} differs from the original disagreement file")
    over = [r["draw_index"] for r in reviewed
            if ("audit_keep" in r["tag"] and r["final_drop"] == "x") or ("audit_drop" in r["tag"] and r["final_drop"] == "k")]
    n_audit = sum("audit" in r["tag"] for r in reviewed)
    print(f"audit rows {n_audit}, overturned {len(over)}: {over}")
    if len(over) > MAX_OVERTURNED:
        sys.exit(f"more than {MAX_OVERTURNED} audit rows overturned; nothing applied")
    return over


def decide(items=HERE):
    base = by_index(items / "csqa_review.csv")
    mine = by_index(items / "csqa_review_mine.csv")
    ep = by_index(items / "csqa_review_claude_pass.csv")
    cl = by_index(items / "csqa_review_claude.csv")
    reviewed = bdf.read_rows(items / "csqa_disagreements_reviewed_by_author.csv")
    validate(reviewed, bdf.read_rows(items / "csqa_disagreements.csv"))
    dis = {int(r["draw_index"]): r for r in reviewed}
    is_d = lambda r: bool(r["drop"].strip())
    out = {}
    for d in range(bdf.N_ROWS):
        r = base[d]
        if bde.has_duplicate_options(r):
            out[d] = ("x", bde.MECH_REASON, "mechanical")
        elif is_d(mine[d]):
            out[d] = ("x", mine[d]["reason"].strip(), "author")
        elif d in dis:
            if dis[d]["final_drop"] == "x":
                reason = (cl[d]["reason"].strip() if is_d(cl[d]) else "") or \
                         (ep[d]["reason"].strip() if d in ep and is_d(ep[d]) else "") or AUTHOR_REASON
                out[d] = ("x", reason, "author-final")
            else:
                out[d] = ("", "", "author-final")
        else:
            if d < 200:
                if is_d(ep[d]) != is_d(cl[d]):
                    sys.exit(f"draw_index {d}: passes differ but the row is not in the disagreement file")
                src = "claude-agreed"
            else:
                src = "claude-only"
            out[d] = (("x", cl[d]["reason"].strip(), src) if is_d(cl[d]) else ("", "", src))
    for d, (drop, reason, _) in out.items():
        if drop and not reason:
            sys.exit(f"draw_index {d}: dropped without a reason")
    return base, out


if __name__ == "__main__":
    base, out = decide()
    rows = [dict(base[d]) for d in range(bdf.N_ROWS)]
    path = HERE / "csqa_review.csv"
    raw = path.read_bytes().decode("utf-8")
    assert bde.serialize([dict(base[d]) for d in range(bdf.N_ROWS)]) == raw, "csqa_review.csv does not round-trip; refusing"
    for d, (drop, reason, _) in out.items():
        rows[d]["drop"], rows[d]["reason"] = drop, reason
    path.write_bytes(bde.serialize(rows).encode("utf-8"))
    after = by_index(path)
    assert all(bdf.row_hash(after[d]) == bdf.row_hash(base[d]) for d in range(bdf.N_ROWS)), "a non-drop column changed"
    prov = {str(d): src for d, (_, _, src) in out.items()}
    (HERE / "csqa_review_provenance.json").write_text(json.dumps(
        {"sources": {"mechanical": "identical option text, marked by build_D_extend.py",
                     "author": "author's own mark in csqa_review_mine.csv",
                     "author-final": "disagreement-file row, final_drop from the author-reviewed copy",
                     "claude-agreed": "draw_index 0-199, both Claude passes agree",
                     "claude-only": "draw_index 200-349, the Claude pass alone"},
         "per_draw_index": prov}, indent=1) + chr(10), encoding="utf-8")
    print("decisions by source:", dict(collections.Counter(prov.values())))
    surv = [d for d, (drop, _, _) in out.items() if not drop]
    print(f"drops {bdf.N_ROWS - len(surv)}, survivors {len(surv)}; 150th survivor draw_index {surv[149]}")
    first = surv[:150]
    print("first 150 survivors by source:", dict(collections.Counter(prov[str(d)] for d in first)),
          "| from draw_index >= 200:", sum(d >= 200 for d in first))
