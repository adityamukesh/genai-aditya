"""Regression check: answerable questions must contain the right facts and cite the
right printed page; unanswerable ones must return exactly "Not found".

    python eval.py            # prints a table, writes eval_results.md
"""
from pathlib import Path

from rich.console import Console
from rich.table import Table

from config import NOT_FOUND
from rag import answer, cite

# (question, facts the answer must contain, printed page the answer must cite)
# facts=None means the document does not answer it -> expect "Not found".
CASES = [
    ("What are the three main types of inspection?", ["standby", "running", "disassembly"], 21),
    ("What does a maintenance factor of two mean?", ["half"], 5),
    ("What is the purchase price of this gas turbine?", None, None),
    ("What is the melting point of titanium?", None, None),
    ("What is the recommended combustion-inspection interval in hours?", ["8,000", "12,000", "24,000", "32,000"], 29),
    ("What is the hot gas path inspection interval for a 7F.03?", ["24,000"], 29),
    ("When is cool-down considered complete before an inspection?", ["150"], 22),
    ("What trip severity factor applies to a trip from full load?", ["8:1"], 11),
    ("Who is the CEO of GE Vernova?", None, None),
    ("What is the warranty period for the turbine?", None, None),
    ("What is the capital of France?", None, None),
]


def check(ans, facts, page) -> tuple[bool, str]:
    if facts is None:
        return ans.text == NOT_FOUND, "expects Not found"
    if not ans.found:
        return False, "answered Not found"
    missing = [f for f in facts if f.lower() not in ans.text.lower()]
    if missing:
        return False, f"missing {missing}"
    if not any(d.metadata["page_start"] <= page <= d.metadata["page_end"] for d in ans.references):
        return False, f"no citation of p. {page}"
    return True, f"facts present, cites p. {page}"


def main():
    rows, passed = [], 0
    for question, facts, page in CASES:
        ans = answer(question)
        ok, why = check(ans, facts, page)
        passed += ok
        refs = "; ".join(cite(d) for d in ans.references) or "-"
        rows.append((question, "answerable" if facts else "not in PDF", ans.text if not ans.found else refs, ok, why))

    table = Table("Question", "Type", "Result / reference", "Pass", title=f"{passed}/{len(CASES)} passed")
    for q, kind, result, ok, why in rows:
        table.add_row(q, kind, result, ("[green]PASS[/green] " if ok else "[red]FAIL[/red] ") + why)
    Console().print(table)

    lines = [f"**{passed}/{len(CASES)} passed**", "", "| Question | Type | Result / reference | Check |", "|---|---|---|---|"]
    for q, kind, result, ok, why in rows:
        lines.append(f"| {q} | {kind} | {result} | {'PASS' if ok else 'FAIL'}: {why} |")
    Path("eval_results.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
