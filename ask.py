"""Ask GER-3620P questions from the terminal.

    python ask.py "What are the three main types of inspection?"
    python ask.py                       # interactive
    python ask.py --file questions.txt --report answers.md --screenshots screenshots
"""
import argparse
import re
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from config import GROQ_MODEL
from rag import Answer, answer, cite


def render(console: Console, ans: Answer, show_context: bool = False) -> None:
    console.print(Panel(Text(ans.question, style="bold"), title="Question", title_align="left", border_style="cyan"))
    body = Text(ans.text, style="bold red" if not ans.found else "")
    if ans.found:
        body.append("\n\nReference" + ("s" if len(ans.references) > 1 else "") + ":", style="bold green")
        for i, doc in enumerate(ans.references, 1):
            body.append(f"\n  [{i}] {cite(doc)}", style="green")
    elif ans.note:
        body.append(f"\n\n{ans.note}", style="dim")
    engine = f"Groq {GROQ_MODEL}" if ans.mode == "llm" else "extractive (no GROQ_API_KEY)"
    console.print(Panel(body, title="Answer", title_align="left", subtitle=engine, border_style="green" if ans.found else "red"))

    if show_context:
        table = Table("score", "retrieved excerpt (after rerank)", show_lines=False)
        for doc, score in ans.candidates:
            table.add_row(f"{score:.2f}", cite(doc))
        console.print(table)


def write_report(answers: list[Answer], path: Path) -> None:
    lines = ["| # | Question | Answer | Reference |", "|---|---|---|---|"]
    for i, a in enumerate(answers, 1):
        refs = "<br>".join(f"[{j}] {cite(d)}" for j, d in enumerate(a.references, 1)) or "-"
        text = re.sub(r"\s+", " ", a.text).replace("|", "\\|")
        lines.append(f"| {i} | {a.question} | {text} | {refs} |")
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="?", help="question to ask (omit for interactive mode)")
    parser.add_argument("--file", type=Path, help="text file with one question per line")
    parser.add_argument("--report", type=Path, help="write a Markdown table of the answers here")
    parser.add_argument("--screenshots", type=Path, help="save one terminal capture (SVG) per question here")
    parser.add_argument("--show-context", action="store_true", help="also list the retrieved excerpts and scores")
    args = parser.parse_args()

    console = Console()
    if args.file:
        questions = [q.strip() for q in args.file.read_text().splitlines() if q.strip()]
        answers = []
        if args.screenshots:
            args.screenshots.mkdir(parents=True, exist_ok=True)
        for i, q in enumerate(questions, 1):
            ans = answer(q)
            answers.append(ans)
            shot = Console(record=True, width=100)
            shot.print(Text(f'$ python ask.py "{q}"', style="bold"))
            render(shot, ans, args.show_context)
            if args.screenshots:
                shot.save_svg(str(args.screenshots / f"q{i}.svg"), title=f"Question {i}")
        if args.report:
            write_report(answers, args.report)
            console.print(f"Report written to {args.report}")
    elif args.question:
        render(console, answer(args.question), args.show_context)
    else:
        console.print("[bold]GER-3620P maintenance assistant[/bold] - type a question, or 'exit'.")
        while (q := console.input("\n[cyan]>[/cyan] ").strip()).lower() not in {"exit", "quit"}:
            if q:
                render(console, answer(q), args.show_context)


if __name__ == "__main__":
    main()
