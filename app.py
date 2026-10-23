"""Web UI for the GER-3620P maintenance assistant.

    streamlit run app.py        ->  http://localhost:8501
"""
import html
import math
import re
import time

import streamlit as st

from config import EMBED_MODEL, GROQ_MODEL, MIN_RERANK_SCORE, ROOT, TOP_K
from rag import Answer, answer, cite, excerpt_text, get_bm25, get_llm, get_reranker, get_vectorstore

REPO_URL = "https://github.com/adityamukesh/genai-aditya"
PDF_URL = (
    "https://www.gevernova.com/content/dam/gepower-new/global/en_US/downloads/"
    "gas-new-site/resources/reference/GER-3620-P.pdf"
)
EXAMPLES = [q for q in (ROOT / "questions.txt").read_text().splitlines() if q.strip()]
EXAMPLE_ICONS = [":material/search:", ":material/calculate:", ":material/payments:", ":material/science:", ":material/schedule:"]

st.set_page_config(
    page_title="Gas Turbine Maintenance Assistant",
    page_icon=":material/precision_manufacturing:",
    layout="centered",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
.block-container { padding-top: 3.6rem; padding-bottom: 6rem; max-width: 820px; }
[data-testid="stSidebar"] button[kind="tertiary"] { justify-content: flex-start; text-align: left; padding: 4px 0; }
[data-testid="stSidebar"] button[kind="tertiary"] p { text-align: left; font-size: .9rem; }
[data-testid="stSidebar"] button[kind="tertiary"]:hover p { color: #0b7a75; }

.hero {
  background: linear-gradient(135deg, #064e4b 0%, #0b7a75 55%, #13a39b 100%);
  color: #fff; border-radius: 18px; padding: 26px 28px 22px; margin-bottom: 18px;
  box-shadow: 0 10px 30px -12px rgba(6, 78, 75, .55);
}
.hero h1 { color: #fff; font-size: 1.85rem; line-height: 1.2; margin: 0 0 6px; padding: 0; font-weight: 700; }
.hero p { color: rgba(255,255,255,.88); margin: 0 0 14px; font-size: 1rem; }
.hero .eyebrow { text-transform: uppercase; letter-spacing: .12em; font-size: .72rem; font-weight: 600;
  color: rgba(255,255,255,.75); margin-bottom: 6px; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; }
.chip { background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.25); color: #fff;
  border-radius: 999px; padding: 4px 12px; font-size: .8rem; white-space: nowrap; }

.section-label { font-size: .78rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase;
  opacity: .6; margin: 6px 0 8px; }

sup.cite { background: rgba(11,122,117,.14); color: #0b7a75; border-radius: 6px; padding: 0 5px;
  font-size: .72rem; font-weight: 700; margin-left: 2px; }

.sources { margin-top: 12px; padding-top: 10px; border-top: 1px solid rgba(128,128,128,.22); }
.src { display: flex; gap: 10px; align-items: baseline; padding: 6px 0; font-size: .9rem; }
.src .n { flex: none; min-width: 26px; text-align: center; background: #0b7a75; color: #fff; border-radius: 6px;
  font-size: .75rem; font-weight: 700; padding: 1px 6px; }
.src .where { font-weight: 600; }
.src .page { opacity: .7; white-space: nowrap; }

.notfound { border: 1px solid rgba(220,38,38,.35); background: rgba(220,38,38,.07); border-radius: 14px;
  padding: 14px 16px; }
.notfound .t { font-weight: 700; color: #dc2626; font-size: 1.05rem; margin-bottom: 2px; }
.notfound .d { opacity: .8; font-size: .92rem; }

.meta { display: flex; flex-wrap: wrap; gap: 14px; font-size: .78rem; opacity: .6; margin-top: 10px; }

.gate { font-size: .85rem; padding: 8px 12px; border-radius: 10px; background: rgba(128,128,128,.08); margin-bottom: 10px; }
.ex { border: 1px solid rgba(128,128,128,.2); border-radius: 12px; padding: 10px 12px; margin-bottom: 10px; }
.ex .head { display: flex; justify-content: space-between; gap: 12px; align-items: center; font-size: .85rem; }
.ex .bar { height: 6px; background: rgba(128,128,128,.18); border-radius: 99px; margin: 8px 0; overflow: hidden; }
.ex .bar > span { display: block; height: 100%; background: linear-gradient(90deg, #13a39b, #0b7a75); }
.ex .txt { white-space: pre-wrap; font-size: .8rem; opacity: .85; max-height: 160px; overflow: auto;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.score { font-variant-numeric: tabular-nums; font-weight: 700; }

[data-testid="stSidebar"] .side-title { font-weight: 700; font-size: 1.05rem; margin-bottom: 2px; }
.status { display: inline-flex; align-items: center; gap: 8px; font-size: .85rem; padding: 6px 10px;
  border-radius: 999px; background: rgba(11,122,117,.10); }
.status .dot { width: 8px; height: 8px; border-radius: 50%; background: #16a34a; box-shadow: 0 0 0 3px rgba(22,163,74,.2); }
.status.warn .dot { background: #d97706; box-shadow: 0 0 0 3px rgba(217,119,6,.2); }
.steps { counter-reset: s; padding-left: 0; list-style: none; margin: 0; }
.steps li { counter-increment: s; position: relative; padding-left: 30px; margin-bottom: 10px; font-size: .88rem; }
.steps li::before { content: counter(s); position: absolute; left: 0; top: 0; width: 20px; height: 20px;
  border-radius: 50%; background: #0b7a75; color: #fff; font-size: .72rem; font-weight: 700;
  display: flex; align-items: center; justify-content: center; }
</style>
"""


@st.cache_resource(show_spinner="Loading the GER-3620P index and models...")
def warm_up() -> int:
    """Load the vector store, BM25 index, reranker and LLM client once per server."""
    get_bm25()
    get_reranker()
    get_llm()
    return len(get_vectorstore().get(include=[])["ids"])


def source_line(i: int, doc) -> str:
    where, _, pages = cite(doc).rpartition(", p. ")
    return (
        f'<div class="src"><span class="n">{i}</span>'
        f'<span><span class="where">{html.escape(where.replace(" > ", " › "))}</span> '
        f'<span class="page">· p. {html.escape(pages)}</span></span></div>'
    )


def answer_html(text: str) -> str:
    text = html.escape(text).replace("$", "\\$")
    return re.sub(r"\[(\d+)\]", r'<sup class="cite">\1</sup>', text)


def relevance(score: float) -> float:
    return 1 / (1 + math.exp(-score / 2))  # cross-encoder logit -> 0..1 for the bar


def render_answer(ans: Answer, seconds: float) -> None:
    if ans.found:
        st.markdown(answer_html(ans.text), unsafe_allow_html=True)
        label = "Source" if len(ans.references) == 1 else "Sources"
        lines = "".join(source_line(i, d) for i, d in enumerate(ans.references, 1))
        st.markdown(
            f'<div class="sources"><div class="section-label">{label} in GER-3620P</div>{lines}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="notfound"><div class="t">Not found</div>'
            '<div class="d">GER-3620P does not contain the answer to this question, so the assistant '
            "does not guess.</div></div>",
            unsafe_allow_html=True,
        )

    engine = f"Groq · {GROQ_MODEL}" if ans.mode == "llm" else "Extractive mode (no API key)"
    st.markdown(
        f'<div class="meta"><span>⏱ {seconds:.1f} s</span><span>◆ {html.escape(engine)}</span>'
        f"<span>▤ {len(ans.candidates)} excerpts ranked</span></div>",
        unsafe_allow_html=True,
    )

    with st.expander("How this answer was found", icon=":material/manage_search:"):
        best = ans.candidates[0][1] if ans.candidates else float("-inf")
        if best >= MIN_RERANK_SCORE:
            gate = f"Relevance gate passed: best score <b>{best:.2f}</b> ≥ {MIN_RERANK_SCORE}, so the top excerpts were sent to the LLM."
        else:
            gate = f"Relevance gate blocked: best score <b>{best:.2f}</b> &lt; {MIN_RERANK_SCORE}, so nothing relevant exists and the LLM was not called."
        st.markdown(f'<div class="gate">{gate}</div>', unsafe_allow_html=True)
        for doc, score in ans.candidates:
            st.markdown(
                f'<div class="ex"><div class="head"><span>{html.escape(cite(doc).replace(" > ", " › "))}</span>'
                f'<span class="score">{score:.2f}</span></div>'
                f'<div class="bar"><span style="width:{relevance(score) * 100:.0f}%"></span></div>'
                f'<div class="txt">{html.escape(excerpt_text(doc)[:900])}</div></div>',
                unsafe_allow_html=True,
            )


def ask(question: str) -> None:
    st.session_state.pending = question


def sidebar(chunks: int) -> None:
    with st.sidebar:
        st.markdown('<div class="side-title">GER-3620P Assistant</div>', unsafe_allow_html=True)
        st.caption("GE heavy-duty gas turbine O&M, Rev P")
        if get_llm():
            st.markdown(f'<span class="status"><span class="dot"></span>Groq · {GROQ_MODEL}</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status warn"><span class="dot"></span>Extractive mode: no GROQ_API_KEY</span>', unsafe_allow_html=True)
        st.write("")
        if st.button("New chat", icon=":material/add_comment:", width="stretch", type="primary"):
            st.session_state.history = []
            st.rerun()

        st.markdown('<div class="section-label">Assignment questions</div>', unsafe_allow_html=True)
        for i, q in enumerate(EXAMPLES):
            st.button(q, key=f"side-{i}", on_click=ask, args=(q,), width="stretch", type="tertiary")

        st.divider()
        st.markdown('<div class="section-label">How it works</div>', unsafe_allow_html=True)
        st.markdown(
            '<ol class="steps">'
            f"<li>The 60-page PDF is split into {chunks} chunks that keep their <b>section</b> and <b>page</b>; tables stay readable.</li>"
            f"<li><b>Hybrid search</b>: {EMBED_MODEL.split('/')[-1]} vectors in ChromaDB + BM25 keywords.</li>"
            "<li>A <b>cross-encoder</b> reranks the candidates.</li>"
            f"<li>A <b>relevance gate</b> answers <i>Not found</i> when nothing scores above {MIN_RERANK_SCORE}.</li>"
            f"<li>The <b>LLM</b> answers only from the top {TOP_K} excerpts; references come from chunk metadata.</li>"
            "</ol>",
            unsafe_allow_html=True,
        )
        st.divider()
        st.link_button("Source code on GitHub", REPO_URL, icon=":material/code:", width="stretch")
        st.link_button("Official GER-3620P PDF", PDF_URL, icon=":material/picture_as_pdf:", width="stretch")


def main() -> None:
    st.html(CSS)
    chunks = warm_up()
    st.session_state.setdefault("history", [])
    sidebar(chunks)

    st.markdown(
        f"""<div class="hero">
          <div class="eyebrow">RAG · Python · LangChain · ChromaDB</div>
          <h1>Gas Turbine Maintenance Assistant</h1>
          <p>Ask about operating and maintaining GE heavy-duty gas turbines. Every answer comes from
          GE document GER-3620P and cites the section and page; if the document doesn't say, you get <b>Not found</b>.</p>
          <div class="chips">
            <span class="chip">📄 60-page GE manual</span><span class="chip">🧩 {chunks} indexed chunks</span>
            <span class="chip">🔎 Hybrid search + rerank</span><span class="chip">⚡ Groq LLM</span>
          </div></div>""",
        unsafe_allow_html=True,
    )

    question = st.chat_input("Ask about inspections, intervals, maintenance factors...")
    question = st.session_state.pop("pending", None) or question

    if not st.session_state.history and not question:
        st.markdown('<div class="section-label">Try a question</div>', unsafe_allow_html=True)
        cols = st.columns(2)
        for i, q in enumerate(EXAMPLES):
            cols[i % 2].button(q, key=f"ex-{i}", on_click=ask, args=(q,), icon=EXAMPLE_ICONS[i % len(EXAMPLE_ICONS)], width="stretch")

    for turn in st.session_state.history:
        with st.chat_message("user", avatar=":material/person:"):
            st.markdown(html.escape(turn["question"]))
        with st.chat_message("assistant", avatar=":material/precision_manufacturing:"):
            render_answer(turn["answer"], turn["seconds"])

    if question:
        with st.chat_message("user", avatar=":material/person:"):
            st.markdown(html.escape(question))
        with st.chat_message("assistant", avatar=":material/precision_manufacturing:"):
            with st.spinner("Searching GER-3620P..."):
                start = time.perf_counter()
                result = answer(question)
                seconds = time.perf_counter() - start
            render_answer(result, seconds)
        st.session_state.history.append({"question": question, "answer": result, "seconds": seconds})


main()
