"""Question answering over the GER-3620P index.

Flow:  question
         -> hybrid search: Chroma vector search (bge-small embeddings)
            + BM25 keyword search, FETCH_K candidates each
         -> cross-encoder rerank of the union (keep TOP_K)
         -> relevance gate: best score < MIN_RERANK_SCORE  => "Not found"
         -> Groq LLM, prompted to answer only from the numbered excerpts and
            cite them as [n], or reply exactly "Not found"
         -> [n] tags resolved to section + page references
"""
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache

from fastembed.rerank.cross_encoder import TextCrossEncoder
from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import (
    CHROMA_DIR,
    COLLECTION,
    EMBED_MODEL,
    FETCH_K,
    GROQ_MODEL,
    MIN_RERANK_SCORE,
    NOT_FOUND,
    RERANK_MODEL,
    TOP_K,
)

SYSTEM_PROMPT = """You are a maintenance assistant for GE heavy-duty gas turbines.
You answer ONLY from the numbered excerpts of GE document GER-3620P provided below.

Rules:
1. If the excerpts do not contain the answer, reply with exactly: Not found
   Do not use outside knowledge, even if you know the answer.
2. Cite every excerpt you rely on with its tag in square brackets, e.g. [1] or [2][4].
3. Be concise: one to four sentences, or a short list.
4. If the answer depends on the turbine model or operating conditions, open with a
   one-sentence direct answer that says so and gives the overall range. Then list the
   values exactly as the excerpts give them, one line per model family in the order
   the table gives them, covering every model in the excerpts, for example:
     - B/E-class: 6B.03 12,000 h; 7E.03 8,000 h (non-DLN) / 12,000 h (DLN)
   In interval tables, "24000/900" means 24,000 factored hours or 900 factored starts.
5. Put each citation once at the end of the sentence or list it supports, not on every line.
6. Plain text only (no Markdown). Write numbers with commas, e.g. 12,000."""

USER_PROMPT = """Excerpts:
{context}

Question: {question}
Answer:"""


# [3], 【3】 and gpt-oss style 【3†L9-L13】 all mean "excerpt 3".
CITATION_RE = re.compile(r"[\[【](\d+)(?:†[^\]】]*)?[\]】]")


@dataclass
class Answer:
    question: str
    text: str
    found: bool
    references: list[Document] = field(default_factory=list)  # references[i] is citation [i+1] in text
    candidates: list[tuple[Document, float]] = field(default_factory=list)  # reranked context
    mode: str = "llm"
    note: str = ""  # why the answer is "Not found"


@lru_cache
def get_embeddings():
    return FastEmbedEmbeddings(model_name=EMBED_MODEL)


@lru_cache
def get_vectorstore():
    if not CHROMA_DIR.exists():
        raise SystemExit("No index found. Run `python ingest.py` first.")
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())  # keeps "7f.03" as one token


@lru_cache
def get_bm25():
    """Keyword index over the same chunks, read back from Chroma."""
    data = get_vectorstore().get(include=["documents", "metadatas"])
    docs = [Document(page_content=t, metadata=m) for t, m in zip(data["documents"], data["metadatas"])]
    return BM25Retriever.from_documents(docs, k=FETCH_K, preprocess_func=_tokenize)


@lru_cache
def get_reranker():
    return TextCrossEncoder(model_name=RERANK_MODEL)


@lru_cache
def get_llm():
    if not os.getenv("GROQ_API_KEY"):
        return None
    from langchain_groq import ChatGroq

    return ChatGroq(model=GROQ_MODEL, temperature=0, max_retries=6)  # free tier: 8k tokens/min


def cite(doc: Document) -> str:
    """Location as 'Section > Subsection (Figure N), p. <printed> (PDF p. <viewer page>)'."""
    m = doc.metadata
    where = m["section"]
    if m.get("figure"):
        where += f" ({m['figure'].split('.')[0]})"  # "(Figure 36)"

    def span(a, b):
        return f"{a}" if a == b else f"{a}-{b}"

    printed = span(m["page_start"], m["page_end"])
    viewer = span(m["pdf_page_start"], m["pdf_page_end"])
    return f"{where}, p. {printed} (PDF p. {viewer})"


def excerpt_text(doc: Document) -> str:
    return doc.page_content.split("\n", 1)[1]  # drop the "[section]" line added for embedding


def retrieve(question: str) -> list[tuple[Document, float]]:
    """Vector + keyword search for recall, then a cross-encoder for precision."""
    hits = get_vectorstore().similarity_search(question, k=FETCH_K) + get_bm25().invoke(question)
    candidates = list({d.page_content: d for d in hits}.values())
    scores = list(get_reranker().rerank(question, [d.page_content for d in candidates]))
    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    return ranked[:TOP_K]


def _generate(question: str, ranked: list[tuple[Document, float]]) -> tuple[str, list[Document]]:
    docs = [d for d, _ in ranked]
    context = "\n\n".join(f"[{i}] ({cite(d)})\n{excerpt_text(d)}" for i, d in enumerate(docs, 1))
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", USER_PROMPT)])
    chain = prompt | get_llm() | StrOutputParser()
    text = chain.invoke({"context": context, "question": question}).strip()

    if re.match(r"^\W*not found\b", text, re.IGNORECASE):
        return NOT_FOUND, []

    # Renumber citations 1..k in order of first use so they line up with the reference
    # list; excerpts that resolve to the same section and page share one number.
    labels: list[str] = []
    refs: list[Document] = []
    number: dict[int, int] = {}
    for n in (int(m) for m in CITATION_RE.findall(text)):
        if 1 <= n <= len(docs) and n not in number:
            label = cite(docs[n - 1])
            if label not in labels:
                labels.append(label)
                refs.append(docs[n - 1])
            number[n] = labels.index(label) + 1
    if not refs:
        return f"{text} [1]", docs[:1]
    text = CITATION_RE.sub(lambda m: f"[{number[int(m[1])]}]" if int(m[1]) in number else "", text)
    text = text.replace("**", "").replace("\u2011", "-").replace("\u202f", " ").replace("\u00a0", " ")
    return _tidy_citations(text), refs


def _tidy_citations(text: str) -> str:
    """'A [1]; B [1]; C [1]' -> 'A; B; C [1]' (one tag per line instead of one per item)."""
    text = re.sub(r"(?<=[^\s\[\]])\[(\d+)\]", r" [\1]", text)  # "24,000 h[1]" -> "24,000 h [1]"
    tags = re.findall(r"\[\d+\]", text)
    if len(set(tags)) == 1 and len(tags) > 1:  # everything comes from one excerpt: cite it once
        return re.sub(r"\s*\[\d+\]", "", text).rstrip() + " " + tags[0]
    lines = []
    for line in text.splitlines():
        tags = re.findall(r"\[\d+\]", line)
        unique = list(dict.fromkeys(tags))
        if len(tags) > len(unique):
            line = re.sub(r"\s*\[\d+\]", "", line).rstrip(" ;,") + " " + "".join(unique)
        lines.append(line.rstrip())
    return "\n".join(lines)


def _extract(question: str, ranked: list[tuple[Document, float]]) -> tuple[str, list[Document]]:
    """No LLM available: return the best-matching sentences of the top excerpt."""
    best = ranked[0][0]
    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n", excerpt_text(best)) if len(s.split()) > 3]
    scores = list(get_reranker().rerank(question, sentences))
    top = sorted(sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)[:2])
    return " ".join(sentences[i] for i in top) + " [1]", [best]


def answer(question: str) -> Answer:
    ranked = retrieve(question)
    mode = "llm" if get_llm() else "extractive"
    if not ranked or ranked[0][1] < MIN_RERANK_SCORE:
        best = f"best match {ranked[0][1]:.2f}: {cite(ranked[0][0])}" if ranked else "index is empty"
        note = f"No passage passed the relevance gate (score >= {MIN_RERANK_SCORE}; {best})."
        return Answer(question, NOT_FOUND, False, [], ranked, mode, note)

    text, refs = (_generate if mode == "llm" else _extract)(question, ranked)
    found = text != NOT_FOUND
    note = "" if found else f"The LLM found no answer in the top {len(ranked)} excerpts."
    return Answer(question, text, found, refs, ranked, mode, note)
