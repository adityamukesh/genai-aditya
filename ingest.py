"""Build the Chroma index: PDF -> section-aware chunks -> embeddings -> chroma_db/."""
import shutil
from collections import Counter

from langchain_chroma import Chroma

from config import CHROMA_DIR, COLLECTION, PDF_PATH
from pdf_loader import load_documents
from rag import get_embeddings


def main():
    if not PDF_PATH.exists():
        raise SystemExit(f"PDF not found at {PDF_PATH}. See README > Setup.")

    print(f"Parsing {PDF_PATH.name} ...")
    docs = load_documents(PDF_PATH)
    kinds = Counter(d.metadata["kind"] for d in docs)
    print(f"  {len(docs)} chunks ({kinds['text']} text, {kinds['table']} table)")

    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)  # rebuild from scratch so re-runs never duplicate chunks
    print("Embedding and writing to ChromaDB ...")
    Chroma.from_documents(
        docs,
        get_embeddings(),
        ids=[f"chunk-{i:04d}" for i in range(len(docs))],
        collection_name=COLLECTION,
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"Done. Index stored in {CHROMA_DIR.relative_to(CHROMA_DIR.parent)}/")


if __name__ == "__main__":
    main()
