"""Paths, model names and retrieval knobs shared by ingest.py and rag.py."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

PDF_PATH = ROOT / "data" / "GER-3620-P.pdf"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION = "ger3620p"

# Local ONNX models via FastEmbed (no GPU / PyTorch needed).
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"

# LLM served by Groq (free tier). Override with GROQ_MODEL in .env.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-large-v3-turbo")  # speech-to-text for voice questions

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TABLE_CHUNK_SIZE = 1500

# Retrieval
FETCH_K = 15  # candidates pulled from Chroma by vector similarity
TOP_K = 8  # best candidates (after cross-encoder reranking) shown to the LLM
# Cross-encoder logit below which nothing in the PDF is considered relevant.
# Questions under this bar get "Not found" without calling the LLM; anything
# above it still has to be answered from the excerpts or the LLM says "Not found".
MIN_RERANK_SCORE = -8.0

NOT_FOUND = "Not found"
