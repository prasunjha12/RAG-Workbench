from __future__ import annotations

import json
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import psycopg
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from . import provider
from pydantic import BaseModel, Field
from pypdf import PdfReader
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector

DB = os.getenv("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag_workbench")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
MAX_UPLOAD = int(os.getenv("MAX_UPLOAD_MB", "25")) * 1024 * 1024
UPLOAD_DIR = Path("/app/uploads")


def connect():
    conn = psycopg.connect(DB, row_factory=dict_row)
    register_vector(conn)
    return conn


def init_db():
    with psycopg.connect(DB) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
          id TEXT PRIMARY KEY, filename TEXT NOT NULL, file_type TEXT NOT NULL,
          uploaded_at TIMESTAMPTZ NOT NULL, status TEXT NOT NULL, error TEXT,
          chunk_size INT NOT NULL, chunk_overlap INT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chunks (
          id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
          chunk_index INT NOT NULL, text TEXT NOT NULL, page INT, embedding vector(1536) NOT NULL
        );
        CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists=100);
        CREATE TABLE IF NOT EXISTS cases (
          id TEXT PRIMARY KEY, name TEXT NOT NULL, question TEXT NOT NULL, expected_answer TEXT NOT NULL,
          relevant_chunk_ids JSONB NOT NULL DEFAULT '[]', created_at TIMESTAMPTZ NOT NULL
        );
        CREATE TABLE IF NOT EXISTS retrievals (
          id TEXT PRIMARY KEY, question TEXT NOT NULL, document_ids JSONB NOT NULL, top_k INT NOT NULL,
          config JSONB NOT NULL, retrieved JSONB NOT NULL, retrieval_ms DOUBLE PRECISION NOT NULL,
          created_at TIMESTAMPTZ NOT NULL
        );
        CREATE TABLE IF NOT EXISTS runs (
          id TEXT PRIMARY KEY, retrieval_id TEXT, question TEXT NOT NULL, document_snapshot JSONB NOT NULL,
          config JSONB NOT NULL, retrieved JSONB NOT NULL, answer TEXT NOT NULL, metrics JSONB NOT NULL,
          timings JSONB NOT NULL, rating INT CHECK (rating BETWEEN 1 AND 5), created_at TIMESTAMPTZ NOT NULL
        );
        """)


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title="RAG Evaluation Workbench", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def split_text(text: str, size: int, overlap: int) -> list[str]:
    if size < 100 or overlap < 0 or overlap >= size:
        raise HTTPException(400, "Chunk size must be at least 100 and overlap must be nonnegative and smaller than size.")
    parts, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            floor = start + size // 2
            cut = max(text.rfind("\n\n", floor, end), text.rfind(". ", floor, end), text.rfind(" ", floor, end))
            if cut > start:
                end = cut + (1 if text[cut] == " " else 2 if text[cut:cut+2] == ". " else 0)
        piece = text[start:end].strip()
        if piece:
            parts.append(piece)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return parts


@app.get("/api/health")
def health():
    try:
        with connect() as conn:
            conn.execute("SELECT 1")
        return {"status": "ok", "provider": "openrouter", "provider_configured": provider.configured()}
    except Exception as exc:
        raise HTTPException(503, f"Database unavailable: {exc}") from exc


@app.get("/api/config")
def get_config():
    return {"chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP, **provider.configuration()}


@app.get("/api/documents")
def list_documents():
    with connect() as conn:
        return conn.execute("SELECT id,filename,file_type,uploaded_at,status,error,chunk_size,chunk_overlap,(SELECT count(*) FROM chunks WHERE document_id=documents.id) AS chunk_count FROM documents ORDER BY uploaded_at DESC").fetchall()


@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...), chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in {".txt", ".pdf"}:
        raise HTTPException(415, "Only TXT and PDF files are supported.")
    data = await file.read(MAX_UPLOAD + 1)
    if not data or len(data) > MAX_UPLOAD:
        raise HTTPException(400, f"File must be nonempty and no larger than {MAX_UPLOAD // (1024*1024)} MB.")
    if chunk_size < 100 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise HTTPException(400, "Invalid chunk size or overlap.")
    doc_id, filename = str(uuid.uuid4()), Path(file.filename or "upload").name
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_path = UPLOAD_DIR / f"{doc_id}{ext}"
    file_path.write_bytes(data)
    now = datetime.now(timezone.utc)
    try:
        if ext == ".txt":
            pages = [(None, data.decode("utf-8-sig").strip())]
        else:
            reader = PdfReader(str(file_path))
            pages = [(i + 1, (page.extract_text() or "").strip()) for i, page in enumerate(reader.pages)]
        chunks = [{"id": str(uuid.uuid4()), "page": page, "text": piece} for page, text in pages for piece in split_text(text, chunk_size, chunk_overlap)]
        if not chunks:
            raise ValueError("No extractable text was found. Scanned PDFs without a text layer are not supported.")
        vectors = []
        for offset in range(0, len(chunks), 100):
            vectors.extend(provider.embed([c["text"] for c in chunks[offset:offset + 100]]))
        with connect() as conn:
            conn.execute("INSERT INTO documents VALUES (%s,%s,%s,%s,'processed',NULL,%s,%s)", (doc_id, filename, ext[1:], now, chunk_size, chunk_overlap))
            for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
                conn.execute("INSERT INTO chunks VALUES (%s,%s,%s,%s,%s,%s)", (chunk["id"], doc_id, i, chunk["text"], chunk["page"], vector))
        return {"id": doc_id, "filename": filename, "status": "processed", "chunk_count": len(chunks)}
    except HTTPException as exc:
        with connect() as conn:
            conn.execute("INSERT INTO documents VALUES (%s,%s,%s,%s,'error',%s,%s,%s) ON CONFLICT (id) DO UPDATE SET status='error',error=EXCLUDED.error", (doc_id, filename, ext[1:], now, str(exc.detail), chunk_size, chunk_overlap))
        raise
    except Exception as exc:
        with connect() as conn:
            conn.execute("INSERT INTO documents VALUES (%s,%s,%s,%s,'error',%s,%s,%s) ON CONFLICT (id) DO UPDATE SET status='error',error=EXCLUDED.error", (doc_id, filename, ext[1:], now, str(exc), chunk_size, chunk_overlap))
        raise HTTPException(502 if not isinstance(exc, (ValueError, UnicodeDecodeError)) else 422, str(exc)) from exc


@app.get("/api/documents/{doc_id}/chunks")
def document_chunks(doc_id: str):
    with connect() as conn:
        rows = conn.execute("SELECT c.id,c.document_id,c.chunk_index,c.text,c.page,d.filename FROM chunks c JOIN documents d ON d.id=c.document_id WHERE c.document_id=%s ORDER BY c.chunk_index", (doc_id,)).fetchall()
    if not rows:
        raise HTTPException(404, "Document or chunks not found.")
    return rows


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    with connect() as conn:
        cur = conn.execute("DELETE FROM documents WHERE id=%s", (doc_id,))
    if not cur.rowcount:
        raise HTTPException(404, "Document not found.")
    return {"deleted": doc_id}


class RetrievalRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10000)
    document_ids: list[str] = []
    top_k: int = Field(default=5, ge=1, le=50)


@app.post("/api/retrievals")
def retrieve(request: RetrievalRequest):
    started = time.perf_counter()
    try:
        vector = provider.embed([request.question])[0]
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, f"Query embedding failed: {exc}") from exc
    with connect() as conn:
        if request.document_ids:
            rows = conn.execute("SELECT c.id,c.document_id,d.filename,c.chunk_index,c.text,c.page,1-(c.embedding <=> %s) AS score FROM chunks c JOIN documents d ON d.id=c.document_id WHERE c.document_id=ANY(%s) ORDER BY c.embedding <=> %s LIMIT %s", (vector, request.document_ids, vector, request.top_k)).fetchall()
            docs = conn.execute("SELECT id,filename FROM documents WHERE id=ANY(%s)", (request.document_ids,)).fetchall()
        else:
            rows = conn.execute("SELECT c.id,c.document_id,d.filename,c.chunk_index,c.text,c.page,1-(c.embedding <=> %s) AS score FROM chunks c JOIN documents d ON d.id=c.document_id ORDER BY c.embedding <=> %s LIMIT %s", (vector, vector, request.top_k)).fetchall()
            docs = conn.execute("SELECT id,filename FROM documents").fetchall()
        elapsed, result_id = (time.perf_counter() - started) * 1000, str(uuid.uuid4())
        config = {"top_k": request.top_k, **provider.configuration()}
        used_ids = [d["id"] for d in docs]
        conn.execute("INSERT INTO retrievals VALUES (%s,%s,%s,%s,%s,%s,%s,%s)", (result_id, request.question, json.dumps(used_ids), request.top_k, json.dumps(config), json.dumps(rows), elapsed, datetime.now(timezone.utc)))
    return {"id": result_id, "question": request.question, "retrieved": rows, "retrieval_ms": elapsed, "config": config, "documents": docs}


class GenerateRequest(BaseModel):
    retrieval_id: str
    case_id: str | None = None


@app.post("/api/runs")
def generate(request: GenerateRequest):
    with connect() as conn:
        retrieval = conn.execute("SELECT * FROM retrievals WHERE id=%s", (request.retrieval_id,)).fetchone()
        if not retrieval:
            raise HTTPException(404, "Retrieval result not found.")
        docs = conn.execute("SELECT id,filename,chunk_size,chunk_overlap FROM documents WHERE id=ANY(%s)", (retrieval["document_ids"],)).fetchall()
        case = conn.execute("SELECT * FROM cases WHERE id=%s", (request.case_id,)).fetchone() if request.case_id else None
    retrieved = retrieval["retrieved"]
    if not retrieved:
        raise HTTPException(422, "No context was retrieved. Add documents or broaden the selected sources before generating.")
    context = "\n\n".join(f"[{i+1}] Source: {r['filename']} (page {r['page'] or 'n/a'}, chunk {r['chunk_index']})\n{r['text']}" for i, r in enumerate(retrieved))
    started = time.perf_counter()
    try:
        answer, returned_model = provider.answer(retrieval['question'], context)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, f"Answer generation failed: {exc}") from exc
    generation_ms = (time.perf_counter() - started) * 1000
    metrics = {"recall_at_k": None, "answer_token_overlap": None}
    if case:
        relevant, found = set(case["relevant_chunk_ids"] or []), {r["id"] for r in retrieved}
        metrics["recall_at_k"] = len(relevant & found) / len(relevant) if relevant else None
        tokens = lambda s: set(re.findall(r"[a-z0-9]+", s.lower()))
        expected, actual = tokens(case["expected_answer"]), tokens(answer)
        metrics["answer_token_overlap"] = len(expected & actual) / len(expected) if expected else None
    run_id, config = str(uuid.uuid4()), retrieval["config"]
    config["generation_provider"] = "openrouter"
    config["chat_model"] = provider.CHAT_MODEL
    config["returned_chat_model"] = returned_model
    config["chunking"] = [{"document_id": d["id"], "chunk_size": d["chunk_size"], "chunk_overlap": d["chunk_overlap"]} for d in docs]
    if case:
        config["evaluation_case"] = {"id": case["id"], "name": case["name"], "question": case["question"], "expected_answer": case["expected_answer"], "relevant_chunk_ids": case["relevant_chunk_ids"]}
    timings = {"retrieval_ms": retrieval["retrieval_ms"], "generation_ms": generation_ms, "total_ms": retrieval["retrieval_ms"] + generation_ms}
    with connect() as conn:
        conn.execute("INSERT INTO runs VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,NULL,%s)", (run_id, retrieval["id"], retrieval["question"], json.dumps(docs), json.dumps(config), json.dumps(retrieved), answer, json.dumps(metrics), json.dumps(timings), datetime.now(timezone.utc)))
    return {"id": run_id, "question": retrieval["question"], "retrieved": retrieved, "answer": answer, "metrics": metrics, "timings": timings, "config": config, "documents": docs, "rating": None}


class CaseInput(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    question: str = Field(min_length=1)
    expected_answer: str = ""
    relevant_chunk_ids: list[str] = []


@app.get("/api/cases")
def list_cases():
    with connect() as conn:
        return conn.execute("SELECT * FROM cases ORDER BY created_at DESC").fetchall()


@app.post("/api/cases")
def create_case(case: CaseInput):
    case_id = str(uuid.uuid4())
    with connect() as conn:
        conn.execute("INSERT INTO cases VALUES (%s,%s,%s,%s,%s,%s)", (case_id, case.name, case.question, case.expected_answer, json.dumps(case.relevant_chunk_ids), datetime.now(timezone.utc)))
    return {"id": case_id, **case.model_dump()}


@app.put("/api/cases/{case_id}")
def update_case(case_id: str, case: CaseInput):
    with connect() as conn:
        cur = conn.execute("UPDATE cases SET name=%s,question=%s,expected_answer=%s,relevant_chunk_ids=%s WHERE id=%s", (case.name, case.question, case.expected_answer, json.dumps(case.relevant_chunk_ids), case_id))
    if not cur.rowcount:
        raise HTTPException(404, "Evaluation case not found.")
    return {"id": case_id, **case.model_dump()}


@app.delete("/api/cases/{case_id}")
def delete_case(case_id: str):
    with connect() as conn:
        cur = conn.execute("DELETE FROM cases WHERE id=%s", (case_id,))
    if not cur.rowcount:
        raise HTTPException(404, "Evaluation case not found.")
    return {"deleted": case_id}


@app.get("/api/runs")
def list_runs():
    with connect() as conn:
        return conn.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT 200").fetchall()


class RatingInput(BaseModel):
    rating: int = Field(ge=1, le=5)


@app.patch("/api/runs/{run_id}/rating")
def rate_run(run_id: str, payload: RatingInput):
    with connect() as conn:
        cur = conn.execute("UPDATE runs SET rating=%s WHERE id=%s", (payload.rating, run_id))
    if not cur.rowcount:
        raise HTTPException(404, "Run not found.")
    return {"id": run_id, "rating": payload.rating}
