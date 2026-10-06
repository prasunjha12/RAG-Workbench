# RAG Evaluation Workbench

A local workspace for exploring a practical RAG question: **why did the system produce this answer, and how can I make it better?** Add a document, inspect its chunks, see what retrieval finds, generate a cited answer, and save the run for review.

This first version is designed for one developer running the app locally. You can:

- Upload text-based PDF and TXT files and inspect their chunks.
- Search semantically and inspect ranked passages and similarity scores before generation.
- Generate answers grounded in retrieved context with source citations.
- Create reusable evaluation cases, view basic scores, save run history, and rate answers.

## Run it locally

You’ll need Docker Desktop with Docker Compose and an OpenRouter API key.

1. Create a local environment file if you don’t already have one:

   ```powershell
   Copy-Item .env.example .env
   ```

2. Open `.env` and set `OPENROUTER_API_KEY` to your key. The default models are `openai/text-embedding-3-small` for embeddings and `openai/gpt-5-nano` for answers. Model usage is billed through OpenRouter; check current pricing and your account credits.
3. Start the app from this folder:

   ```powershell
   docker compose up --build -d
   ```

4. Open [the workbench](http://localhost:5173). The API is at [localhost:8000](http://localhost:8000), with interactive API docs at [localhost:8000/docs](http://localhost:8000/docs).

Stop the app with `docker compose down`. Your documents and database remain in Docker’s local volumes and are available the next time you start it.

## Where your data goes

Uploaded files and the PostgreSQL database are kept in persistent Docker volumes on your machine. There is no cloud bucket configured. The app sends extracted text to OpenRouter for embeddings and retrieved passages to OpenRouter to generate answers. Use documents you’re comfortable sending to the model provider.

Your API key belongs in your local `.env` file. Keep it private; Git excludes that file.

This version supports PDFs with an extractable text layer; it does not run OCR on scanned PDFs. Chunking, evaluation, and run-history behavior are still being refined.