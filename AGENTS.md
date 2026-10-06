# Agent instructions

This repository is the RAG Evaluation Workbench. Before changing product behavior, read the README and inspect the current implementation. Local product notes and OpenSpec plans may be intentionally absent from a checkout.

When an active local OpenSpec change is available, follow its propose → review → apply → verify → archive workflow. Keep its proposal, design, specs, and tasks aligned; mark tasks as work progresses. Do not archive until implementation and verification are complete.

V1 is a local single-user React/TypeScript + FastAPI app backed by PostgreSQL/pgvector and Docker Compose. V1's deployment scope does not define the product's long-term boundary: this is the first release of a broader RAG engineering workbench, with a future public release in view. Keep the architecture modular so capabilities and deployment can grow from measured user needs. Use OpenAI models through OpenRouter with OPENROUTER_API_KEY and environment-configured model IDs, defaulting to `openai/text-embedding-3-small` and `openai/gpt-5-nano`. Preserve saved-run provenance when documents are deleted.
