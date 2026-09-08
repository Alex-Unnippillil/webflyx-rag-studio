# Webflyx RAG Studio

Webflyx RAG Studio is a Windows-first desktop application for experimenting with modern retrieval-augmented generation on a movie corpus. It combines lexical search, semantic search, reciprocal-rank fusion, optional cross-encoder reranking, multimodal CLIP search, and grounded LLM generation in one interface.

## Highlights

- **Hybrid retrieval** — BM25 + SentenceTransformer semantic search + RRF
- **Three search modes** — Hybrid, BM25-only, and semantic-only
- **Optional reranking** — cross-encoder quality reranking with graceful fallback if the model is unavailable
- **Multimodal search** — upload an image and search movie text in CLIP's shared embedding space
- **Grounded RAG** — question answering, summaries, citations, recommendations, comparisons, and fact extraction
- **Recursive RAG** — the assistant can issue a follow-up retrieval query when the first retrieval is insufficient
- **Query enhancement** — spell correction, rewriting, and query expansion through OpenRouter
- **LLM evaluation** — score retrieved documents from 0–3 for relevance
- **Secure API-key storage** — OpenRouter keys are stored via the operating-system credential vault using `keyring`
- **Local-first retrieval** — search, semantic embeddings, reranking, and image search do not require an LLM API key
- **Persistent caches** — semantic and CLIP corpus embeddings are cached between runs
- **Model management** — prepare models ahead of time, inspect cache usage, and clear generated caches from the UI
- **Custom movie datasets** — validated JSON datasets can be imported by the application backend
- **Windows packaging** — GitHub Actions builds both a portable ZIP and an Inno Setup installer

## Reliability improvements in v1.3

Earlier builds could be closed while a Transformer model was still loading in a background Qt worker. That could leave Hugging Face/Transformers threads running during Python interpreter shutdown and trigger errors such as `cannot schedule new futures after interpreter shutdown` and `Signal source has been deleted`.

v1.3 addresses this in several ways:

1. Qt signal emission is guarded during object teardown.
2. The application prevents normal window close while a background model/search task is active.
3. Cross-encoder reranking is optional and disabled by default.
4. Reranker initialization failure falls back to RRF results instead of failing the search.
5. Full-corpus sorting was replaced with partial top-k selection where practical.
6. Model/index preparation can be performed explicitly from Diagnostics.

## Search pipeline

```text
User query
   ├── BM25 keyword retrieval ──────┐
   └── semantic embedding retrieval ├──> Reciprocal Rank Fusion
                                    │           │
                                    └───────────┘
                                                │
                                      candidate documents
                                                │
                                  optional cross-encoder rerank
                                                │
                                           top context
                                                │
                                           RAG / LLM
```

For image search:

```text
Uploaded image -> CLIP image embedding
                         |
                         v
                shared vector space
                         ^
                         |
movie title + description -> cached CLIP text embeddings
```

## RAG modes

The RAG Assistant provides:

- **Question** — direct grounded answers
- **Summary** — multi-document synthesis
- **Citations** — answers with numbered source references
- **Recommendations** — ranked suggestions with supporting evidence
- **Compare** — comparison of the strongest retrieved candidates
- **Facts** — source-cited fact extraction
- **Recursive RAG** — up to three retrieval rounds when context is incomplete

It can also enhance the retrieval query with spell correction, rewriting, or expansion.

## Security

An OpenRouter API key is optional. Local retrieval features work without one.

When a key is saved in the UI, Webflyx uses the operating system credential vault through Python `keyring`. The application does not intentionally write saved credentials to source files, JSON config files, ordinary logs, or the Git repository.

Retrieved movie text is sent to OpenRouter only when an AI feature is explicitly invoked.

See [SECURITY.md](SECURITY.md) for more detail.

## Run from source

Requirements:

- Python 3.11–3.13
- [`uv`](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/Alex-Unnippillil/webflyx-rag-studio.git
cd webflyx-rag-studio
uv sync --all-groups
uv run python desktop_app.py
```

On first use, Webflyx may download the movie corpus and one or more ML models. These files are cached under the current user's application-cache directory.

## Development

```bash
uv run ruff check .
uv run pytest -q
```

## Windows releases

Tags matching `v*` trigger the Windows release workflow. The workflow produces:

- `Webflyx-RAG-Studio-Setup.exe`
- `Webflyx-RAG-Studio-portable-windows-x64.zip`

A local Windows build can also be started from PowerShell:

```powershell
.\scripts\build_windows.ps1
```

## Dataset format

The default movie corpus uses this shape:

```json
{
  "movies": [
    {
      "id": 1,
      "title": "Example Movie",
      "description": "Movie description"
    }
  ]
}
```

Imported datasets are validated before replacing the local corpus and are capped at 75 MB.

## License

MIT
