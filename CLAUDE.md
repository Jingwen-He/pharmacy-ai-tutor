# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pharmacy AI Tutor — a Python/Streamlit educational platform using multi-agent AI (Claude API + LangChain/LangGraph) to help pharmacy students learn from uploaded PDF course materials. Provides Q&A with citations and interactive quiz generation/evaluation.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

No test suite or linter is currently configured.

## Architecture

**Multi-agent pipeline** with keyword-based intent routing and LLM fallback:

```
User Input → OrchestratorAgent.route() → intent classification
  ├── greeting → static response
  ├── question → RetrievalAgent.search() → TutorAgent.answer()
  └── quiz/answer → RetrievalAgent.search() → QuizAgent.generate/evaluate()
```

**Three layers:**

- **`app.py`** — Streamlit single-page frontend. Uses `@st.cache_resource` for agent singletons and `st.session_state` for chat history, PDF status, active quiz, and mode (Q&A vs Quiz).
- **`agents/`** — Four agents: `orchestrator.py` (intent routing), `retrieval.py` (PDF ingestion + ChromaDB semantic search), `tutor.py` (cited Q&A), `quiz.py` (question generation + answer evaluation). All LLM agents use `claude-sonnet-4-20250514`.
- **`core/`** — `config.py` (Settings class with env vars and constants), `pdf_processor.py` (PyMuPDF extraction → RecursiveCharacterTextSplitter chunking at 800/200), `vector_store.py` (ChromaDB with `all-MiniLM-L6-v2` embeddings).

**`prompts/`** contains system prompt templates for tutor and quiz agents. Tutor requires citations in format `📖 (Page X, Section: "...")`. Quiz agent returns structured JSON with question, options, correct_answer, explanation, source, and difficulty.

## Key Design Decisions

- Intent classification uses keyword matching first, falls back to LLM (temperature=0) for ambiguous input
- Tutor agent is constrained to answer ONLY from provided teaching material with mandatory source citations
- Quiz agent outputs structured JSON; `_fallback_parse()` handles non-standard LLM JSON responses
- Vector store persists to `data/processed/` (gitignored)
- API key loaded from `.env` via python-dotenv

## Environment

Requires `ANTHROPIC_API_KEY` in `.env` file at project root.
