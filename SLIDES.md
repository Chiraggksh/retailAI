# Northstar Retail AI — Presentation Slides

---

## Slide 1 — Title & Business Problem

### Northstar Retail Customer Support
### Agentic AI Resolution System

---

**Business Problem**

Northstar's customer support team handles hundreds of daily queries about:
- Order status & tracking
- Return eligibility (30-day window, condition checks)
- Refund status & timelines
- Policy questions from a 50-page PDF manual

**Pain Points (Before)**
| Problem | Impact |
|---|---|
| Agents must look up orders manually | 3–5 min avg handle time |
| Policy answers inconsistent across agents | Customer dissatisfaction |
| No 24/7 coverage | Missed queries overnight |
| No audit trail for decisions | Compliance risk |

**Objective**: Build an end-to-end agentic AI system that resolves customer support queries autonomously — grounded in real-time operational data and official policy documents — with full observability and security guardrails.

---

## Slide 2 — Ingestion Pipeline & Structure-Aware Chunking

### Phase 1: Data Ingestion

```
Policy PDF (50 pages)          Operational Database
        │                              │
   PyMuPDF                    JSON/CSV Loaders
   (font size + bold           (customers, orders,
    detection)                  returns, refunds)
        │                              │
   StructureAwareChunker         OperationalService
   (heading-aware splits,         (in-memory lookup,
    512 tokens, 50 overlap,        ownership-validated)
    SHA-256 chunk IDs)                 │
        │                             MCP Server
   EmbeddingManager                (stdio transport)
   (all-MiniLM-L6-v2,
    384-dim, local CPU)
        │
   PolicyVectorStore
   (ChromaDB, cosine similarity,
    idempotent upserts via SHA-256)
```

**Key Design Decisions**
- **Structure-Aware Chunking**: Detects headings by font size/bold → chunks don't split mid-policy-rule
- **SHA-256 Chunk IDs**: Re-running ingestion is safe — identical content produces identical IDs → no duplicate embeddings
- **Local Embeddings**: `all-MiniLM-L6-v2` runs entirely on CPU — zero API cost, offline-capable
- **ChromaDB persistence**: `./data/chroma_db/` survives restarts; no re-ingestion needed

---

## Slide 3 — Hybrid Retrieval & Benchmark Results

### Phase 2: Dense + BM25 + RRF Reranking

**Three Retrieval Methods Compared**

| Method | Mechanism | Strength |
|---|---|---|
| **Dense (MiniLM)** | Cosine similarity on 384-dim vectors | Semantic meaning, paraphrase matching |
| **BM25** | TF-IDF keyword scoring | Exact terms, product codes, SKU numbers |
| **RRF Hybrid** | Reciprocal Rank Fusion (k=60) | Best of both — combined ranked list |

**Reciprocal Rank Fusion Formula**

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}, \quad k = 60$$

**Benchmark Results** — 12 labeled policy queries

| Metric | Dense | BM25 | **RRF Hybrid** |
|---|---|---|---|
| MRR | 1.0 | 0.98 | **1.0** |
| NDCG@3 | 1.0 | 0.97 | **1.0** |
| Hit Rate@5 | 100% | 100% | **100%** |
| Avg Latency | ~18ms | ~2ms | ~20ms |

> **Winner**: RRF Hybrid. MRR=1.0 and NDCG@3=1.0 — every relevant chunk ranked first across all 12 queries.

---

## Slide 4 — Stateful Agent Architecture

### Phase 3: RetailSupportAgent

```
User Query
    │
    ▼
[AgentState]  ←─────────────────────────────────────────┐
  conversation_history                                   │
  pending_tool_calls                                     │
  completed_steps                                        │
    │                                                    │
    ▼                                                    │
[GeminiClient]  ──→  Gemini API (gemini-3.5-flash-lite)  │
  function_declarations                                  │
  system_instruction                                     │
    │                                                    │
    ▼                                                    │
 Tool Call Requested?                                    │
    │                                                    │
   YES ──→ [MCPClient] ──→ [MCP Server subprocess] ──→ result
    │             │               (stdio pipe)           │
    │       ownership check                              │
    │       (C-001 owns NS-1001?)                        │
    │                                                    │
    └────────── Function Response ───────────────────────┘
    │
   NO / Max steps reached
    │
    ▼
[Final Answer]  ←── Policy citations from ChromaDB
```

**Agent Guardrails**
- **Max 6 steps** per conversation turn (prevents infinite loops)
- **Ownership validation** — server rejects cross-customer lookups at MCP layer
- **Offline fallback** — if Gemini API unavailable, deterministic rule-based answer returned
- **503 auto-fallback** — tries `gemini-3.6-flash` → `gemini-3.1-flash-lite` on overload

**What Makes it "Agentic"**
- **Autonomous tool selection**: Model decides which MCP tools to call, in what order
- **Multi-step reasoning**: Can call 3+ tools before composing final answer
- **Stateful memory**: Full conversation history passed on every turn
- **Dynamic function calling**: Tool schemas discovered at runtime from MCP server

---

## Slide 5 — MCP Server & Security

### Model Context Protocol (MCP) — Official Python SDK

**Why MCP?**
- Industry-standard protocol (Anthropic / open standard) for AI ↔ tool communication
- Separates tool _definition_ (server) from tool _execution_ (client) cleanly
- Enables future tool reuse across different AI models

**Architecture**

```
app.py / cli.py
      │
 MCPClient (mcp_client.py)
      │  ← async context manager
      │  ← schema conversion: MCP → Gemini FunctionDeclaration
      │
  stdio pipe
      │
 MCPServer (mcp_server.py)
  ├── get_order_details(order_id, customer_id)
  ├── check_return_eligibility(order_id, customer_id)
  └── get_refund_status(order_id, customer_id)
```

**Security: Server-Side Caller Authorization**

```python
# mcp_server.py — ownership check on EVERY tool call
trusted_caller = os.environ.get("DEMO_CUSTOMER_ID")
if order.customer_id != trusted_caller:
    return {"error": "AUTHORIZATION_FAILED", 
            "message": "You are not authorized..."}
```

- Customer identity injected via **subprocess environment variable** (not prompt text)
- Prompt injection cannot spoof identity (attacker can't change env vars via chat)
- Each MCP tool independently validates ownership before returning any data

---

## Slide 6 — Testing, Observability & UI

### Phase 3–4: Quality & Developer Experience

**Test Suite — 40/40 Passing**

| Test File | Coverage |
|---|---|
| `test_ingestion.py` | PDF chunking, metadata extraction, SHA-256 IDs |
| `test_retrieval.py` | Dense, BM25, RRF retrieval correctness |
| `test_mcp_server.py` | All 3 MCP tools, auth failure, edge cases |
| `test_mcp_client.py` | Schema conversion, async context manager |
| `test_agent.py` | Full agent loop, offline fallback, guardrails |

**Hermetic Tests** — `conftest.py` autouse fixture clears `GEMINI_API_KEY` → tests run fully offline, no API calls, no flakiness.

**Streamlit Web UI — 4 Tabs**

| Tab | Contents |
|---|---|
| 💬 Chat + Observability | Live chat, MCP trace inspector (tool calls + responses), agent step timeline |
| 📋 Operational DB | Browse all customers, orders, returns, refunds in real-time |
| 🔍 Policy Vector Search | Query ChromaDB directly, see similarity scores, raw chunks |
| 📊 Metrics | Retrieval benchmark results, MRR/NDCG scores, latency chart |

**Run command**: `.venv/bin/streamlit run app.py` → http://localhost:8501

---

## Slide 7 — Engineering Tradeoffs & Production Roadmap

### Key Engineering Decisions

| Decision | Chosen Approach | Tradeoff |
|---|---|---|
| **Embeddings** | Local CPU (MiniLM) | Zero cost, offline ✅ vs. slower than GPU |
| **Vector DB** | ChromaDB (local file) | Easy setup ✅ vs. not horizontally scalable |
| **LLM** | Gemini 3.5 Flash Lite | Fast + cheap ✅ vs. smaller context window |
| **Retrieval** | RRF Hybrid | Best accuracy ✅ vs. +20ms latency over BM25-only |
| **Tool Protocol** | Official MCP SDK | Standardized ✅ vs. heavier than raw HTTP |
| **Auth** | Env-var identity | Prompt-injection-safe ✅ vs. not production-grade JWT |

### Latency & Token Cost Profile (per query)

| Component | Latency | Notes |
|---|---|---|
| Embedding (local) | ~5ms | CPU inference |
| ChromaDB retrieval | ~15ms | In-process, file-backed |
| MCP tool call (×1–3) | ~50ms | stdio subprocess roundtrip |
| Gemini LLM call | ~500–2000ms | Network-bound |
| **Total (typical)** | **~800ms** | End-to-end wall time |

Tokens per query: ~400 prompt + ~150 completion = ~\$0.00005 at Flash Lite pricing

### Production Roadmap

```
Now (MVP)                Near-term              Production
─────────                ─────────              ──────────
Local ChromaDB     →     Vertex AI Vector  →    Multi-tenant
                         Search                 Vector DB

Env-var auth       →     JWT + OAuth 2.0   →    RBAC + audit
                                                logging

Single process     →     Async FastAPI      →    Kubernetes
                         backend                microservices

CLI + Streamlit    →     React frontend     →    Mobile app
```

---

*Northstar Retail AI — Built with Python 3.12, ChromaDB, MiniLM, Gemini, MCP, Streamlit*  
*All 4 phases complete · 40/40 tests passing · MRR=1.0 · NDCG@3=1.0*
