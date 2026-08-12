# Engramme: A Technical Review — 핵심 발췌

> 출처: [분석 문서](../../report/[paper]_Engramme_A_Technical_Review_2026_Samsung_NEXT.md) / 원본: `Engramme_A_Technical_Review_2026_Samsung_NEXT.pdf` (Samsung NEXT, 2026-04-17)
>
> 본 발췌는 작성자 Vinod Joseph(Samsung NEXT)가 인용한 Engramme 기술 보고서(engramme.com/research) 및 선행 연구(Srinivasan 2023 Kreiman Lab)의 핵심 수식·메커니즘·수치를 발췌·정리.

---

## Document Identity & Lineage

- **Title**: Engramme: A Technical Review
- **Subtitle**: The Engramme architecture including Large Memory Models (LMM) and Dynamic Recurrent Attractor Network (DRAN), Memorome, Ambient Intelligence Module and a competitive benchmark against Mem0, Letta, Zep, and Supermemory
- **Author of this review**: Vinod Joseph — Samsung NEXT
- **Subject company**: Engramme (formerly Memorious Inc.)
- **Founders**: Gabriel Kreiman, PhD (CEO) · Spandan Madan, PhD (CTO)
- **Date**: April 17, 2026 (문서 식별자 `260421`)
- **Privileged and Confidential** due-diligence technical report
- **Scientific origin**: Kreiman Lab (Harvard Medical School / Boston Children's Hospital), 25+ paper body of work. Key cited papers: Shaham et al. 2022 (#122), Zheng et al. 2024 (#149), Talbot et al. 2025 (#158). Unpublished: Zhang et al. Nature Human Behavior (in press, Turing-style human/AI knowledge comparison); first-of-three personal-memory benchmark dataset "What do people need to remember?".

---

## §1 The Memory Gap — 5 gaps motivating a purpose-built architecture

1. **Scale** — human lifelong memory ~petabytes; orders beyond RAG indexes or LLM context windows. Self-attention O(N²) → 10¹²-byte scale infeasible → "radically new approach needed".
2. **Personalization** — every user's acquaintances/private knowledge/lived experience not capturable in pretraining.
3. **Proactivity** — human memory surfaces spontaneously; AI memory is pull-based (requires a query).
4. **Contextuality** — what comes to mind depends on activity/environment/internal state (Godden & Baddeley, 1975, cited).
5. **Associativity** — abstract associative, not low-level lexical/semantic similarity like dense/sparse retrievers.

> **"Dark matter of memory"** — situationally irretrievable memories a user would benefit from but would never think to query for. RAG / agentic search / LLM fine-tuning all require a prompt → by construction cannot surface dark matter.

---

## §2 Three Pillars of the Engramme Architecture

### 2.1 Ambient Intelligence Module (AIM)

- Recurrent residual **multimodal encoder** fusing user "ambience" into a single state vector **ξ ∈ ℝ<sup>D</sup>**.
- Ambience inventory (broader than typical multimodal models):
  - Company (whom the user is with)
  - Location (with user-specific semantics: this is home, this is work)
  - Time (date, season, anniversaries)
  - Conversational context (what the user is talking about right now)
  - Digital activity (apps open, messages, browsing, searches)
  - Other sensory input (ambient sound, music playing)
  - Recent ambience — recurrent residual model accumulates last hours/days so the immediate moment is couched in recent history
- Explicitly includes **subverbal signals** that gate biological recall but are almost never available to an LLM retrieval layer.
- ξ updated continuously → enables **promptless, searchless recall**: "the environment itself is the query".

### 2.2 Dynamic Recurrent Attractor Network (DRAN)

Inference & storage core of the LMM. Scaled extension of modern Hopfield networks (Krotov & Hopfield 2016; Ramsauer et al. 2021). Four distinctive design choices:

1. **Direct (Hebbian) weight initialization** — given ambience–memory pairs (ξᵢ, xᵢ), DRAN initializes weight matrices **K, V** directly with no gradient descent and no per-user fine-tuning. Key to scalable personalization.
2. **Exponential storage capacity** — interaction terms of the form **E = −exp(K<sup>T</sup>ξ)** retain exponentially many distinct memory patterns within a compact ~200 MB model.
3. **Multi-hop associative recall** — L stacked attractor blocks; each block's recalled memory x_{l,i} becomes the input for the next hop, mirroring chained human recall ("one memory leads to another").
4. **Kinematic gating** — attractor convergence kinematics act as a **memory trigger**; a block only emits output when ξ readily converges to a stable attractor (user-specific memory exists for current ambience). When convergence fails, recall terminates, **preventing confabulation**.

> DRAN pipeline per block: each block runs **n_{l,i} annealed recurrent steps** (input-dependent, not fixed) gated by a Memory Trigger. The Memory Decoder uses constrained decoding against the Memorome to produce the l-hop recalled memory **t_{rec,l}**; the Memory Encoder simultaneously emits **t_{enc,l}** for ingestion back into the memorome — the forward/backward symmetry from which the **Janus instance** takes its name.

### 2.3 Memorome Engine

Long-term structured repository: entities, time, place, headline, narrative, and other attributes extracted from the user's digital life. Three technically distinctive properties:

- **Distillation to ~10⁷× compression** — retains key features of an experience rather than raw sensory data; petabyte-scale multimodal life → ~200 MB personal model (on-device).
- **Constrained decoding grammar** (Willard & Louf, 2023) — Memorome compiles into a grammar restricting the Memory Decoder to valid memories from the user's actual record. Mechanism behind the **1.2% hallucination rate vs. 36–54% for frontier LLMs** on personal-knowledge tasks.
- **Episodic segmentation** — neural architectures inspired by hippocampal "boundary neurons" chunk continuous experience into discrete episodes populating the entity/time/place schema.

---

## §3 Scientific Foundation — Kreiman Lab Lineage & Srinivasan (2023) Precursor

Srinivasan (2023) ETH Zürich master thesis carried out in the Kreiman Lab — "Hebbian attractor to model working memory in complex human behavior". Direct precursor to DRAN. Hebbian attractor network modeling working memory during a naturalistic card-matching task; predicts both:

- **Behavioral signatures**: click counts, reaction times, memory decay
- **Neurophysiological signatures**: intracranial-field-potential correlates of novelty/familiarity and retrieval confidence (epilepsy-patient electrodes)

### Architecture of the thesis network

- Maintains a memory matrix **M_t** updated as:

> **M_t = λ M_{t-1} + η h_t h_t<sup>T</sup>**

  — a **Hebbian outer-product update on the network's own state h_t**, not the raw input x. LeakyReLU nonlinearity + activation normalization.

- 3×3 grid example (Fig. 3.2 of thesis): blue units encode positions x_p, orange units encode image labels x_l, connected by Hebbian weights M_t. Learning regime: a cat observed at position 5 creates strong label–position associations. Inference regime: model shown the label, network dynamics drive correct positional unit to highest activity.

- This is the **working-memory analog of DRAN's ambience→memory trigger → multi-hop recall pipeline**.

### Two claims validated by the thesis

(i) Hebbian initialization is **sufficient for associative recall without gradient fine-tuning** — validated experimentally against human behavior.
(ii) Attractor convergence kinematics carry **interpretable cognitive signals** — same network matches intracranial novelty/familiarity and retrieval-confidence signals.

DRAN's "memory trigger event" gating is a direct descendant. Engramme lifts this to scaled, multi-hop, multimodal setting via the K, V formulation and E = −exp(K<sup>T</sup>ξ).

---

## §4 Engramme's Own Benchmarks (Sec. 7 of the technical report)

Comparison against frontier LLMs (Gemini 2.5 Pro, Claude Opus 4.5, ChatGPT 5.2) with their native memory connectors (Claude/ChatGPT "Company Knowledge", Gemini Google Workspace access) on surfacing user-specific memories for real text ambiences (web browsing, email reading, document writing). LLMs prompted to "retrieve the most relevant pieces of information"; Engramme **received no prompt**. Users compared top memories in **blinded pairwise form**.

| Metric | Engramme | Gemini 2.5 Pro | Claude Opus 4.5 | ChatGPT 5.2 |
|---|---|---|---|---|
| End-to-end recall latency (median) | **1.7 s** | 13.2 s | 34.1 s | 49.4 s |
| Speedup factor | — | 8× | 20× | 29× |
| Blinded user preference win rate | — | 91% | 80% | 87% |
| Hallucination rate | **1.2%** | 36–54% (frontier LLM cohort) | | |

> Latency advantage attributed to: tested LLMs combine agentic search (tens of seconds) + RAG (seconds) per query — unsuited to proactive recall where ambience evolves in under a minute. Engramme's memorome is pre-built + streaming ambience, so **no query-time indexing or tool-use**.

> Error bars on preference: 95% CI of the mean.

---

## §5 Competitive Benchmarks — Mem0, Letta, Zep, Supermemory

Standard benchmarks: **LoCoMo** (Snap Research, ACL 2024; 10 long multi-session conversations, ~1,540 questions) and **LongMemEval** (500 questions across 6 categories including temporal reasoning and knowledge updates).

### §5.1 Mem0 / Mem0g — vector-first selective memory

- arXiv:2504.19413, accepted ECAI 2025. Memory-centric: extract salient facts per turn, then **ADD / UPDATE / DELETE / NOOP tool-call reconciliation** against a vector store.
- Mem0g: adds directed labeled graph (entities = nodes, relationships = edges) for relational reasoning.
- **LoCoMo** (LLM-as-judge, GPT-4o-mini): Mem0 66.9%, Mem0g 68.4%, full-context 72.9%.
- **p95 latency** (LoCoMo): Mem0 1.44 s, Mem0g 2.59 s vs 17.12 s full-context. ~1,800 tokens/conversation vs ~26,000 → 91% p95 latency reduction, >90% token savings.
- Newer (April 2026) Mem0 algorithm: **85.0% LoCoMo** at <7,000 tokens/retrieval.

### §5.2 Letta (MemGPT) — hierarchical agent runtime

- Production evolution of MemGPT (Packer et al. 2023, arXiv:2310.08560). Treats LLM context window like virtual memory: core memory blocks in-context, archival memory external, moved via function calls decided by the agent.
- Filesystem-backed agent: **74.0% LoCoMo** with GPT-4o-mini (above Mem0g's 68.5% in their comparison).
- **DMR benchmark** (MemGPT proposed): MemGPT reaches **93.4%**, against which Zep is measured.
- Leaderboard framework measures read/write/update separately against core vs archival.

### §5.3 Zep (Graphiti) — temporal knowledge graph memory

- arXiv:2501.13956 (2025). Built on **Graphiti**: temporally-aware KG engine with episodic/semantic/community subgraphs; **bitemporal model** — every fact carries a validity window (when true) + ingestion timestamp (when recorded).
- **DMR**: 94.8% (GPT-4 Turbo) / 98.2% (GPT-4o-mini) vs MemGPT 93.4%.
- **LongMemEval**: up to +18.5% accuracy, 90% latency reduction vs baseline (cross-session synthesis + temporal tasks).
- **LoCoMo disputed**: Zep 84% (original) → Mem0 reproduction 58.44% → Zep counter 75.14% (±0.17 over 10 runs).
- Independent: Zep 63.8% LongMemEval (GPT-4o) vs Mem0 49.0%.

### §5.4 Supermemory — vector-graph hybrid with dual timestamps

- Infinite-context memory API; vector-graph engine; 5-layer stack: **connectors → extractors → Super-RAG → memory graph → user profiles**.
- **Dual timestamp**: documentDate (when conversation happened) + eventDate (when described event occurred) → temporal reasoning + knowledge-update handling.
- **LongMemEval-S** (LLM-as-judge): 81.6% (GPT-4o) / 84.6% (GPT-5) / 85.2% (Gemini 3 Pro) vs full-context GPT-4o 60.2% and Zep GPT-4o 71.2%.
- Operational: sub-300 ms retrieval at 100 B+ tokens/month.

---

## §6 Cross-System Comparison Matrix (consolidated, directional)

| Dimension | Engramme (LMM) | Mem0 / Mem0g | Letta (MemGPT) | Zep (Graphiti) | Supermemory |
|---|---|---|---|---|---|
| Core paradigm | Large Memory Model — associative network replaces retrieval substrate | Memory layer above an LLM — extract + reconcile facts into vector (+ graph) | Agent runtime with hierarchical context (core in-context + archival external), LLM-managed | Temporal knowledge graph with episodic/semantic/community subgraphs | Vector-graph hybrid engine with dual-timestamp temporal reasoning |
| Data structure | DRAN associative weight matrices (K, V) + Memorome (entity/time/place/headline/narrative) | Vector DB of extracted facts; Mem0g adds directed labeled graph | In-context memory blocks + external archival store (vector-based) | Bitemporal KG: facts carry validity + ingestion timestamps; multi-level subgraphs | Facts-on-facts graph (not entity-relation-entity triples); automatic relationship tracking |
| Retrieval trigger | Promptless/searchless — ambience vector ξ triggers attractor convergence; kinematic gating | Query-driven semantic search; selective retrieval k≈1–5 | Agent tool-call (LLM decides when to search archival memory) | Query-driven multi-hop graph traversal with temporal filters | Query-driven hybrid vector + keyword + graph retrieval |
| Personalization | Hebbian weight init from (ξᵢ, xᵢ) pairs — no fine-tuning; RLHF on engagement signals | None at model level; per-user memory profile built from conversation history | User-specific memory blocks + persona; optional self-editing memory | Per-user graph with entity/community summaries | Per-user profile layer + user-owned memory hub (MCP) |
| Hallucination control | Constrained decoding grammar over memorome → 1.2% reported | ADD-only single-pass extraction (new algorithm) preserves provenance | LLM-policed; inherits base-model hallucination | Edge invalidation + provenance preserved in graph | Contradiction resolution via Update relationships; noise filtering |
| Latency (p50/p95) | 1.7 s median end-to-end recall (vs 13.2/34.1/49.4 s Gemini/Claude/ChatGPT w/ connectors) | 0.71 s / 1.44 s (Mem0); 1.18 s / 2.59 s (Mem0g) on LoCoMo | Not separately reported; dominated by LLM tool-call roundtrips | ~90% latency reduction on LongMemEval vs baseline; 3 s (GPT-4 Turbo) vs 30 s baseline | Sub-300 ms recall at 100 B+ tokens/month (vendor-reported) |
| Token/context cost | ~200 MB personal model; no per-query token stuffing (recall is in-model) | ~1,800 tokens/conversation vs ~26,000 full-context — >90% reduction | Bounded to core-memory block size; archival retrieval adds variable cost | ~1,600 tokens/query on LongMemEval vs ~115,000 baseline | Under 7,000 tokens/retrieval (latest generation) |
| LoCoMo (LLM-as-judge) | Not published | 66.9% (Mem0) / 68.4% (Mem0g); 85.0% new algorithm | 74.0% filesystem-backed GPT-4o-mini | Disputed: 84% → 58.44% (repro) → 75.14% (corrected) | #1 per vendor; exact score not published (April 2026) |
| LongMemEval | Not published | 49.0% (GPT-4o, independent) | Not published | 63.8% (GPT-4o, independent) / up to ~71% in Supermemory's eval | 81.6% (GPT-4o) / 84.6% (GPT-5) / 85.2% (Gemini 3 Pro) |
| DMR (MemGPT benchmark) | Not published | Not published | 93.4% (MemGPT original) | 94.8% (GPT-4 Turbo) / 98.2% (GPT-4o-mini) | Not published |
| Deployment model | Compact on-device personal model; ingests from email, calendar, messaging, location, photos/videos, docs, browsing, app use | Open-source + managed cloud; drop-in memory layer | Open-source (Apache) agent runtime | Managed cloud + open-source Graphiti engine (24K+ GitHub stars) | Managed cloud with MCP universal memory server |
| Core differentiator | Proactive / promptless recall driven by ambient context; an actual model, not a retrieval pipeline | Lowest-latency vector memory layer; mature production SDKs | True agent runtime — memory managed as OS primitives | Native temporal reasoning over validity-windowed facts | Infinite-context drop-in proxy with enterprise connectors |

> **Reading note** — LoCoMo and LongMemEval scores sensitive to underlying LLM and evaluation setup. Zep/Mem0 LoCoMo dispute is public (GitHub issue #5, getzep/zep-papers, May 7 2025). Independent reproductions remain rare.

---

## §7 Strategic Positioning — Traditional IR vs Engramme step-by-step

| Step | Engramme Architecture | Traditional IR (RAG/Agentic) | Performance & Novelty |
|---|---|---|---|
| 1. Data Ingestion | Live Source Connection — continuous sync of communications, location, digital activity | Static Indexing — periodic batch of text documents into vector DB | Real-time: live updates ensure memory never stale |
| 2. Compression | Episodic Distillation — "boundary neurons" chunk data into schema records (Entities, Time, Place) | Semantic Chunking — fixed-size or recursive character splitting for embedding | **10⁷ reduction**: petabytes → ~200 MB without losing episodic context |
| 3. Memory Storage | Dual-Write (Hebbian) — structured data in Memorome + sub-symbolic associations in DRAN weights | Database Entry — vectors in flat/graph index (Pinecone, Milvus) | One-Shot: bypasses expensive gradient descent for near-instant personalization |
| 4. Query Input | State Vector (ξ) — AIM fuses multimodal ambience into a state vector | User Prompt — explicit manually formulated queries / search anchors | Searchless: proactive recall removes cognitive load of formulating prompts |
| 5. Retrieval Logic | Kinematic Gating — DRAN monitors attractor convergence to identify "memory trigger events" | Similarity Search — top-k nearest neighbor (cosine/dot product) | Zero Hallucination at Dynamics: prevents output if no stable attractor found |
| 6. Context Depth | Multi-hop Chaining — stacked layers L represent associative steps where one memory seeds the next | Single Retrieval Pass — flat set of results based on a single query vector | Associative: emulates human "train of thought" recall |
| 7. Final Output | Constrained Decoding — vectors → tokens via grammar compiled from the Memorome | Unconstrained Generation — LLM generates text from retrieved context, prone to mixing sources | Grounded: 1.2% hallucination vs 36–54% LLMs |
| 8. Latency | Sub-1.7 s median — optimized for real-time on-device associative lookup | 10–50 s — agentic search often takes tens of seconds | 8–29× faster: recall in 1.6–1.7 s vs 13–49 s for frontier LLMs |

---

## §7.1 Differentiators architecturally unique to Engramme

1. **Ambience-triggered recall** — none of Mem0/Letta/Zep/Supermemory has an AIM analog; recall is always query-gated in those systems. Only mechanism capable of addressing the "dark matter of memory" gap.
2. **Model-level personalization without fine-tuning** — DRAN's Hebbian initialization is not a retrieval trick on top of a base model; memories are encoded into the weight matrices themselves. Srinivasan thesis empirically grounds sufficiency for behavior-matching recall.
3. **Kinematic gating as a hallucination controller** — attractor convergence doubles as a confidence signal (no convergence → no memory emitted). More principled, mechanism-level answer to hallucination than grammar/graph-based post-filters in Mem0/Zep/Supermemory.
4. **On-device footprint** — ~200 MB compressed personal memory; privacy- and latency-relevant deployment property cloud-first memory APIs do not compete on.

---

## §7.2 Open questions (due-diligence perspective)

- **Standardized benchmarks** — Engramme has not published LoCoMo, LongMemEval, or DMR. Own comparison vs LLMs-with-connectors rather than purpose-built memory systems. Head-to-head vs Mem0g/Zep/Supermemory on LongMemEval would close the most obvious credibility gap.
- **Structured world knowledge coverage** — architecture is purpose-built for personal memory. Queries blending personal + world knowledge (e.g. "what did my doctor say about this drug?") not described.
- **Scaling behavior of attractor convergence** — paper asserts exponential storage capacity, but at 10¹²-byte memorome scale the empirical ratio of true triggers to false/missed triggers will determine UX. Kreiman lab's upcoming benchmark series is the right vehicle.
- **Constrained decoding breadth** — 1.2% hallucination vs frontier LLMs at 36–54%; evaluation must cover same task distributions as LoCoMo/LongMemEval for comparability.

---

## §8 Data Flow — Storage (Ingestion) Pipeline

### Stage 1 — Source connection & retrospective ingestion

Live connections to user-authorized sources; pulls historical data back as far as user permits. Full source catalog:

- **Communications**: meetings, conversations, phone, email, messaging, videoconferencing
- **Location & maps**: home/work, map pins, parking, travel, locations of interest
- **Photos & media**: videos, music, books, movies
- **Digital activities**: browsing, computer use, app use, fitness tracking
- **Documents**: purchases, financial statements
- **Work & productivity**: calendar, PDFs, notes
- **Other digital files**: local files, Google Drive, OneDrive, iCloud, Dropbox

Integrations synchronize **continuously** with the Memorome as a live repository, not one-time import.

### Stage 2 — Episodic segmentation & attribute extraction (the ~10⁷× compression step)

Raw records chunked into discrete episodes by a segmentation network inspired by hippocampal "boundary neurons". Each episode distilled into a fixed-schema record:

> **Entities · Time · Place · Headline · Narrative · Additional modality-specific attributes**

Keeps features of the experience, not raw sensory data — same philosophy as human episodic memory. This is where petabyte → ~200 MB happens.

### Stage 3 — Dual write: Memorome + DRAN weights

Episode becomes a training pair (ξᵢ, xᵢ):

- **ξᵢ** = ambience vector at the time of the event (AIM's encoding of who/where/when/what-was-open)
- **xᵢ** = the encoded memory vector

DRAN uses Hebbian-like direct weight initialization with **no gradient descent, no per-user fine-tuning**. Outer-product construction:

> **W = Σ xᵢ ⊗ ξᵢ**

writes each pair directly into the K and V weight matrices. The paper is explicit: "the associative weights form a representation of the user's memories." Data exists in two places simultaneously — structured (Memorome) and sub-symbolic (DRAN weights).

### Stage 4 — Exponential storage via modern Hopfield dynamics

DRAN = modern Hopfield network (Krotov & Hopfield 2016; Ramsauer et al. 2021) using:

> **E = −exp(K<sup>T</sup>ξ)**

Storage scales **exponentially in memory pattern count** for a given weight dimension → ~200 MB of weights holds a lifetime of distinct patterns.

### Stage 5 — Continuous write-back & adaptation

Two additional write paths keep the system live:

- **Streaming ambience → Memorome**: AIM's real-time ambience stream is itself ingested as a memory source → "what did I do this morning?" answerable without query-time search.
- **Memory Encoder → Memorome**: on every recall pass, DRAN's Memory Encoder emits t_{enc,l} of the current ambience at each hop and writes it back. "the DRAN simultaneously looks forward and backward in time."
- **RLHF tuning**: engagement signals (reads, scrolls, explicit feedback) tune DRAN end-to-end. Initial writes gradient-free (Hebbian), weights refinable later.

---

## §8 Data Flow — Retrieval Pipeline

### Stage 1 — Ambience encoding in the AIM

AIM continuously fuses a multimodal token stream t = (t₁, t₂, …, t_T). Outputs **ξ ∈ ℝ<sup>D</sup>**, a unified multimodal state vector that **replaces a search query** — "the environment is the query".

### Stage 2 — DRAN associative dynamics, layer by layer

ξ enters DRAN's stack of L attractor blocks. Each block l:

1. Applies associative attractor dynamics parameterized by (K, V) from the Hebbian initialization.
2. Anneals recurrently for a **dynamic, input-dependent number of steps n_{l,i}** — not fixed; depends on how fast the dynamics settle for this particular ambience.
3. Produces a candidate memory vector x_{l,i}.

### Stage 3 — Kinematic gating (the "memory trigger event")

Before emitting anything, the block inspects the kinematics of convergence — how readily the dynamics settle into a stable attractor:

- Convergence fast and clean → user-specific memory exists → block emits x_{l,i}. "memory trigger event."
- Convergence sluggish or fails → no memory emitted. Multi-hop recall terminates after l ≥ 0 hops when dynamics no longer converge readily.

> **Mechanistically important**: hallucination isn't filtered downstream, it's prevented at the dynamics level. **No convergence, no output.**

### Stage 4 — Multi-hop associative chaining

Emitted output becomes input seed for block l+1. Each hop = one associative step — "this memory leads to that memory." Chain continues until depth budget L reached or kinematic gating terminates.

### Stage 5 — Constrained decoding to tokens

Each emitted memory vector x_{l,i} passes to the Memory Decoder → tokens t_{rec,l}. Decoder operates under a **constrained decoding grammar** (Willard & Louf, 2023) compiled from the user's actual Memorome. Final hallucination guardrail: the decoder cannot emit anything that isn't a valid memory already in the Memorome. Mechanism behind 1.2% hallucination vs 36–54% for frontier LLMs.

### Stage 6 — Spontaneous Recall pipeline surface

Gated, decoded memories flow through the Spontaneous Recall pipeline:

> **Memory Trigger → Candidate Generation → Personal Utility Module → Proactive Memories**

Personal Utility Module ranks and filters candidates before surfacing. Engagement signals close the loop via RLHF.

---

## §9 Conclusion (excerpt)

> "Engramme's Large Memory Model is the first production-oriented system we are aware of that re-architects memory at the model layer rather than as a retrieval adapter. Its scientific lineage — modern Hopfield networks, the Hebbian-attractor working-memory thesis from the Kreiman Lab, and the lab's broader AI+memory program — is unusually well grounded for the category."

> "Against the four most-cited memory stacks — Mem0, Letta, Zep, Supermemory — Engramme's differentiators (ambience-triggered recall, Hebbian personalization, kinematic gating, on-device footprint) are architecturally unique and address gaps the existing stacks do not close. The outstanding work is evaluation standardization: publishing DMR, LoCoMo, and LongMemEval numbers, ideally on the same LLMs used in vendor reports, would let the LMM compete on the benchmark battlefield where this category is currently being decided."

---

## References (as cited in the review)

1. Engramme. *Looking Forward and Backward in Time with Dynamic Recurrent Attractor Networks*. Engramme technical report, 2025. engramme.com/research.
2. R. F. Srinivasan. *Hebbian attractor to model working memory in complex human behavior*. Master Thesis, ETH Zürich (carried out in Kreiman Lab, Harvard Medical School). Sept 27, 2023.
3. P. Chhikara et al. *Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory*. arXiv:2504.19413, 2025 (ECAI 2025).
4. C. Packer et al. *MemGPT: Towards LLMs as Operating Systems*. arXiv:2310.08560, 2023.
5. Letta. *Benchmarking AI Agent Memory: Is a Filesystem All You Need?* Aug 12, 2025. letta.com/blog/benchmarking-ai-agent-memory.
6. P. Rasmussen et al. *Zep: A Temporal Knowledge Graph Architecture for Agent Memory*. arXiv:2501.13956, 2025.
7. Supermemory. *Research — State-of-the-Art Agent Memory*. supermemory.ai/research.
8. Engramme research brief — Kreiman Lab publication list. klab.tch.harvard.edu/publications.
9. Mem0. *Introducing The Token-Efficient Memory Algorithm*. April 13, 2026. mem0.ai/blog/mem0-the-token-efficient-memory-algorithm.
10. Vectorize. *Mem0 vs Letta (MemGPT): AI Agent Memory Compared (2026)*. March 15, 2026.
11. GetZep. *Revisiting Zep's 84% LoCoMo Claim: Corrected Evaluation & 58.44% Result*. Issue #5, getzep/zep-papers, May 7, 2025.
