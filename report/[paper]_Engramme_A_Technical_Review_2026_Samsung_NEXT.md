> [paper] engramme.com/research · 원본: [source/paper/Engramme_A_Technical_Review_2026_Samsung_NEXT.pdf](../source/paper/Engramme_A_Technical_Review_2026_Samsung_NEXT.pdf)

# Engramme: A Technical Review

> 작성자: Vinod Joseph (Samsung NEXT) · 대상 회사: Engramme (formerly Memorious Inc.) · Founders: Gabriel Kreiman (CEO) · Spandan Madan (CTO) · 2026-04-17 · Privileged & Confidential due-diligence 기술 리뷰.
>
> 본 문서는 학술 논문이 아니라 Samsung NEXT가 투자·기술 평가 목적으로 작성한 **기술 실사(due-diligence) 보고서**다. Engramme의 원본 기술 보고서(engramme.com/research)와 Kreiman Lab 선행 연구(Srinivasan 2023)를 cross-reference하며, 동시에 동 카테고리의 4개 경쟁(memory stack: Mem0, Letta, Zep, Supermemory)을 정량 벤치마크 기준으로 비교한다.

## Summary & Outline

**한 줄 요약**: Engramme은 표준 LLM 위에 memory layer를 얹는 방식(Mem0/Letta/Zep/Supermemory가 모두 이 사분면에 속함)이 아니라, **연상 회상(associative recall)을 1st primitive로 삼는 새 모델 클래스 Large Memory Model(LMM)**을 제안한다 — 핵심은 modern Hopfield 기반 DRAN + ambient context 기반 promptless recall + Memorome constrained-decoding 으로 hallucination 1.2%·latency 1.7s·80-91% blinded preference win을 보고하나, **표준 벤치마크(LoCoMo/LongMemEval/DMR) 미발표**가 가장 큰 credibility gap.

**Outline**:
1. Memory Gap — 5가지 갭(scale, personalization, proactivity, contextuality, associativity) + "dark matter of memory"
2. Three Pillars — AIM / DRAN / Memorome Engine
3. Scientific Foundation — Kreiman Lab lineage, Srinivasan 2023 ETH/Kreiman Lab thesis가 DRAN의 직접 precursor
4. Engramme 자체 벤치마크 — vs Gemini 2.5 Pro / Claude Opus 4.5 / ChatGPT 5.2 (with native memory connectors)
5. 경쟁 분석 — Mem0 / Letta / Zep / Supermemory on LoCoMo·LongMemEval·DMR
6. Cross-System Comparison Matrix (step-by-step + 다차원 매트릭스)
7. Strategic Positioning + 4가지 architecturally-unique differentiator
8. Data Flow — Storage/Retrieval pipeline (stage-by-stage)
9. Conclusion + open questions

상세 발췌 → [excerpt](../source/paper/Engramme_A_Technical_Review_2026_Samsung_NEXT.md)

## Problem & Motivation

- **연구 배경**: 인간 평생 기억은 ~petabyte 규모이고 O(N²) self-attention 으로는 10¹²-byte scale 컨텍스트 확장이 computationally infeasible → "radically new approach needed". 또한 생체 기억은 자발적으로 surface 되지만, 현재 AI memory는 pull-based(query 필요)로 근본적 mismatch.
- **풀고자 하는 문제**: lifelong personal memory의 **proactive, promptless, context-triggered recall** — 사용자가 query하지 않아도 현재 ambience에 의해 자발적으로 surface 되는 "Spontaneous Recall". task 명칭은 personal lifelong memory recall (the report frames it as "promptless, searchless recall").
- **기존 접근의 한계**:
  - RAG / agentic search / LLM fine-tuning 은 모두 **prompt가 전제** → "dark matter of memory"(사용자가 query할 생각조차 안 하는 상황적 기억)를 구조적으로 surface 불가
  - Mem0/Letta/Zep/Supermemory는 "LLM 위의 memory layer"로 retrieval substrate 경쟁 — recall이 항상 query-gated
  - LLM의 native memory connector(Gemini Workspace, Claude/ChatGPT Company Knowledge)는 agentic search(수십 초) + RAG(초)를 매 쿼리마다 결합 → ambience가 1분 미만으로 evolve 하는 proactive recall에 부적합

## Contributions

- **방법론**: associative recall을 1st primitive로 하는 새 모델 클래스 **Large Memory Model (LMM)** 제안. retrieval adapter가 아니라 model layer 자체에서 memory 재설계.
  - DRAN(Dynamic Recurrent Attractor Network): modern Hopfield 기반, Hebbian weight init(no fine-tuning) + exponential interaction term `E = −exp(K^T ξ)` + multi-hop + **kinematic gating** 으로 hallucination을 dynamics level에서 prevent
  - AIM(Ambient Intelligence Module): multimodal ambience → state vector ξ 로 "환경 자체가 query"
  - Memorome Engine: constrained decoding grammar 로 decoding 단에서 hallucination 1.2% 보장
- **실증**: 1.7s median recall latency (8–29× faster), 80–91% blinded preference win rate, hallucination 1.2% vs 36–54%
- **비교 분석**: 동 카테고리 4개 stack(Mem0, Letta, Zep, Supermemory)을 LoCoMo/LongMemEval/DMR 기준 정리 → 모두 "LLM 위의 memory layer" 사분면, Engramme만 별개 lane.

## Method

### 전체 아키텍처 (ASCII)

```
                       ┌──────────────── Engramme LMM ────────────────┐
                       │                                               │
  USER DIGITAL LIFE ──▶│  (1) Memorome Engine                          │
  (email/cal/msg/      │      • live source connectors                 │  ──▶ Structured records
   loc/photo/docs/    │      • episodic segmentation ("boundary        │       (entity/time/place/
   browse/app/drive)   │        neurons" → schema records)             │        headline/narrative)
                       │      • ~10⁷× compression → ~200 MB            │
                       │                                               │              │
                       │              dual-write (W = Σ xᵢ ⊗ ξᵢ)       │              │
                       │              (Hebbian, no GD)                  │              │
                       │                    │                           │              │
                       │                    ▼                           │              │
  AMBIENCE STREAM ────▶│  (2) AIM ──────▶ ξ ∈ ℝ^D  ◀───── DRAN weights (K, V)         │
  (company/loc/time/   │      (recurrent     │         (~200 MB, exp storage)          │
   talk/digital/       │       residual      │                                         │
   sensory/recent)     │       multimodal    ▼                                         │
                       │       encoder)   (3) DRAN — L stacked attractor blocks        │
                       │                   │  • E = −exp(K^T ξ)                        │
                       │                   │  • per-block n_{l,i} input-dependent      │
                       │                   │    annealed steps                         │
                       │                   │  • kinematic gating (memory trigger)      │
                       │                   │  • multi-hop chaining  e_{l} → e_{l+1}    │
                       │                   │                                           │
                       │                   ▼                                           │
                       │              Memory Decoder (constrained decoding            │
                       │               grammar from Memorome → tokens t_rec,l)        │
                       │                   │                                           │
                       │                   ▼                                           │
                       │        Spontaneous Recall: Trigger → Candidate Gen →         │  ──▶ Proactive Memories
                       │           Personal Utility Module (rank/filter)              │       (surface to user)
                       │                                                           │
                       └──────────── RLHF (engagement signals ◀──── user) ───────────┘
```

핵심: **세支柱(AIM/DRAN/Memorome)이 아니라 세 저장 동선**으로 보면 clearer:
1. symbolic 저장 = Memorome (structured records + constrained-decoding grammar 원천)
2. subsymbolic 저장 = DRAN weights K, V (Hebbian `W = Σ xᵢ ⊗ ξᵢ` 로 직접 주입)
3. trigger = AIM state vector ξ 가 DRAN attractor convergence를 유도

### 핵심 메커니즘 4종 (상세는 발췌 §2·§8)

| 메커니즘 | 핵심 수식/동작 | 의미 |
|---|---|---|
| **Hebbian weight init** | `W = Σ xᵢ ⊗ ξᵢ` (ambience–memory pair 외적, no GD) | personalization이 fine-tuning 없이 weight 자체에 인코딩 → scalable personalization. Srinivasan 2023 thesis가 behavior matching으로 sufficiency 입증 |
| **Exponential storage** | `E = −exp(K^T ξ)` (modern Hopfield interaction) | 저장 용량이 pattern 수에 exponential → ~200 MB 가 lifetime patterns 보유 |
| **Multi-hop chaining** | ① ξ → block 1 → x₁ → block 2 → x₂ → … (최대 L) | 인간의 "one memory leads to another" 연상 사슬 모방 |
| **Kinematic gating** | convergence fast & clean → emit x_{l,i}; sluggish/fail → terminate | hallucination을 downstream filter가 아니라 **dynamics level에서 prevent** ("no convergence, no output") |

### Constrained decoding (Memorome 기반 final guardrail)

Memory Decoder는 Memorome이 compile 한 grammar(Willard & Louf 2023) 아래 동작 → 사용자의 실제 기록에 없는 memory는 token 단에서 emit 불가. **1.2% hallucination vs 36–54% LLM**의 직접 원인. kinematic gating(dynamics) + constrained decoding(grammar)의 **2단 hallucination 억제**가 구조적 강점.

### AIM (Ambient Intelligence Module)

다른 multimodal encoder와 차별점: 일반적인 시각/음성이 아니라 **subverbal signals** (whom the user is with, music playing, apps open, recent hours/days)를 융합. recurrent residual 로 최근 시간 창을 누적 → 현재 순간이 recent history에 couched. 출력 ξ ∈ ℝ^D 가 곧 query를 대체 ("environment is the query").

### Srinivasan (2023) — DRAN의 직접 precursor (Kreiman Lab / ETH Zürich)

- Hebbian attractor network 로 자연적 card-matching task working memory 모델링
- update 식: `M_t = λ M_{t-1} + η h_t h_t^T` (네트워크 상태 h_t 자신에 대한 outer-product, LeakyReLU + activation normalization)
- **behavior 신호(click 횟수, RT, memory decay) + intracranial neural 신호(novelty/familiarity/retrieval confidence, epilepsy 환자 전극)** 동시 매칭
- → (i) Hebbian init만으로 fine-tuning 없이 recall sufficient, (ii) attractor convergence kinematics가 해석 가능한 cognitive signal carry 한다는 2가지 claim 실험적으로 입증. DRAN의 "memory trigger event" gating은 이의 직접 후손.

### Engramme vs 4 경쟁 — paradigm 사분면

```
                                │
      Retrieval layer on LLM    │   New model class
      (query-gated recall)      │   (ambience-triggered recall)
  ──────────────────────────────┼─────────────────────────────
      Mem0 · Letta · Zep        │           Engramme (LMM)
      Supermemory               │            (DRAN + AIM + Memorome)
                                │
```

> 핵심 positioning: 다른 4사는 모두 "어떻게 표현·점수화·검색하는가"만 다를 뿐 동일 lane(retrieval substrate for an LLM-driven agent). Engramme은 lane 자체가 다름 — 1st primitive가 next-token generation이 아니라 **associative recall**.

상세 비교표·step-by-step 비교 → [발췌 §6·§7](../source/paper/Engramme_A_Technical_Review_2026_Samsung_NEXT.md)

## Experiments & Results

### Benchmark Datasets

- **Engramme 자체 비교**: frontier LLM(Gemini 2.5 Pro, Claude Opus 4.5, ChatGPT 5.2) with native memory connectors (Claude/ChatGPT "Company Knowledge", Gemini Google Workspace). real text ambiences (web browsing, email reading, document writing). **Engramme은 prompt 없음**. blinded pairwise top-memory 비교.
- **표준 memory 벤치마크 (경쟁사 비교용)**:
  - **LoCoMo** (Snap Research, ACL 2024) — 10개 long multi-session 대화, ~1,540 questions
  - **LongMemEval** — 500 questions / 6 categories (temporal reasoning, knowledge updates 포함)
  - **DMR** (MemGPT 제안 benchmark) — Letta와 Zep 비교용

### Setup

- LLM judges / LLM-as-judge (대부분 GPT-4o-mini 또는 GPT-4o). 평가 LLM이 vendor마다 상이 → 숫자는 directional 로 읽어야 함.
- Engramme: 1.7 s median end-to-end recall (memorome pre-built + streaming ambience, no query-time indexing).
- 경쟁: 각 vendor의 published 논문·기술 페이지에서 인용. Zep/Mem0 LoCoMo dispute은 공식 GitHub issue(#5, getzep/zep-papers)로 공개됨.

### Results

**Engramme 자체 벤치마크** (vs frontier LLM with connectors):

| Metric | Engramme | Gemini 2.5 Pro | Claude Opus 4.5 | ChatGPT 5.2 |
|---|---|---|---|---|
| median end-to-end recall latency | **1.7 s** | 13.2 s | 34.1 s | 49.4 s |
| speedup | — | 8× | 20× | 29× |
| blinded user preference win rate | — | 91% | 80% | 87% |
| hallucination rate | **1.2%** | 36–54% (frontier cohort) | | |

**표준 벤치마크 cross-vendor** (발췌 §5·§6 수치 — 비교 방향성 참고용):

| Vendor | LoCoMo | LongMemEval | DMR | p95 latency |
|---|---|---|---|---|
| Engramme | **미발표** | **미발표** | **미발표** | 1.7 s (자체) |
| Mem0 / Mem0g | 66.9 / 68.4 (new algo 85.0) | 49.0 (indep, GPT-4o) | 미발표 | 1.44 s / 2.59 s |
| Letta (MemGPT) | 74.0 (filesystem) | 미발표 | 93.4% | LLM roundtrip-dominant |
| Zep (Graphiti) | disputed (84 → 58.44 → 75.14) | 63.8 (indep, GPT-4o) | 94.8 / 98.2% | 3 s (GPT-4 Turbo) vs 30 s baseline |
| Supermemory | #1 per vendor (score 미공개) | 81.6 / 84.6 / 85.2% | 미발표 | sub-300 ms @100B+ tokens/mo |

### Findings & Implications

- **latency 차이의 원인**: LLM+connector은 매 query마다 agentic search(수십 초) + RAG(초) 결합. Engramme은 memorome pre-built + streaming ambience → query-time indexing/tool-use 없음. ambience가 1분 미만으로 evolve 하는 proactive recall에만 유의미한 차이.
- **hallucination 1.2%의 메커니즘**: kinematic gating(attractor 미수렴 시 emit 안 함) + constrained decoding(grammar가 Memorome 외 emit 차단)의 2단 억제. 다른 4사의 post-filter(graph/grammar/Update) 방식보다 mechanism-level.
- **경쟁 비교의 asymmetric 평가**: Supermemory가 LongMemEval-S 81.6–85.2%로 가장 높으나, Engramme은 동 벤치마크 미참여 → 동일한 battlefield 비교 결여. Zep-Mem0 LoCoMo dispute은 평가 setup sensitivity를 명확히 보여줌.
- **differentiator 의의**: Engramme의 4가지 독자적 differentiator(ambience-triggered recall, Hebbian personalization without fine-tuning, kinematic gating, on-device ~200MB)는 다른 4사가 구조적으로 close 못하는 갭. 단, vendor 자체 평가 한계 인정 필요.

## Analysis

### Strengths & Significance

1. **lane 자체가 다름** — 4개 경쟁이 모두 "retrieval layer on LLM" 사분면에서 세부 차이만 경쟁할 때, model class 자체를 교체하는 시도. LLM-with-memory 가 정답이라는 전제 자체에 대한 대안.
2. **과학적 lineage 가 unusually well-grounded** for the category — Kreiman Lab(Harvard Medical School)의 25+ 논문, 그 중 Srinivasan 2023 thesis가 (i) Hebbian no-fine-tune sufficiency (ii) attractor kinematics의 cognitive signal 해석성을 모두 실험 입증. 즉 DRAN의 핵심 2가지 claim이 카테고리 평균보다 훨씬 강력한 empirical 근거 위에 있음.
3. **hallucination 억제의 mechanism-level 접근** — post-hoc filter가 아니라 dynamics(no convergence, no output) + grammar(Memorome 외 차단)의 2단. 신뢰성 측면에서 구조적 강점.
4. **온디바이스 ~200MB** — cloud-first 4사와 다른 deployment 차원의 차별(personal memory의 privacy latency 가치에 부합).
5. **dark matter of memory** 개념 도입 — prompt 전제 구조의 본질적 한계를 정확히 명명. RAG/agentic search/fine-tuning 모두 해당 한계 공유.

### Limitations

- **표준 벤치마크 전무** — LoCoMo/LongMemEval/DMR 미발표. 자체 비교는 "LLM+connector" baseline 으로, purpose-built memory systems(Mem0g/Zep/Supermemory)와 head-to-head 아님. 가장 큰 credibility gap.
- **평가 비대칭성** — Engramme은 prompt 없이, LLM은 "retrieve the most relevant pieces of information" prompt 받음. fairness 검증 필요.
- **structured world knowledge 처리 불명확** — personal memory 특화. "what did my doctor say about this drug?" 처럼 personal + world knowledge 가 blend 되는 query 처리가 기술보고서에 서술 안 됨.
- **10¹² scale empirical 미검증** — exponential storage capacity는 수식적 주장이나, petabyte memorome 실제 동작 시 true vs false/missed trigger 비율(=UX 결정 변수) 미검증. Kreiman Lab 의 upcoming personal-memory benchmark series 가 적절 검증 수단.
- **constrained decoding breadth** — 1.2% hallucination 수치가 LoCoMo/LongMemEval 동일 task distribution 과 비교 가능한지 미확정.
- **RLHF tuning** — initial writes는 gradient-free(Hebbian)이나 engagement tuning은 end-to-end → "no fine-tuning" claim 부분 완화 필요.

### Future Work / Improvements

- **standardized evaluation** — 동일 LLM(judge GPT-4o-mini 등)에서 Mem0g/Zep/Supermemory와 head-to-head on LoCoMo·LongMemEval·DMR. "benchmark battlefield" 참여가 가장 시급.
- **personal-world knowledge blend** — Memorome + 외부 KG(world knowledge)의 통합 또는 routing 방안.
- **scaling 실증** — 10¹²-byte memorome 에서 convergence kinematics의 false/missed trigger ratio 측정.
- **공정 baseline 설계** — "prompt 없는 Engramme" vs "prompt 있는 LLM" 비교의 asymmetric setup 을同时也是 prompt 없는 경쟁사 액터와의 비교로 확장.
- **비용/privacy 분석** — on-device ~200MB 의 장점을 정량화한 비용/latency/privacy 비교표.

## References

- 분석 원본: [Engramme_A_Technical_Review_2026_Samsung_NEXT.pdf](../source/paper/Engramme_A_Technical_Review_2026_Samsung_NEXT.pdf) (Samsung NEXT, 2026-04-17)
- 발췌: [Engramme_A_Technical_Review_2026_Samsung_NEXT.md](../source/paper/Engramme_A_Technical_Review_2026_Samsung_NEXT.md)
- Engramme 기술 보고서: engramme.com/research (ref. 1, "Looking Forward and Backward in Time with Dynamic Recurrent Attractor Networks", 2025)
- Srinivasan 2023 (DRAN precursor, Kreiman Lab / ETH Zürich): ref. 2
- Kreiman Lab publications: klab.tch.harvard.edu/publications
- 경쟁사: Mem0 (arXiv:2504.19413, ECAI 2025), Letta/MemGPT (arXiv:2310.08560), Zep/Graphiti (arXiv:2501.13956), Supermemory (supermemory.ai/research)
- 표준 벤치마크: LoCoMo (Snap, ACL 2024), LongMemEval (500Q/6cat), DMR (MemGPT 제안)
