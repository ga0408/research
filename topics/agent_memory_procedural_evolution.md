> [topic]

# Agent Memory: 대화 기반 → 관찰 기반 절차적 기억으로의 진화

## Scope

- **agentic-memory (우리 개발물)**: `/home/jjlee/workspace/AgenticFW/memory/agentic-memory` — L0→L1→L2 3계층, Signal-First 2-pass 추출, Phase E skill promotion, dreaming
- [Memora (Microsoft, ICML 2026)](../report/[paper][git]_Memora_A_Harmonic_Memory_Representation_Balancing_Abstraction_and_Specificity_2026_ICML.md) — Harmonic Memory Representation, cue anchor, policy-guided retrieval, LoCoMo/LongMemEval SOTA
- [AgenticSTS (Alaya Lab, arXiv 2026)](../report/[paper]_AgenticSTS_A_Bounded-Memory_Testbed_for_Long-Horizon_LLM_Agents_2026_arxiv.md) — Bounded-memory contract, 5 typed layers, Slay the Spire 2 testbed, mistake-driven skill discovery

---

## 1. 공통 High-Level 구조: episodic → semantic → procedural 3단계 진화

세 시스템 모두 **"관찰/경험 → 사실 추출 → 반복 패턴 → 일반화된 절차"**라는 동일한 진화 경로를 가진다. 각 단계가 다른 이름으로 불릴 뿐, 구조적 동형(isomorphism)이 존재한다.

```
                  Episode (경험)         Semantic Fact (사실)       Procedural Skill (절차)
                  ─────────────          ────────────────           ──────────────────────
agentic-memory    L0 episodic_summary    L1 atomic items            L2 personal_skill
                  (Pass 1 SESSION_TRIAGE)(Pass 2 UNIFIED_EXTRACTION) (Phase E: plan cluster → skill)

Memora            Episodic Memory        Factual Memory             (명시적 skill layer 없음)
                  (segment 단위)         (primary abstraction+value) (consolidation으로 대체)

AgenticSTS        L4 Episodes            L3 Rules (정적)            L5 Skills
                  (postrun, per-combat)  (card/relic/monster data)  (mistake-driven / template)
```

### 핵심 통찰: AgenticSTS의 L4/L5 lifecycle이 우리의 episodic → skill pipeline과 구조적으로 동일

| 단계 | agentic-memory | AgenticSTS |
|------|----------------|------------|
| **경험 단위** | session (사용자 발화 + 도구 실행) | combat (게임 내 전투 1회) |
| **저장 시점** | turn/session 종료 후 | run 종료 후 (postrun) |
| **경험 내용** | episodic_summary (narrative 요약) + plan item (실행 계획+결과) | episode summary (round별 요약: HP loss, 전략적 의미) |
| **누적 방식** | plan item에 resource_ids 누적, reinforcement_count 증가 | 동일 키(char×ascension×act×enemy)에 여러 run의 episode append |
| **일반화 트리거** | T1(reinforce≥3) / T2(cluster≥3) / T4(correction/outcome) | loss_ratio > baseline (실수 감지) |
| **검증** | (없음 — 트리거 조건만으로 승급) | A/B resample(B=3, 2/3 개선+0 harmful) → 4-level write gate |
| **스킬 저장** | L2 category(is_personal_skill=True) + trigger_query embedding | skills.json (trigger condition + prose policy) |
| **스킬 발동** | `/memories/skills/search` cosine 매칭 | per-decision trigger match (enemy+threat+hand) |

> **우리 시스템에 없는 것: 승급 전 검증(A/B verification)**. AgenticSTS는 skill 후보가 "이것이 있었으면 어땠을까"를 3번 resample하여 실제 개선을 확인한다. 우리는 trigger 조건(reinforcement count, cluster size)만으로 승급하므로, 품질 낮은 skill이 승급될 위험이 있다.

---

## 2. Turn = State: 대화 턴과 agent 결정의 동형성

### 2.1 대화 기반 시스템에서의 state

현재 agentic-memory의 1개 처리 단위는 **사용자 발화 1개(또는 1세션)**이다:
```
state = (user utterance, conversation history, working memory active window)
        → Pass 1 SESSION_TRIAGE → signals + episodic_summary
        → Pass 2 UNIFIED_EXTRACTION → L1 atomic facts
        → Phase E check_promotion → L2 skill (조건부)
```

### 2.2 AgenticSTS에서의 state

AgenticSTS의 1개 결정 단위는 **게임 state 1개**이다:
```
state = (hand, enemy intent, HP, energy, turn number, relics, deck)
        → per-decision composition: L1(protocol) + L2(schema) + L3(rules) + L4(episodes) + L5(skills)
        → LLM decision → action
```

### 2.3 동형성과 차이

```
대화 기반 state                    Agentic state (목표)
──────────────────                 ──────────────────────
user utterance (text)         ←→   perception observation (sensor/device/env data)
conversation history          ←→   action history (이전 결정들의 요약)
working memory active window  ←→   bounded context window (~5k tokens)

Pass 1 SESSION_TRIAGE         ←→   state triage (관찰에서 signal 추출)
Pass 2 UNIFIED_EXTRACTION     ←→   observation extraction (관찰에서 fact 추출)
episodic_summary 저장          ←→   episode 저장 (postrun)
Phase E skill promotion       ←→   L5 skill distillation (mistake-driven)
```

**핵심 차이**: 대화 시스템은 "사용자가 말한 것"이 state의 전부이지만, agentic 시스템에서는 **agent가 관찰한 것(perception)**이 state를 구성한다. AgenticFW에서는 perception을 통해 관측된 정보 및 주변 기기 정보가 state의 구성 요소가 된다.

현재 agentic-memory의 Pass 1/2 prompt는 **대화 transcript**를 입력으로 설계되어 있다. 관찰 기반 시스템으로 전환하려면:
- 입력이 `text://` URI의 단일 텍스트가 아니라, 다중 센서/기기 상태의 구조화된 데이터
- SESSION_TRIAGE가 "이 관찰에서 어떤 signal이 있는가?"를 판단 (예: 기기 이상, 루틴 편차, 사용자 상태 변화)
- UNIFIED_EXTRACTION이 관찰에서 fact를 추출 (예: "거실 온도 28도", "사용자 10분 전 귀가")

---

## 3. Episodic 저장: 매 turn 종료 시, 그러나 semantic 추출이 차별점

### 3.1 세 시스템의 episodic 저장 비교

| | agentic-memory | Memora | AgenticSTS |
|---|---|---|---|
| **저장 단위** | session 1개 | segment 1개 | combat 1개 |
| **저장 시점** | memorize 호출 시 | `add()` 호출 시 | run 종료 후 (postrun) |
| **내용** | LLM 요약 narrative | segment 원문 또는 LLM 요약 | analysis tier 모델이 combat 전체를 분석한 요약 |
| **일반 검색 포함** | ✗ (제외됨) | ✓ (검색은 factual만 반환, episodic은 app 레이어에서 참조) | ✓ (L4 slot으로 주입) |
| **cross-session** | daily_log에 날짜별 누적 | segment 내에서만 | 동일 키에 여러 run 누적 |

### 3.2 agentic-memory의 차별점: semantic 추출 고도화

Memora와 AgenticSTS는 episodic를 "raw 경험의 보관"으로 처리하지만, agentic-memory는 **Pass 2 UNIFIED_EXTRACTION**으로 episodic와 동시에 5종 typed fact(profile, event, knowledge, behavior, tool)를 추출한다.

```
[agentic-memory]  session → Pass 1: episodic_summary + signals
                          → Pass 2: {profile, event, knowledge, behavior, tool, plan} 동시 추출

[Memora]          segment → episodic memory (별도)
                          → factual memory (별도, LLM 추출)

[AgenticSTS]      run → L4: episode summary (postrun, 별도)
                        L5: skill (mistake-driven, 별단계)
```

이 구조의 장점은 **episodic 경험이 semantic fact로 동시에 분해**된다는 점이다. "사용자가 어제 병원에 갔다"는 episodic event로 저장되면서 동시에 "사용자의 주치의는 OOO이다"라는 profile fact로도 추출된다. 이는 벤치마크(LoCoMo류)에서 강점으로 작용하지만, 관찰 기반 시스템에서는 이 추출 구조를 **관찰 데이터**에 맞게 재설계해야 한다.

---

## 4. Episodic 반복 → Skill 추출: 세 시스템의 접근 비교

### 4.1 Skill 추출 파이프라인 비교

```
                    [경험 누적]           [트리거]              [검증]              [저장]
                    ───────────           ─────────             ──────              ──────
agentic-memory      plan items            T1: reinforce ≥3     (없음)              L2 category
                    resource_ids 누적     T2: cluster ≥3                           trigger_query embed
                    reinforcement_count   T4: correction/outcome                    (Option D)

AgenticSTS          L4 episodes           loss_ratio >          A/B resample (B=3)  skills.json
                    동일 키에 누적         baseline              2/3 개선 + 0 harmful → trigger condition
                                          (실수 감지)           + 4-level write gate   + prose policy

Memora              (명시적 skill 없음 — factual consolidation으로 대체)
                    intelligent upsert:   유사 메모리 발견시     LLM update decision  병합된 entry
                    history 누적          sim ≥ threshold       (should_update?)
```

### 4.2 핵심 차이: 검증 메커니즘

**AgenticSTS의 가장 큰 차별점은 skill 후보를 실제로 검증한다는 것**이다:

1. **Mistake detection**: combat의 loss_ratio를 per-enemy baseline과 비교 → 실수가 있었음을 객관적으로 감지
2. **A/B resample**: "이 skill이 있었으면 어땠을까?"를 3번 재실행 → 2/3 이상 개선 + 0번 harmful이어야 통과
3. **4-level write gate**: cosine 유사도(중복 merge) → Jaccard(token 겹침) → LLM judge(품질) → optional reap

우리의 Phase E는 trigger 조건(reinforcement count, cluster size, correction signal)만으로 승급한다. 이는 **"반복되었으니 중요하다"**는 빈도 기반 판단이지, **"이 skill이 실제로 효과적인가"**를 검증하지 않는다.

> **Insight**: AgenticSTS의 A/B verification은 도메인(게임)에 특화된 것처럼 보이지만, 본질은 **counterfactual evaluation**이다. "이 절차를 적용했을 때 결과가 나아지는가?" 이 개념은 care service에도 적용 가능하다 — "이 루틴을 제안했을 때 사용자가 수락/개선했는가?"

### 4.3 Skill 발동 메커니즘 비교

| | agentic-memory | AgenticSTS |
|---|---|---|
| **발동 방식** | `/memories/skills/search` API 호출 (query-AITTU) | per-decision trigger match (자동) |
| **매칭 기준** | trigger_query embedding cosine | trigger condition (enemy_names, threat_levels, requires_hand_capabilities) |
| **주입 방식** | 검색 결과로 반환 (agent가 선택) | user message의 L5 slot에 자동 주입 |
| **proactive?** | ✗ (요청 시만) | ✓ (매 결정마다 자동 체크) |

AgenticSTS의 trigger는 **구조화된 조건**(enemy name, threat level, hand capability)이고, 우리는 **의미적 유사도**(trigger_query embedding)이다. 관찰 기반 care service에서는 "현재 시간 + 기기 상태 + 사용자 위치" 같은 구조화된 trigger가 더 적합할 수 있다.

---

## 5. Episodic 검색: 유사 상황에서의 과거 경험 회상

### 5.1 현재 agentic-memory의 episode 검색

현재 구조에서 plan items는 plan-only 검색(`/memories/plan/search`)으로만 recall된다:
- dedupe: sim ≥ 0.85로 기존 plan 탐색 (memorize 시점)
- promotion cluster: sim ≥ 0.75로 유사 plan 탐색 (promotion 시점)
- 일반 RAG 검색에서 `plan`과 `episodic_summary`는 **제외됨**

이는 **reactive**(memorize/요청 시점)이지 **proactive**(관찰 시점 자동 recall)가 아니다.

### 5.2 AgenticSTS의 episode 검색

AgenticSTS L4는 **새 run에서 같은 enemy를 만났을 때** 자동으로 과거 run들의 요약을 retrieve한다:
- 키: (character × ascension × act × enemy class) — **구조화된 정확 매칭**
- 여러 run의 episode가 누적되어 "패턴"을 형성
- L4 slot으로 per-decision user message에 **자동 주입**

이는 **proactive recall**이다. agent가 새로운 상황에 직면했을 때, 유사한 과거 경험을 자동으로 끌어온다.

### 5.3 Memora의 접근: policy-guided multi-hop

Memora는 검색 자체를 MDP로 정식화하여, 단일 query가 놓치는 관련 메모리를 frontier expansion으로 추적한다:
- cue anchor의 다대다 구조를 따라가며 multi-hop 의존성 포착
- PromptedPolicyRetriever: EXPAND(frontier 확장) → RE_QUERY(재검색) → STOP
- "relative answer" 탐지: W가 포인터만 제공하면 구체값을 추적

### 5.4 관찰 기반 시스템에서의 proactive recall 설계 방향

```
[현재: reactive]                    [목표: proactive]
사용자 질문 → 검색 → 결과 반환      관찰 → state 유사도 판단 → 과거 episode 자동 recall → context 주입

적용 방안:
1. state fingerprint 생성: (시간대, 기기 상태, 사용자 활동, 환경 데이터) → 구조화된 키 또는 embedding
2. episode store에 state fingerprint 인덱스 추가 (AgenticSTS의 char×ascension×act×enemy와 유사)
3. 관찰 시마다 state fingerprint로 episode 검색 → working memory에 자동 주입
4. Memora식 multi-hop: 현재 state와 관련된 episode → 그 episode에 link된 다른 episode 확장
```

---

## 6. AgenticSTS 결과 활용 가능성

### 6.1 직접 활용 가능한 산출물

| 산출물 | 활용 방안 |
|--------|-----------|
| 298 trajectory 아카이브 (condition tag + SHA snapshot) | bounded-memory contract의 효과를 우리 시스템과 비교할 때 benchmark reference로 활용 |
| 5-condition ablation 방법론 | 우리 시스템의 L0/L1/L2/Phase E를 독립적으로 on/off하는 ablation 설계에 참고 |
| Auto-mode ascension ladder | 점진적 난이도 증가 환경에서 memory 시스템의 한계점 측정 방법론 참고 |

### 6.2 핵심 발견의 적용 가능성

**발견 1: "Skill layer 존재 자체가 효과"** (Mode B template = Mode A human-authored = 6/10)
- 우리 시스템에 대한 시사점: Phase E skill의 **prose 품질**보다 **skill layer의 존재와 trigger 매칭**이 더 중요할 수 있다. 이는 skill 생성 LLM 호출 비용을 줄이고(간단한 template로 대체), trigger 매칭 정확도에 투자하는 것이 더 효과적일 수 있음을 시사한다.

**발견 2: Bounded-memory contract (항상 ~5k tokens)**
- 우리 시스텀의 inject categories(3종, 항상 system prompt에 prepend)가 이미 bounded contract의 일부이다.
- 확장 방향: L1 item retrievl + L2 category summary + skill을 하나의 bounded "per-decision composition"으로 통합.
- 현재는 검색이 query-driven이지만, 관찰 event-driven로 전환 시 매 event마다 typed slice를 조립하는 구조 필요.

**발견 3: Postrun extraction (run 종료 후 batch 처리)**
- 우리의 **Dreaming** pipeline이 이미 이 역할을 한다 (NIGHTLY: dirty resummary + deep forget).
- AgenticSTS의 postrun analysis tier를 Dreaming의 DELAYED timing으로 구현 가능: "최근 N시간의 관찰에서 패턴/실수 추출".

**발견 4: Token 효율 (66-90×)**
- accumulating-context 방식 대비 bounded contract의 효율성은 care service의 장기 실행(long-horizon) 환경에서 직접 적용 가능.
- 24시간 care agent가 매 결정마다 과거 관찰을 전부 context에 넣으면 token 폭발이 발생하므로, bounded contract는 필수.

### 6.3 직접 활용의 한계

- **도메인 차이**: Slay the Spire 2는 closed-rule, turn-based, text-readable 게임. care service는 open-ended, real-time, multi-modal.
- **State 표현**: 게임은 완전한 state가 주어지지만, care service는 perception의 불완전성(noise, missing data)이 있다.
- **평가 지표**: 게임은 win rate로 명확하지만, care service는 "적절한 개입"을 어떻게 측정할 것인가가 미정의.
- **Skill 검증**: 게임은 resample로 counterfactual을 직접 테스트 가능하지만, care service는 동일 상황 재현이 어려움.

---

## Comparison

### 전체 비교표

| 항목 | agentic-memory (ours) | Memora | AgenticSTS |
|------|----------------------|--------|------------|
| **메모리 계층** | L0 Signals → L1 Atomic → L2 Category | Episodic + Factual (Harmonic) | L1-L5 Typed Layers |
| **추출 입력** | 대화 transcript (text) | 대화/문서 (text) | 게임 state (구조화된 JSON) |
| **추출 패스** | 2-pass (SESSION_TRIAGE + UNIFIED_EXTRACTION) | 1-pass (factual extraction + cue) | postrun analysis tier |
| **Semantic 추출** | 5종 typed fact (profile/event/knowledge/behavior/tool) | factual memory (primary abstraction + cue anchor) | L3 rules (정적 데이터) |
| **Episodic 저장** | per-session, 검색 제외 | per-segment, app 레이어 참조 | per-combat, L4 slot 주입 |
| **Consolidation** | dedupe (sim≥0.95 LLM) + category rollup | intelligent upsert (history + merge) | episode 누적 + skill merge (cosine/Jaccard) |
| **Skill 추출** | Phase E (T1/T2/T4 trigger) | (없음) | L5 (mistake-driven + A/B verify) |
| **Skill 검증** | 없음 | (N/A) | A/B resample + 4-level gate |
| **Skill 발동** | API 검색 (reactive) | (N/A) | per-decision trigger (proactive) |
| **검색** | 7-step RAG + cross-encoder + BM25 | semantic + policy MDP + GRPO | per-decision typed composition |
| **Context 관리** | WorkingMemory (token budget) | (외부) | bounded contract (~5k 항상) |
| **LLM 호출 (쓰기)** | 2-6회/session | 5-8회/segment | 0회 (postrun 별도) |
| **LLM 호출 (읽기)** | 0-4회/query | 1-6회/query | 0회 (composition만) |
| **배치 처리** | Dreaming (NIGHTLY) | (없음) | postrun analysis + evolution tier |
| **벤치마크** | custom scenarios | LoCoMo, LongMemEval | Slay the Spire 2 (game) |
| **저장소** | PostgreSQL + pgvector (HNSW) | ChromaDB (vector) | JSONL files |

### 구조적 동형 관계

```
agentic-memory             AgenticSTS                Memora
──────────────              ──────────                ──────
L0 Signals (SESSION_TRIAGE) ≅  (없음 — state가 곧 signal)
L1 Atomic (UNIFIED_EXTRACTION) ≅  L3 Rules + L4 Episodes  ≅  Factual Memory (extraction)
L2 Category (rollup)        ≅  (없음)
L2 personal_skill (Phase E) ≅  L5 Skills                ≅  (consolidation으로 대체)
WorkingMemory (active/archive) ≅  Bounded context (~5k)  ≅  (외부)
Dreaming (NIGHTLY)          ≅  Postrun analysis tier     ≅  (없음)
(plan search sim≥0.85)      ≅  L4 episode retrieval       ≅  Policy MDP (frontier expand)
```

---

## Synthesis

### 결론 1: 대화 → 관찰 전환의 핵심은 "입력 representation"과 "state fingerprint"

agentic-memory의 Pass 1/2 파이프라인 구조(episodic → semantic → skill)는 관찰 기반 시스템에서도 유효하다. 전환의 병목은 **추출 파이프라인이 아니라 입력 representation**이다:

- 현재: `text://{timestamp}` URI의 단일 텍스트 입력
- 목표: 다중 관찰 스트림(센서, 기기, 환경)의 구조화된 입력 + state fingerprint

AgenticSTS의 state는 (hand, intent, HP, energy, turn)으로 구조화된 JSON이다. care service의 state는 (시간대, 실내 온도, 기기 ON/OFF 상태, 사용자 위치, 활동)이 될 것이다. Pass 1 SESSION_TRIAGE가 이 구조화된 state를 입력받아 "이 관찰에서 어떤 signal이 있는가?"를 판단하도록 재설계해야 한다.

### 결론 2: Skill 검증 메커니즘 도입 필요

AgenticSTS의 가장 강력한 기여는 **skill 후보를 counterfactual로 검증**하는 것이다. 우리의 Phase E는 빈도/패턴 기반 trigger만으로 승급하므로:

- **단기 개선**: skill 승급 시 LLM judge 품질 평가 추가 (AgenticSTS의 4-level write gate 중 3번째 레벨)
- **중기 개선**: "이 skill을 적용했을 때의 결과"를 추적하는 feedback loop 추가 — skill 주입 후 사용자 수락/거절/개선 결과를 역수집하여 skill 품질을 지속 평가

### 결론 3: Proactive recall 구조 설계

현재 검색은 query-driven(reactive)이다. 관찰 기반 care service에서는 event-driven(proactive) recall이 필요하다:

- **State fingerprint**: 관찰에서 구조화된 키 생성 (AgenticSTS의 char×ascension×act×enemy와 유사하게, 예: user_id×time_slot×device_state×activity)
- ** episode store에 fingerprint 인덱스 추가**
- **관찰 시마다 자동 recall**: 현재 state의 fingerprint로 과거 episode 검색 → working memory에 주입
- **per-decision bounded composition**: AgenticSTS처럼 매 결정마다 typed slice(protocol + rules + episodes + skills)를 조립하여 ~5k token context 유지

### 결론 4: Bounded-memory contract 철학 도입

AgenticSTS의 핵심 통찰 — "memory는 얼마나 많이 저장하느냐가 아니라, 각 결정에 무엇을 보여주느냐의 contract" — 는 care service의 장기 실행 환경에서 필수적이다. 우리의 inject categories(3종)가 이미 bounded contract의 시초이지만, 이를 확장하여:

- L1 item + L2 category + L4 episode + L5 skill을 하나의 **per-decision composition**으로 통합
- run 길이(24시간 care 등)와 무관하게 항상 일정한 context 크기 유지
- 각 layer를 독립적으로 on/off 가능한 ablatable 구조로 설계 (AgenticSTS의 evaluation handle 참고)

### 결론 5: AgenticSTS 결과의 직접 활용은 제한적이지만 방법론은 차용 가능

298 trajectory 아카이브는 게임 도메인 특화이므로 직접 활용이 어렵다. 하지만:
- **Ablation 방법론**: 우리의 L0/L1/L2/Phase E를 독립 toggle하는 실험 설계
- **Bounded contract 측정**: context 크기가 run 길이와 무관하게 일정한지 검증
- **Skill layer 효과 분리**: "skill 존재 자체가 효과인가, skill 품질이 효과인가"를 분리 측정
- **난이도 ladder**: 점진적 복잡도 증가 환경에서 memory 시스템 한계점 측정

이 방법론들을 care service 도메인에 맞게 재설계하면, 관찰 기반 agentic memory의 효과를 정량적으로 평가할 수 있는 프레임워크를 구축할 수 있다.
