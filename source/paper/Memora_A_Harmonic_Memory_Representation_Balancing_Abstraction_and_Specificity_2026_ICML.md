# Memora: A Harmonic Memory Representation — 핵심 발췌

> 출처: [분석 문서](../../report/[paper][git]_Memora_A_Harmonic_Memory_Representation_Balancing_Abstraction_and_Specificity_2026_ICML.md) / 원본: [arXiv:2602.03315](https://arxiv.org/abs/2602.03315)

## Memory Construction (§3)

### Problem Formulation (§3.1)

메모리 관리를 구조화된 store의 유지 문제로 정식화:

```
Fm: D → M          (memory construction function: raw data → structured memory set)
Q(q, M) → Mq       (retrieval function: query → relevant subset Mq ⊆ M, |Mq| ≪ |M|)
```

핵심 설계 목표: Mq의 관련성을 최대화하면서 크기·지연시간 최소화. 고수준 semantic scanning과 저수준 contextual lookup을 모두 지원하는 표현 필요.

**Memora의 핵심 혁신**: "what is stored"와 "how it is accessed"를 decouple. memory content는 rich하게 유지하고, 별도의 structural layer(primary abstractions + cue anchors)가 검색 신호를 담당.

### Segmentation (§3.3)

```
S(d) = {s1, ..., sk}     # data item d를 semantically coherent segments로 분해
```
- 각 segment si가 memory construction의 입력 단위.
- 단일 segment가 여러 memory entry를 생성할 수 있음.
- 구현: 비정형 narrative는 prompt-based extraction, 정형 파일은 structural hierarchy(문서 헤더 등) 활용.

### Episodic Memory (§3.4)

```
ei = E(si)              # 각 segment si에 대해 episodic memory 생성
```
- 해당 segment에서 파생된 모든 memory entry의 공유 narrative grounding.
- 표현은 유연: extracted high-level summary(참여자·의도·시간 범위) 또는 raw segment text 그대로(정확한 표현 보존).
- 검색 시 같은 episodic memory에 속한 entry들을 grouping해 narrative coherence 제공 → multi-step reasoning 지원.

### Primary Abstraction (§3.5)

메모리를 안정적인 개념 중심으로 조직해 fragmentation 방지. 2단계: extraction → consolidation.

**Extraction**: 새 segment s에서 candidate memory entries 생성
```
Fa(s) = {mi}^N_{i=1},   mi = (ai, vi)
# ai: primary abstraction (메모리가 근본적으로 무엇에 대한 것인지)
# vi: memory value (구체적 details)
```

**Consolidation**: candidate를 기존 store M에 통합. 새 candidate (ai, vi)에 대해:

(1) top-k 유사 기존 entry 검색:
```
R(ai) = TopK_{m∈M} sim(ai, am; k)
```

(2) similarity threshold γ로 필터링:
```
U(ai) = {m ∈ R(ai) | sim(ai, am) ≥ γ}
```

(3) LLM-based selection function J가 동일 개념인지 판단:
```
m⋆(ai) = J(ai, U(ai))    # match → target entry 반환, no match → ∅
```

(4) create-or-update rule:
```
mi = { Update(m⋆(ai), ai, vi),  m⋆(ai) ≠ ∅   # 기존 entry에 vi 병합, abstraction a'i로 갱신 가능
       Create(ai, vi),          m⋆(ai) = ∅ }  # 신규 entry 생성
```

→ 각 entry가 단일 primary abstraction에 anchor되고, 의미적으로 정렬된 새 정보가 점진적 통합됨. 불필요한 중복 없이 기존 개념을 풍부하게 하거나 필요시에만 새 abstraction 생성.

### Cue Anchors (§3.6)

Primary abstraction이 의도적으로 coarse하므로, fine-grained semantic hook으로 cue anchor를 추가.

```
Fc(ai, vi) = {cij}^{|Ci|}_{j=1},   cij ∈ Ci
```
- 각 cue anchor는 memory content의 salient aspect/attribute/contextual perspective.
- 형식: **[Main Entity/Topic] + [Key Aspect]** (2-4어)
- **non-exclusive, many-to-many**: 한 entry에 여러 cue, 같은 cue가 여러 entry에 걸쳐 등장 가능.
- 기존 anchor 존재 시 link만 추가, 없으면 새 anchor 생성.
- memory entry 삭제/병합 시 cue–memory link 갱신, 모든 연관이 끊긴 anchor는 자동으로 삭제되어 cue 집합이 compact하게 유지됨.

### 통합 구조: Implicit Memory Graph

Primary abstractions(1:1) + cue anchors(n:m)가 결합해 **explicit edge 없이 implicit memory graph** 형성. 같은 cue를 공유하거나 abstraction-level 관계가 있는 entry들이 연결. 이 구조가 retrieval 시 단순 similarity를 넘어 구조적 연결성으로 multi-hop 의존성 포착을 가능케 함.

## Policy-Guided Memory Retrieval (§4)

### Retrieval as MDP (§4.1)

정적 semantic search는 multi-hop 의존성 포착에 실패. Memora는 검색을 **Markov Decision Process**로 정식화.

**State** (step t):
```
st = (qt, Wt, Ft, bt)
# qt: current query (REFINE로 갱신 가능)
# Wt: working set — 지금까지 검색된 memory entries
# Ft: frontier — Wt에 link된 후보 중 아직 검색 안 된 것들 (확장 가능성 관찰)
# bt: remaining retrieval budget
```

**Actions** (3개 atomic operations):
```
REFINE: 현재 query가 부족/부정렬 시 재생성·재구성 (검색 전략 pivot)
EXPAND: frontier Ft에서 관련 memory 선택 → working set Wt에 추가
STOP:  충분한 정보 수집 시 종료
```

**Transition**:
```
Apply(at, st, S) → st+1
Wt+1 = Wt ∪ ΔWt
Ft+1 = UpdateFrontier(Ft, ΔFt)     # 새로 검색된 item들의 neighbor 추가
bt+1 = bt - Cost(at)
```

종료 조건: STOP 선택 또는 budget 소진. 최종 Wt가 Mq로 반환됨.

### Algorithm 1: Policy-Guided Sequential Retrieval

```
Require: Query q, memory system S, policy πθ, budget B, max steps T
1:  q0 ← q, M0 ← ∅, b0 ← B
2:  F0 ← InitFrontier(q0, S)           # 초기 검색 + frontier 구축
3:  for t = 0, 1, ..., T-1 do
4:      st ← (qt, Wt, Ft, bt)          # 관찰
5:      at ∼ πθ(· | st)                # 정책이 action 샘플링
6:      if at = STOP or bt ≤ 0 then break
7:      (ΔWt, ΔFt, qt+1) ← Apply(at, st, S)   # action 실행
8:      Wt+1 ← Wt ∪ ΔWt
9:      Ft+1 ← UpdateFrontier(Ft, ΔFt)
10:     bt+1 ← bt − Cost(at)
11: end for
12: Mq ← Wt
13: return Mq
```

### Group-Relative Policy Updates / GRPO (§4.2, Appendix C)

정책 πθ 구현: prompt-guided LLM(zero-shot)부터 fully trained model까지 다양. 비용-정보이득 균형이 어려워 GRPO로 최적화.

**Trajectory 생성**: query q에 대해 G개 trajectory 샘플링
```
τ^(i) = {(st^(i), at^(i))}^{T_i}_{t=0}
```

**Trajectory scoring** (3기준):
```
Ground(τ) = JUDGE_ground(q, W)                       # groundedness (답이 검색 메모리로 지지되는가)
Redund(τ) = (1/|W|²) Σ_{mi,mj∈W} I[sim(mi,mj) > δ]   # redundancy 패널티
Cost(τ) = Σ_t Cost(at)                               # 검색 비용

J(τ) = w1·Ground(τ) − w2·Redund(τ) − w3·Cost(τ)      # 종합 스코어
```

**Group-relative advantage** (절대 스코어 대신 그룹 내 상대 비교):
```
Ã(i) = J(τ^(i)) − (1/G) Σ_{i'=1}^{G} J(τ^(i'))      # zero-mean within group
```

**Policy update** (positive advantage trajectory의 action 확률 증가):
```
L_GR(θ) = − Σ_{i=1}^{G} Ã^(i) Σ_t log πθ(at^(i) | st^(i))

# KL 정규화 (optional, policy drift 방지):
L(θ) = L_GR(θ) + β Σ_t KL(πθ(·|st) ‖ πref(·|st))
```

→ sparse supervision 하에서도 preference-based 최적화 가능. MDP 기반 sequential retrieval과 자연 정렬.

### Unifying Theory: RAG & KG as Special Cases (Appendix D)

**Theorem D.1 (Flat RAG)**: chunk = memory entry, abstraction = content(a=v=s), cue = 없음, action = {QUERY_A, STOP}, policy가 QUERY_A 한 번 후 STOP → 단일 step RAG과 동일.

**Theorem D.2 (Implicit KG)**: cue space = entity space(V), 각 entry에 단일 cue(entity) 연결, 유사도 기반 L-hop traversal → implicit KG의 L-hop neighborhood 검색과 동일.

**Theorem D.3 (Explicit KG)**: cue anchors = entities + relations(V∪E), cue–cue traversal이 KG edge 구조를 mirror → explicit KG retrieval과 동일.

→ Memora는 RAG·KG의 엄격한 일반화. mixed-key retrieval과 abstraction-first scoping으로 더 풍부한 검색 행동·효율성 제공.

## 실험 결과 상세 (§5)

### Table 1: LoCoMo 카테고리별 성능 (LLM-as-Judge score)

| Method | Multi-hop | Temporal | Open-domain | Single-hop | Overall |
|---|---|---|---|---|---|
| Full Context | 0.766 | 0.819 | 0.500 | 0.885 | 0.825 |
| RAG | 0.557 | 0.548 | 0.458 | 0.710 | 0.633 |
| HippoRAG | 0.390 | 0.224 | 0.510 | 0.587 | 0.471 |
| Zep | 0.537 | 0.602 | 0.438 | 0.669 | 0.616 |
| Mem0 | 0.624 | 0.660 | 0.500 | 0.677 | 0.653 |
| LangMem | 0.710 | 0.508 | 0.590 | 0.845 | 0.734 |
| Nemori | 0.751 | 0.776 | 0.510 | 0.849 | 0.794 |
| **Memora (S)** | 0.784 | 0.851 | 0.594 | 0.900 | 0.849 |
| **Memora (P)** | 0.787 | 0.866 | 0.594 | 0.918 | **0.863** |

### Table 2: LongMemEval

| Method | Context length | Avg Accuracy |
|---|---|---|
| Full Context | 115k | 65.6% |
| Nemori | 3.7-4.8k | 74.6% |
| **Memora (S)** | 2.1k | 83.8% |
| **Memora (P)** | 2.9k | **87.4%** |

### Table 3: Component build-up ablation (LoCoMo Overall LLM)

| Configuration | Score | 설명 |
|---|---|---|
| Memora w/o abstraction (= Mem0) | 0.653 | abstraction layer 제거 → Mem0로 퇴화 |
| + primary abstraction (no update) | 0.795 | consolidation 없이 추출만 |
| + update | 0.801 | create-or-update rule 추가 |
| + semantic retriever (full) | 0.849 | cue anchor + semantic 검색 |
| + policy retriever (full) | **0.863** | cue anchor + policy 검색 |

→ abstraction layer만 추가해도 0.653→0.795 (+0.142). update 추가로 +0.006. cue+semantic +0.048. policy +0.014.

### Table 4: Memory granularity ablation (LoCoMo, LLM score + Avg Tokens)

| Retriever | Memory Type | LLM Score | Avg Tokens |
|---|---|---|---|
| **Policy** | Episodic (Segment) + Factual | **0.863** | 1,853 |
| Policy | Episodic (Segment) only | 0.851 | — |
| Policy | Episodic (Segment) + Factual w/o cue | 0.850 | — |
| Policy | Episodic (Extracted) + Factual | 0.844 | — |
| Policy | Factual only | 0.833 | — |
| **Semantic** | Episodic (Segment) + Factual | 0.849 | 8,499 |
| Semantic | Episodic (Segment) only | 0.844 | 6,624 |
| Semantic | Episodic (Segment) + Factual w/o cue | 0.850 | 8,425 |
| Semantic | Episodic (Extracted) + Factual | 0.831 | 4,467 |
| Semantic | Factual only | 0.833 | — |

핵심 발견:
- **Policy retriever가 cue anchor 없으면 semantic과 동일** → policy의 이점은 복잡한 네트워크가 아니라 cue anchor를 순회하는 능력에서 발생
- **Episodic (Segment) > Extracted > Factual only**: raw segment가 가장 풍부한 context
- **Token 효율**: Policy retriever(1,853 tokens)가 Semantic(8,499 tokens) 대비 78% 적은 토큰으로 더 높은 성능

### Table 5: Latency (LoCoMo, 초)

| Retriever | Configuration | E2E Mean | E2E P95 | Search Mean | Search P95 | Avg Steps |
|---|---|---|---|---|---|---|
| Policy | Episodic (S) + Factual | 5.697 | 10.974 | 1.062 | 1.487 | 3.45 |
| Policy | Episodic (E) + Factual | 5.438 | 10.593 | 0.958 | 1.336 | 3.39 |
| Policy | Factual only | 4.653 | 9.388 | 0.733 | 1.006 | 3.36 |
| Semantic | Episodic (S) + Factual | — | — | 0.235 | 0.256 | 1 |
| Semantic | Episodic (E) + Factual | — | — | 0.221 | 0.260 | 1 |
| Semantic | Factual only | — | — | 0.200 | 0.245 | 1 |

→ Policy retriever는 평균 3.36-3.45 steps, step마다 LLM 호출 → search latency 4-5배. trade-off 존재.

### Table 6: Memory construction time per conversation (LoCoMo)

| System | Time (s) | Performance |
|---|---|---|
| Mem0 | 1,350.9 | 0.653 |
| Memora | 1,322.0 | 0.863 |
| Memora (offset) | 739.9 | 0.860 |

→ Memora construction time이 Mem0와 비슷하면서 성능은 크게 상향. offset 최적화 시 construction 45% 단축.

### Table 7: Smaller memory-construction model (LoCoMo)

| Construction Model | Retriever | Multi | Temp | Open | Single | Overall |
|---|---|---|---|---|---|---|
| gpt-5.4-nano | Semantic | 0.713 | 0.620 | 0.479 | 0.867 | 0.763 |
| gpt-4.1-mini | Semantic | 0.784 | 0.851 | 0.594 | 0.900 | 0.849 |
| gpt-5.4-nano | Policy | 0.773 | 0.879 | 0.625 | 0.893 | 0.851 |
| gpt-4.1-mini | Policy | 0.787 | 0.866 | 0.594 | 0.918 | **0.863** |

→ 더 작은 construction model(gpt-5.4-nano) + Policy retriever(0.851)가 더 큰 model(gpt-4.1-mini) + Semantic(0.849)과 맞먹음 → policy retriever가 construction 품질을 보완.

### GRPO 결과 (Figure 2, Qwen2.5-1.5B, LoCoMo test split)

| Configuration | Multi-hop | Temporal | Open-domain | Single-hop | Overall |
|---|---|---|---|---|---|
| Qwen 2.5 1.5B (Base) | — | — | — | — | 0.686 |
| Qwen 2.5 1.5B (GRPO) | 0.698 | 0.857 | 0.517 | 0.912 | **0.816** |

→ GRPO 훈련으로 base 대비 +0.130. GPT-4.1-mini Policy(0.863)에는 못 미치지만, 로컬 소형 모델로 0.816 달성 → 비용·지연 절감 with 경쟁력.

### Memory store 통계

- Memora: 평균 344 memory entries per conversation (Mem0: 651) → abstraction layer가 entry 수를 절반으로 줄이면서 각 entry는 더 풍부
- Token 소모: full-context 대비 최대 98% 절감
