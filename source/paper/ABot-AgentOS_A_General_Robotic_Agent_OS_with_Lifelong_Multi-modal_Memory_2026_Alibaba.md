# ABot-AgentOS: A General Robotic Agent OS with Lifelong Multi-modal Memory — 핵심 발췌

> 출처: [분석 문서](../../report/[paper]_ABot-AgentOS_A_General_Robotic_Agent_OS_with_Lifelong_Multi-modal_Memory_2026_Alibaba.md) / 원본: [arXiv:2607.10350](https://arxiv.org/abs/2607.10350)

## Abstract

> Recent VLM and VLA systems have improved robotic perception and action prediction, yet long-horizon embodied agents still require a general runtime layer for reasoning, memory, tool use, verification, and cross-embodiment execution. We present ABot-AgentOS, a general robotic Agent Operating System that sits above low-level controllers and provides a deliberative agent layer for scene-conditioned planning, context-isolated skill execution, multi-stage verification, multi-modal memory, and edge-cloud collaboration. To evaluate such systems, we introduce EmbodiedWorldBench, an executable benchmark with 16 indoor, outdoor, and hybrid scenes, four difficulty levels, and over 200 tasks involving navigation, object search, NPC dialogue, dynamic events, and trace-grounded scoring.
>
> ABot-AgentOS further introduces Universal Multi-modal Graph Memory, a persistent source-grounded substrate that converts dialogue, visual observations, spatial context, temporal relations, and task traces into typed nodes and edges. A failure-driven self-evolution loop converts diagnosed memory failures into gated runtime evo-assets that are promoted only to later evaluation splits, preventing current-split ground-truth leakage while enabling continual improvement. On an initial EmbodiedWorldBench subset, ABot-AgentOS improves over a single-controller baseline in both task success and goal completion. Across memory benchmarks, ABot-AgentOS Static achieves 87.5 on LoCoMo, 59.9 on OpenEQA EM-EQA, 88.6 on Mem-Gallery, and 76.5 Acc@All on NExT-QA; self-evolution further improves LoCoMo to 88.7, OpenEQA to 60.4, and Mem-Gallery to 89.0.

## Agent Framework (§2.1–2.2)

> 원문 §2.1 Architecture Overview / §2.2 Agent Harness / §2.2.1–2.2.4 발췌.

### §2.1 Architecture Overview

- ABot-AgentOS는 로봇 하드웨어·저수준 컨트롤러 위에 올라가는 **deliberative agent layer** (기존 control stack을 대체하지 않고 통합 cognitive layer 제공). humanoid·quadruped·mobile manipulator·robotic arm 등 이형 embodiment에 재사용.
- runtime = **edge Tiny LLM (매 턴 저지연: perception, instruction 이해, state tracking, routine decision) + cloud Large LLM (복잡 추론/장기계획/모호성 해소 시 on-demand)** 구조.
- Agent Harness layer = Verification-aware ReAct + Context Management + Skill Evolvement.
- Skills/Tools layer: manipulation, navigation, motion control, vision 등 — 같은 agent logic으로 다양한 embodiment 동작.
- Memory: edge private (map, semantic, multi-modal interaction history, user preferences, local env experience) + cloud common (reusable map 표현, 일반 task 경험, cross-robot 전송).

### §2.2 Agent Harness — 설계 동기

> 원문: "Unlike code agents and ordinary tool-use agents, such as SWE-agent ... Toolformer ... and ReAct ..., embodied agents often lack explicit completion signals for intermediate steps. An agent may invoke a navigation command without actually leaving its current location, or rotate and collide repeatedly while still believing, at the language level, that the task is progressing. Our framework therefore does not simply make the LLM call more tools. Instead, it organizes embodied execution as a closed loop of reasoning, execution, and verification."

- 단일 controller가 아니라 **main LLM / Skill Runner / Verifier** 3역할 분리.
- main LLM → tool 직접 호출 또는 Skill Runner 위임 → Verifier가 runtime·finish 양쪽 감시 → 반려 시 missing condition을 main LLM로 회귀.
- 목표: procedural drift, premature termination을 줄이고 LLM reasoning을 environment fact에 grounding.

### §2.2.1 Scene-Conditioned Task Planning (Main LLM)

> 원문 핵심: "This planning step is scene-conditioned rather than purely linguistic. The same instruction may require different execution strategies depending on the agent's current location, available visual evidence, known map structure, reachable regions, nearby objects, and interaction history."

- main LLM = semantic planner (저수준 movement를 일일이 issue하지 않음).
- 관측: memory, robot state, recent history, available skills.
- 행동 전 분류: navigation / search / human interaction / reporting / manipulation / additional observation 중 무엇이 필요한지.
- 수립: **explicit completion condition을 포함한 revisable high-level plan**.
- reasoning 대상: 이미 충분한 정보인지 / 관측·쿼리 필요한지 / 현재 상태에서 feasible한 skill / 어떤 observable condition이 completion인지.
- plan update 트리거: new observation, tool result, skill summary, verifier feedback이 context에 들어올 때.
- 결정: 추구할 goal / 직접 tool 충분 여부 / Skill Runner 위임 여부 / 종료 가능 여부.
- local 실행 디테일은 main reasoning thread에서 배제 → global task state coherent 유지, lower-level은 skill에 위임.

### §2.2.2 Skill Runner for Procedural Execution

> 원문: "A skill is not treated as a one-shot tool call or a simple action macro; it is executed by a skill-level subagent with an isolated local context that includes the subgoal, recent observations, skill state, failed attempts, and recovery strategy."

- main LLM은 full intermediate action sequence가 아니라 **compressed skill progress/outcome** 수신.
- context isolation이유: 반복 movement·observation·relocalization·view 조정·recovery가 매 턴 main context에 append되면 global objective가 procedural detail에 가려짐.
- procedural continuity 유지: 진행 여부 / 추가 관측·국소 복구 필요 시점 / main LLM으로 control 반환 시점을 self-check.
- 종료 시 반환하는 **compact semantic summary** 항목: subgoal 달성 여부 / 실패 원인 / 발견한 scene 정보 / 시도한 복구 / main LLM 활용 가이드.

### §2.2.3 Multi-Stage Verification

> 원문: "Unlike many digital tasks, where success can be checked through explicit external signals such as tests, API responses, or page states, embodied tasks often require consistency between the agent's declared progress, the execution trajectory, and the observed scene state."

- 세 단계:
  1. **Runtime verification**: recent trajectory + skill state 모니터 — effective progress인지, 아니면 stagnation / repeated collisions / local loops / 현재 plan과 불일치 행동인지.
  2. **Skill-level verification**: 위임 subtask가 semantic objective를 실제로 충족했는지 — tool/subagent가 정상 반환했다고 success로 accept하지 않음.
  3. **Finish-time verification**: 종료 시도 시 원 instruction · 현재 plan · execution history · skill summaries · observations · interaction 중 새 요구사항 종합 평가 후 종료 허용.
- Verifier는 final evaluator가 아니라 **실행 before/during/after에 작용하는 supervisory signal** → missing condition을 reasoning layer로 반환 → stagnation·misjudgment·premature termination 감소.

### §2.2.4 Edge-Cloud Collaborative Routing

- on-device model이 매 턴 task·context를 먼저 observe, local 처리 or cloud escalate.
- routing policy = **학습된 policy** (training sample + execution feedback 기반, 고정 규칙 아님).
- 결정 대상: local tool로 해결 가능한지 / cloud planning 필요한지 / routing 전 추가 관측 필요한지.
- 지연·비용 통제하면서 cloud-scale capability를 필요 시점에만 사용.

## Graph Memory Representation (§2.3.1)

장기 경험을 typed graph로 표현:

```
G = (V, E)   ... (1)
```

- 노드 v ∈ V: entity 또는 evidence unit (source container, evidence unit, entity, place, session, semantic event 등)
- 엣지 e ∈ E: temporal / semantic / spatial / identity / interaction / provenance 관계
- 각 노드는 compact JSON 필드 보유: schema version, dataset/source reference, time reference, evidence summary, confidence, adapter-specific fields, provenance (source id, adapter version, extractor model 등)
- 텍스트 chunk 기반 RAG와의 차이: 텍스트 검색은 의미적으로 유사한 snippet만 반환하지만, 그래프 메모리는 seed node를 찾은 뒤 local evidence subgraph로 확장하여 identity·time·location·participation·provenance·spatial relation까지 함께 추론.

## Hybrid Seed Retrieval 공식 (§2.3.3)

쿼리 q에 대한 후보 seed node 랭킹:

```
s(q, v) = λ_sem * s_sem(q, v) + λ_lex * s_lex(q, v) + λ_meta * s_meta(q, v) + λ_type * s_type(q, v)   ... (2)
```

- `s_sem`: 임베딩 유사도
- `s_lex`: 어휘 중첩(lexical overlap)
- `s_meta`: 시간/소스/모달리티/장소 등 메타데이터 호환성
- `s_type`: 쿼리가 기대하는 노드 타입 선호도

top-ranked seed는 고정 depth·evidence-token budget 하에 typed edge를 따라 확장되어 local evidence subgraph를 생성 → 압축된 evidence context로 직렬화되어 answerer에 주입, retrieval trace도 함께 기록.

## Failure-Driven Lifelong Self-Evolution (§2.3.4)

split-wise protocol: 순서가 있는 disjoint split D1, ..., DT.

```
T_t, G_t = Run(D_t; G_{t-1}, A_{<t})              ... (3)
ΔA_t   = Gate(Compile(Propose(Diagnose(T_t))))     ... (4)
A_{≤t} = A_{<t} ∪ ΔA_t                             ... (5)
```

- `G_t`: 누적 memory graph, `A_{<t}`: 이전 split까지 promote된 evo-asset 집합
- split t에서 생성된 evo-asset은 split t 평가에 절대 사용되지 않음 (no-leakage) → 이후 split에만 적용
- Diagnose → Propose → Compile → Gate 파이프라인으로 실패 원인을 진단하고 후보 repair를 JSON DSL evo-asset으로 컴파일

Gate 승인 조건 (target 성능 향상 + regression 억제):

```
Accept(a) = I[ ΔS_target(a) ≥ τ_gain  ∧  ΔS_reg(a) ≥ -τ_reg ]   ... (6)
```

- evo-asset은 Python 코드를 직접 실행하지 않는 lifecycle-managed JSON DSL record (target layer, triggering condition, permitted action, safety constraint, provenance, validation result, version id 포함)
- writer-side/frame-policy asset은 새 그래프 구축이 필요, evidence-selection/answerer asset은 runtime에 로드 가능

## Edge-Cloud Collaborative Memory (§2.3.5)

- Private memory(edge): map, semantic memory, multi-modal memory 등 로봇별 로컬 경험 전부 저장
- Common memory(cloud): public/low-sensitivity 정보(지도, 장애물, 표지 등)만 저장 — "private-by-default"
- Privacy-aware gating: 개인식별정보(얼굴, 이름, 개인 소지품, 소유관계)를 포함하면 non-shareable로 분류. 자체 구축한 privacy classification 데이터셋에서 99% 이상 정확도로 업로드 여부 판정.

## Self-Evolving Reward Engine (§4.5)

Episode-level reward 분해:

```
r_episode(τ) = λ_eff * R_eff(τ) + λ_cons * R_cons(τ) + λ_comp * R_comp(τ)   ... (8)
```

Merged return (턴 단위 reward + episode reward):

```
R(τ) = Σ_{t=1}^{T} r̂_t + r_episode(τ)   ... (9)
```

Meta-Judge quality score (5개 검증 차원: accuracy, logical soundness, completeness, clarity, feedback value):

```
Q(x_t) = Σ_{k=1}^{5} w_k * q_k          ... (10)
LowQualityJudgeCase(x_t) = I[Q(x_t) < θ] ... (11)
```

- Multi-Agent Self-Evolution: Cluster(스킬·실패유형별 그룹화) → Analyzer(결함 rubric/example 식별) → Refiner(국소 수정) → Validator(전체 validation set 재평가, 악화 시 rollback)
- 초기 Judge Model: 인간 정합도 약 60% → Meta-Judge 기반 self-evolution 후 90% 이상으로 향상

## 주요 결과 표 (요약)

### Table 1 — EmbodiedWorldBench subset (Agent Evaluation)

| Agent | Model | TSR | GCR |
|---|---|---|---|
| ReAct | Qwen3.6-Plus | 49.97% | 57.95% |
| ABot-AgentOS | Qwen3.6-Plus | 61.96% | 68.79% |
| ABot-AgentOS | DeepSeek-V4-Pro | 68.18% | 74.62% |

### Table 2 — LoCoMo (Overall)

| Method | Overall |
|---|---|
| Mem0 | 85.6 |
| MemGPT | 80.3 |
| **ABot-AgentOS Static** | **87.5** |
| Human | 87.9 |

### Table 3 — OpenEQA EM-EQA (Overall)

| Method | Frames | Overall |
|---|---|---|
| GaussExplorer | n/a | 57.8 |
| **ABot-AgentOS Static** | 24 | **59.9** |
| Human | – | 86.8 |

### Table 4 — Mem-Gallery (Overall)

| Method | Overall |
|---|---|
| MemGPT (captioned) | 87.6 |
| UniversalRAG | 84.7 |
| **ABot-AgentOS Static** | **88.6** |

### Table 5 — NExT-QA (Acc@All)

| Method | Acc@All |
|---|---|
| GraphVideoAgent | 73.3 |
| **ABot-AgentOS Static** | **76.5** |

### Table 6 — EgoLifeQA (Avg. Accuracy)

| Method | Frames | Avg |
|---|---|---|
| EGAgent-Gemini2.5 Pro | 1FPS→50 | 57.5 |
| WorldMM-Qwen3.5 Flash | Full | 56.0 |
| **ABot-AgentOS-Qwen3.5 Flash** | 1FPS→1 | **65.4** |

### Self-Evolution 개선 (Static → +Self-evo)

| Benchmark | Static | +Self-evo | Δ |
|---|---|---|---|
| LoCoMo | 87.5 | 88.7 | +1.2 |
| OpenEQA | 59.9 | 60.4 | +0.5(~1.2 primary) |
| Mem-Gallery | 88.6 | 89.0 | +0.4 |
| NExT-QA | (Acc@All) | — | +4.1 (최대) |
| EgoLife | 65.4 | 66.2 | +0.8 |

## Appendix B — Lifelong Self-Evolution Process Example (OpenEQA split_00)

> 원문 발췌 (Diagnosis):
> "upstream scene evidence contained an object observation, such as a throw blanket on a bed; the graph did not expose a corresponding typed object node or directed object-room/object-object relation; retrieval returned unrelated object records instead of the target object evidence; the answerer therefore produced an unsupported or missing answer. This diagnosis points to a memory-system failure, not a direct answer-format patch."

**Table 7 — Candidate assets proposed for the representative OpenEQA self-evolution split**

| Candidate | Layer | Proposed evolution |
|---|---|---|
| Writer materialization | Writer | Promote adapter keyframe object mentions into typed object observations with aliases, attributes, observed state facts, room/place linkage, last-seen evidence, source reference, frame evidence, confidence, and provenance. |
| Retriever room-anchor/last-seen focus | Retriever | Prefer observed object memories using room/place anchors, last-seen evidence, object identity, and directed spatial grounding rather than broad scene-level lexical similarity. |
| Writer directed support relations | Writer | Write directed spatial support relations only when both endpoints and relation direction are explicitly grounded by observation evidence. |

**Table 8 — Candidate-level gate outcomes for split_00**

| Candidate | Gate result | Main reason |
|---|---|---|
| Writer materialization | Rejected | Although target and global score improved, protected object-state recognition regressed slightly beyond the configured tolerance. |
| Retriever room-anchor/last-seen focus | Accepted | Target delta was +0.800, global delta was +0.044, and low-score count decreased by 10. |
| Writer directed support relations | Rejected | Target delta was negative and protected object-state recognition regressed. |

> Stack confirmation: normalized baseline mean 0.6053 → normalized stack mean 0.6545 (stack delta +0.0492); low-score count 91→77; protected object-state recognition did not regress; protected object-localization improved by approximately +0.073.

> 승인된 최종 runtime policy 원문:
> "When an OpenEQA retrieval query contains an object anchor, room/place cue, or spatial phrase, rank observation-grounded object memories ahead of broad scene summaries. Prefer records with matching room/place, explicit last_seen, source_ref/frame evidence, object identity, and directed spatial relation evidence. Do not infer room priors, reverse relation direction, or use ungrounded scene summaries as object evidence."
> "This is a generic retrieval policy. It does not contain gold answers, manual answer rules, or question-specific shortcuts."

**Table 9 — Eight-split trace of the OpenEQA lifelong self-evolution run** (원문 전체). Candidate-level acceptance만으로는 충분하지 않으며, 신규 asset은 full stack confirmation을 통과해야 최종 승격됨.

| Split | Main proposed evolution | Candidate gate outcome | Stack outcome |
|---|---|---|---|
| `split_00` | Writer materialization of observed adapter objects; retriever room-anchor/last-seen focus; writer directed support relations | Retriever room-anchor/last-seen policy accepted; two writer policies rejected | **Accepted and carried forward** |
| `split_01` | Retriever anchor-transition trace ranking; writer grounded room-transition facts | Both rejected due target/global regression and protected-category risk | No new asset promoted |
| `split_02` | Retriever last-seen/state conflict tracing; retriever directed spatial relation guard; writer observation/state alias completion | All rejected due insufficient target gain or global/protected regression | No new asset promoted |
| `split_03` | Retriever grounded spatial/recency/object ranking; writer last-seen completeness; writer identity/alias preservation | Retriever candidate passed candidate gate; writer candidates rejected | **Stack failed; newly accepted retriever asset deprecated** |
| `split_04` | Retriever recency/directional evidence reranking; writer object-observation provenance completion | Both rejected due target/global regression | No new asset promoted |
| `split_05` | Writer keyframe object promotion; retriever last-seen/directed spatial ranking | Writer candidate accepted by global-rescue gate; retriever rejected | **Stack failed; newly accepted writer asset deprecated** |
| `split_06` | Retriever anchor-directed same-room binding; writer typed object observation exposure | Both rejected due insufficient target gain or global regression | No new asset promoted |
| `split_07` | Retriever directed spatial relation first; writer object/state/last-seen consolidation | Retriever candidate passed candidate gate; writer rejected | **Stack failed; newly accepted retriever asset deprecated** |

> Takeaway(원문): "The successful evolution was not an answer patch. It changed the retriever policy so that evidence already present in graph memory is ranked and exposed more reliably. Most proposed writer policies were rejected because they risked object-state or localization regressions, and candidate-level wins were not enough unless the full evolved stack also passed confirmation."

## §5.2.3 / §5.2.4 — Self-Evolution 결과 및 벤치마크별 실패 유형

> §5.2.4 line 1326-1328 원문: "The most consistent gains appear in categories that expose systematic pipeline errors: **temporal normalization** in LoCoMo, **embodied scene disambiguation** in OpenEQA, **relation and conflict handling** in Mem-Gallery, and **causal-temporal reasoning** in NExT-QA."

| Benchmark | self-evolution이 개선한 주요 실패 유형 | Static → +Self-evo |
|---|---|---|
| LoCoMo | temporal normalization (상대 날짜 해석 오류) | 87.5 → 88.7 (+1.2) |
| OpenEQA | embodied scene disambiguation (object-room materialization) | 59.9 → 60.4 (primary +1.2) |
| Mem-Gallery | relation/conflict handling (multi-entity, conflict detection/refusal) | 88.6 → 89.0 (+0.4, 카테고리별 집중) |
| NExT-QA | causal-temporal reasoning (인과/시간 비디오 QA) | Acc@All 최대 +4.1 |
| EgoLife | TaskMaster 카테고리 (egocentric 장기 일상) | 65.4 → 66.2 (+0.8) |

> §5.2.3 line 1273-1280 (no-leakage 강조): "ground-truth signal is used only after a split has already been evaluated. It is not available during inference, is not written into the memory graph, and does not generate assets for the same split. This is analogous to how a deployed embodied agent would use interaction feedback in the real world."

## Appendix B — procedure (line 1900-1946 원문 요약)

For each split:
1. Incumbent evaluation — 현 memory 시스템으로 split 실행
2. Diagnosis — Diagnoser가 failed QA rows·retrieval traces·graph evidence 검사 → memory root cause별 클러스터링 (답변 포맷 오류로 분류하지 않음)
3. Hypothesis generation — HypothesisGenerator가 writer/retriever/answerer/frame-policy layer의 generic JSON DSL 후보 제안
4. Compilation & safety review — CompilerCritic이 executable code·schema migration·direct answer resolver·fixed answer table·qid/gold-answer rule reject
5. Gate analysis — GateAnalyst이 protected-category constraint·activation guard·cost/lifecycle constraint·regression risk 추가
6. Candidate evaluation — target 개선 + global regression 억제 + low-score-count 증가 금지 + protected-category regression 금지
7. Stack confirmation — 통과 asset 전체를 stack으로 재평가, incumbent를 이겨야 최종 승격, 아니면 해당 round 신규 asset 전부 deprecated

## Figure 4 — Concrete memory failure-to-evolution examples (caption)

> "Left: visual memory QA retrieves image-grounded identity evidence but can expose missing breed-specific cues. Right: temporal text memory QA uses session metadata to resolve relative dates but can reveal temporal-normalization errors. In both cases, the failure trace is converted into targeted memory-writing, evidence-selection, frame-selection, or answering improvements."

## §2.3.2 저장 예시 원문 (dog-adoption utterance)

> "For instance, the utterance 'I adopted a Maltese dog yesterday' is represented as a time-grounded semantic event that links the utterance, resolved temporal context, identity hypothesis, confidence estimate, and source evidence, rather than as a raw transcript line. When an image is attached, the image node is connected to the event as supporting multi-modal evidence."
