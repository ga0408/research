# Remember When It Matters: Proactive Memory Agent for Long-Horizon Agents — 핵심 발췌

> 출처: [분석 문서](../../report/[paper]_Remember_When_It_Matters_Proactive_Memory_Agent_for_Long-Horizon_Agents_2026_Meta_AI.md) / 원본: [arXiv:2607.08716](https://arxiv.org/abs/2607.08716)

## Behavioral State Decay (핵심 개념)

**정의 (논문 §1):**

> "During long-horizon execution, information that should shape future actions like task requirements, environment facts, previous attempts, failure diagnoses, intermediate discoveries, and open subgoals stops influencing the agent's next decision. The information may still be present in the transcript, or may even remain within the model's context window, but it no longer exerts reliable control over behavior."

**구체적 실패 양상 (논문 §1):**

| 실패 양상 | 예시 |
| --- | --- |
| 요구사항 위반 | task 초반에 requirement를 식별했으나, 무관한 bug를 수정하다가 나중에 그 requirement를 위반 |
| 동일 실패 반복 | 특정 command·parameter·구현 경로가 실패했음을 관찰했으나, 나중에 거의 동일한 변형을 재시도 |
| 진단 망각 | error pattern을 진단했으나, 나중에 같은 pattern을 새로운 것으로 취급 |

**핵심 구분:** 정보가 transcript나 context window에 여전히 존재하더라도 행동에 대한 제어력을 잃는 현상. 많은 memory 접근법이 "저장·검색"에 집중하지만, long-horizon task 실행에서는 "기억된 정보가 언제 next action에 영향을 주어야 하는가"를 결정하는 것도 필요. Memory는 write/retrieval 문제가 아니라 **intervention 문제**.

> "Effective memory for long-horizon agents is not only a write and retrieval problem, but also an intervention problem."

## Problem Setup (§3.1)

Action agent가 환경과 trajectory τ = (o₁, a₁, o₂, a₂, ..., o_T) 상호작용.

- action: aₜ ~ π_A(aₜ | x, τ<ₜ), π_A는 LLM + tool-use scaffold
- long-horizon에서 scaffold가 π_A에 노출하는 context는 truncation/summarization/filtering 될 수 있음

Memory agent π_M 추가:
- 첫 step 및 고정 interval마다 호출
- 각 memory timestep t에서 관찰: task description x, recent trajectory window wₜ = Wₖ(τ<ₜ, oₜ), current memory bank Bₜ₋₁
- 두 단계 결정:

```
Bₜ ~ π_M^edit(· | x, wₜ, Bₜ₋₁)         # Phase 1: memory bank update
iₜ ~ π_M^intervene(· | x, wₜ, Bₜ)       # Phase 2: intervention decision
```

- iₜ ∈ {∅, text reminder}
- non-null intervention → next action-agent call에 transient memory context로 주입
- null → action agent context 변경 없음

> "This formulation treats memory as a policy over interventions."

## Memory Bank 구조 (§3.2)

```
Bₜ = (sₜ, Kₜ, Pₜ)
```

| 컴포넌트 | 의미 | t시점 범위 | action agent 노출 |
| --- | --- | --- | --- |
| sₜ (status) | memory agent의 private 진행/이슈/리스크 추적 | 현재 t시점 상태 | X (공개 안 됨) |
| Kₜ (knowledge) | 안정적 사실: task requirements, environment properties, file paths, config, verified facts | t까지 축적된 모든 knowledge entry | 간접 (reminder로만) |
| Pₜ (procedural) | 시도·결과: failed commands, successful fixes, ruled-out hypotheses, diagnostic signals | t까지 축적된 모든 procedural entry | 간접 (reminder로만) |

**시점별 저장 방식:**
- **sₜ (status):** t시점의 현재 진행 상황·열린 이슈·미해결 리스크를 나타내는 단일 상태 필드. memory agent가 매 memory step마다 `memory_update_status`로 덮어쓰기. 축적이 아닌 현재 상태 스냅샷.
- **Kₜ (knowledge):** task 진행 중 발견한 안정적 사실들을 entry 단위로 축적. 각 entry는 `memory_save_knowledge`로 추가되며, 식별자로 명시적 update/delete 가능. t시점까지 축적된 전체 집합.
- **Pₜ (procedural):** 시도한 명령·해결책·가설 등을 entry 단위로 축적. 각 entry는 `memory_save_procedural`로 추가. t시점까지 축적된 전체 집합.

**Entry 구조:**
- short identifier + natural-language content + metadata (creation time, access stats)
- identifier로 stale entry update/delete 가능
- compact tagged format (environment facts, paths, task facts, bugs, performance observations 등)

### 검색 메커니즘 — 없음 (LLM 판단이 검색을 대체)

이 논문의 memory bank는 **별도 검색 메커니즘(vector search, BM25, 키워드 검색 등)을 제공하지 않음**. 핵심 설계:

1. **Bank를 "compact"하게 유지** — memory agent가 매 memory step마다 **전체 bank** Bₜ를 읽음
2. **Phase 2에서 LLM이 전체 bank + recent trajectory를 보고 판단** — 어떤 entry가 다음 action에 관련 있는지를 LLM 자체 판단으로 결정. retrieval step 없음.
3. **Ablation에서 명시적 대조** — Mem0 대조군(vector+BM25 top-10 retrieval)과 구분됨. 본 방법은 retrieval 대신 "selective intervention"으로 relevant 정보 선별.

> "Unlike embedding-based retrieval-augmented prompting, which retrieves fixed records from a store, this variant still uses the memory model to synthesize a reminder from the maintained bank." (§4.3, Always inject ablation 설명)

즉 **retrieval이 아니라 LLM 기반 selection + synthesis**. memory agent는 전체 bank를 읽고, 현재 trajectory 상황에서 어떤 기억이 행동에 영향을 줘야 하는지 직접 판단한 뒤, 필요시 새로운 reminder 텍스트를 합성(compose)하여 주입. 기존 entry를 그대로 가져오는 것이 아니라 상황에 맞게 재구성함.

## Two-Phase Memory Agent (§3.3)

### Phase 1: Memory management

- memory agent가 predefined tool call 리스트 반환 (직접 bank rewrite 아님)
- system이 순서대로 실행 → updated bank
- tool call 없으면 bank 변경 없음

| Tool call | 용도 |
| --- | --- |
| `memory_update_status` | private progress tracking (action agent에 노출 X) |
| `memory_save_knowledge` | task requirements, environment facts, file paths, API details, key constraints |
| `memory_save_procedural` | debugging experience, failed approaches, solutions, error patterns, successful fixes |
| `memory_delete` | outdated/incorrect entry 제거 (by identifier) |

> "The Phase 1 output is a sequence of bank edits, not a free-form summary of the trajectory."

### Phase 2: Intervention selection and transient injection

- updated bank + recent trajectory 조건으로 intervention action 선택
- bank 수정 안 함
- 결과: reminder rₜ 또는 null intervention ∅

**reminder 주입 방식:**
- rₜ를 next action-agent call에 separate transient memory context로 제공
- action agent의 base instructions, tools, decoding procedure 변경 없음
- 유일한 변화: call time에 optional memory context

**null intervention은 explicit action:**
- 유용한 intervention: about-to-be-violated requirements, 현재 observation을 설명하는 environment facts, 반복하지 않아야 할 previous attempts, 관련 diagnoses, neglect된 open subgoals
- 비권장: broad strategic advice, 이미 visible한 정보 재진술, action agent planning 대체

> "Intervention timing is part of the memory policy rather than a consequence of every memory update."

## Triggering (§3.4)

- main implementation: 첫 step + 고정 interval
- 더 선택적 trigger 가능 (tool errors, failed tests, repeated commands, large context shifts 후)
- 고정 interval 사용 → memory intervention policy 자체의 효과 분리 목적

## Learning Memory Intervention Policies (§3.5)

- main instantiation: prompted model (training 불필요)
- training은 "같은 intervention policy를 학습 가능한가?" 탐색
- action agent 고정, memory agent만 학습

**SFT (Supervised Fine-Tuning):**
- prompted memory agent의 trajectory distill
- Phase 1 bank operations + Phase 2 intervention decisions 학습
- compact writing, stale state update, 불필요한 reminder 회피 교육

**GRPO (Reinforcement Learning):**
- imitation만으로는 downstream effect 최적화 불가 → RL로 calibration
- 목표: memory 사용량 최대화가 아니라 intervention policy 개선
- sparse task-level verifier rewards → pivot turns에 update 집중 (offline rollout에서 식별)

## Main Results (Table 1)

Memory agent: Claude Opus 4.6. Scores: pass@1 %.

| Benchmark | Domain / Split | Action Model | n | Baseline | + Memory | Δ |
| --- | --- | --- | --- | --- | --- | --- |
| Terminal-Bench 2.0 | full set | Sonnet 4.5 | 85 | 37.6% | 45.9% | +8.3 pp |
| Terminal-Bench 2.0 | full set | Opus 4.6 | 85 | 43.5% | 45.9% | +2.4 pp |
| τ²-Bench | airline | Sonnet 4.5 | 50 | 68.0% | 78.0% | +10.0 pp |
| τ²-Bench | retail | Sonnet 4.5 | 114 | 49.1% | 58.8% | +9.6 pp |
| τ²-Bench | telecom | Sonnet 4.5 | 114 | 55.3% | 57.9% | +2.6 pp |
| τ²-Bench | task-weighted avg. | Sonnet 4.5 | 278 | 55.0% | 61.8% | +6.8 pp |
| τ²-Bench | airline | Opus 4.6 | 50 | 76.0% | 76.0% | +0.0 pp |
| τ²-Bench | retail | Opus 4.6 | 114 | 64.9% | 69.3% | +4.4 pp |
| τ²-Bench | telecom | Opus 4.6 | 114 | 63.2% | 64.9% | +1.8 pp |
| τ²-Bench | task-weighted avg. | Opus 4.6 | 278 | 66.2% | 68.7% | +2.5 pp |

## Ablations (Table 2)

τ²-Bench, Sonnet 4.5 action agent, Opus 4.6 memory agent.

| Variant | Phase 1 | Phase 2 | Airline | Retail | Telecom | Macro | Micro |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Sonnet 4.5 baseline | — | — | 68.0 | 49.1 | 55.3 | 57.5 | 55.0 |
| Full memory agent (ours) | bank mgmt | selective reminder/silence | 78.0↑ | 57.0↑ | 57.9↑ | 64.3↑ | 61.2↑ |
| Full-bank context | bank mgmt | expose full bank | 74.0↑ | 52.6↑ | 57.9↑ | 61.5↑ | 58.6↑ |
| Always inject | bank mgmt | forced reminder every step | 72.0↑ | 58.8↑ | 59.6↑ | 63.5↑ | 61.5↑ |
| Injection-only (no bank) | skipped | selective guidance/silence | 62.0↓ | 54.4↑ | 66.7↑ | 61.0↑ | 60.8↑ |
| Mem0 | Mem0 ADD | vector+BM25 top-10 | 68.0 | 59.6↑ | 58.8↑ | 62.1↑ | 60.8↑ |

**Ablation 핵심 발견:**
- Full memory agent가 macro-average 최고 (모든 domain 개션, airline에서 최대 gain)
- Full-bank context: baseline 대비는 개선 but full system 대비 -2.8 macro, -2.6 micro → "selectAll memory 노출만으로는 부족, selective intervention 필요"
- Always inject: micro에서 +0.3로 근소 우세 but macro에서 selective silence가 우위 (airline에서 특히)
- Injection-only: persistent memory 없이는 불안정, airline에서 baseline 대비 하락
- Mem0: 평균 개선 but airline에서 baseline 동일 (0 gain), macro에서 full system 열세

## Qualitative Mechanisms (Table 3)

| Mechanism | Typical memory content | Representative examples |
| --- | --- | --- |
| Requirement / policy reactivation | Task or domain rule about an allowed action | τ² airline compensation, retail modification rules |
| Environment grounding | Runtime facts, paths, tool limitations, system quirks | Terminal-Bench Git server setup, ARS file-write failure |
| Failure-loop avoidance | Previous attempts and why they failed | Adaptive rejection sampling, telecom diagnostic retries |
| Diagnostic carryover | Root cause of a bug or negative signal | Regex edge cases, SQLite gcov configuration |
| Progress / entity tracking | Which user, order, line, branch, or subgoal is active | τ² telecom line lookup, retail authentication state |

## 구체적 동작 예시 (§4.4 Qualitative Analysis)

### Terminal-Bench — debugging continuity 유지

**regex-log task:**
- memory agent가 **requirements + diagnoses 동시에 reactivation**
- 현재 regex가 task의 boundary condition을 위반하고 있고, single-digit IPv4 octet을 놓치고 있음을 지적
- → action agent가 이를 반영하여 regex 수정

**adaptive-rejection-sampler task:**
- memory agent가 **반복된 file edit 실패를 추적** (procedural memory에 축적)
- 나중에 environment-specific workaround를 surface하여 동일 실패 반복 방지
- → debugging iteration 간 continuity 보존: 시도한 것, 관찰한 것, 실패한 것, 해결을 위해 남은 제약

### τ²-Bench — policy/interaction state reactivation

**airline case 1 (사용자 주장 vs 검증된 기록):**
- user가 Gold status를 주장
- tool output은 Regular member를 표시
- **baseline:** user 주장을 믿고 compensation 실행 (실패)
- **memory-enabled:** action agent에게 "검증된 기록에 의존하라"는 reminder 주입 (성공)
- → Progress/entity tracking 메커니즘

**airline case 2 (basic-economy 수정 방지):**
- action agent가 basic-economy flight 수정을 시도
- memory agent가 "해당 flight는 수정 불가"라는 policy clause를 reactivate
- → invalid modification 방지, Requirement/policy reactivation 메커니즘

**공통 패턴 (τ²-Bench):**
- 성공적인 intervention은 **state-changing tool call 직전**에 발생
- reminder 내용: authentication requirements, eligibility conditions, one-shot tool limits, 시행되지 않는 policy clauses

### 성공·실패 패턴 요약

| 측면 | 설명 |
| --- | --- |
| 성공 intervention 특성 | specific, grounded, timely — execution state의 인과적 영향을 복원 |
| 실패 intervention 유형 | ① speculative inference를 과신 ② action agent가 이미 아는 정보 반복 ③ 불필요한 concern으로 extra verification 유발 |
| 실패 원인 | memory storage 실패가 아닌 **calibration error** → "언제 침묵할까" 결정이 핵심 |

## Training & Transfer (Table 4)

Action agent: frozen Qwen3.5-122B-A10B; memory agent: Qwen3.5-27B. Train on SETA, transfer to Terminal-Bench 2.0.

**(a) SETA validation:**

| Setup | Avg. reward | Solved | Δ |
| --- | --- | --- | --- |
| Action only, no memory | 0.709 | 56 | — |
| + Qwen3.5-27B base memory | 0.693 | 54 | -0.016 |
| + SFT memory | 0.720 | 58 | +0.011 |
| + GRPO memory | 0.734 | 58 | +0.025 |

**(b) Transfer to Terminal-Bench 2.0:**

| Setup | n | Pass@1 | Δ |
| --- | --- | --- | --- |
| Qwen3.5-122B-A10B action only | 85 | 37.6% | — |
| + trained Qwen3.5-27B memory | 85 | 41.1% | +3.5 pp |

- Untrained 27B memory agent는 성능 하락 (0.709 → 0.693)
- SFT로 회복 (0.720), GRPO로 추가 개선 (0.734)
- SETA 학습 → held-out Terminal-Bench에 부분 transfer (37.6% → 41.1%)
