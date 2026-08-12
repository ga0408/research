> [paper] https://arxiv.org/abs/2607.08716

# Remember When It Matters: Proactive Memory Agent for Long-Horizon Agents

## Summary & Outline

**한 줄 요약:** Long-horizon agent의 핵심 실패 모드인 behavioral state decay를 규명하고, action agent를 변경하지 않은 채 별도의 memory agent가 구조화된 memory bank를 관리하고 선택적으로 reminder를 주입하는 proactive intervention 아키텍처를 제안. Terminal-Bench 2.0과 τ²-Bench에서 일관된 성능 향상을 입증하고, open-weight 모델로 intervention policy를 학습 가능함을 초증명.

**논문 구조 outline:**
1. Introduction — behavioral state decay 개념 정의, memory as intervention 제안
2. Related Work — long-horizon agents, agent memory, learned memory policies, reflection/advisors
3. Method — two-phase memory agent 아키텍처 (memory bank + intervention)
4. Experiments and Results — Terminal-Bench 2.0 & τ²-Bench 평가, ablations, qualitative analysis, training
5. Conclusion

## Problem & Motivation

- **연구 배경:** LLM agent가 command-line 실행, multi-step tool use 등 long-horizon task에서 평가받기 시작. 이런 task는 수많은 observation·action·partial decision에 걸쳐 전개되며, 성공은 각 국소 문제 해결뿐 아니라 미래 행동을 제약해야 할 정보를 유지하는 데 달려 있음.
- **풀고자 하는 문제:** **Behavioral state decay** — long-horizon 실행 중 task requirements, environment facts, previous attempts, failure diagnoses, intermediate discoveries, open subgoals 등이 agent의 다음 결정에 더 이상 영향을 주지 못하는 실패 모드. 정보가 transcript나 context window에 여전히 존재하더라도 행동 제어력을 잃는 현상.
  
  논문이 제시하는 구체적 실패 양상:
  
  | 실패 양상 | 설명 |
  | --- | --- |
  | 요구사항 위반 | task 초반에 requirement를 식별했으나, 무관한 bug 수정 중 그 requirement를 위반 |
  | 동일 실패 반복 | 특정 command·parameter·구현 경로가 실패했음을 관찰했으나, 나중에 거의 동일한 변형을 재시도 |
  | 진단 망각 | error pattern을 진단했으나, 나중에 같은 pattern을 새로운 것으로 취급 |
  
  핵심 구분: 정보의 "존재"가 아니라 행동에 대한 "제어력"의 상실. 더 긴 history 제공만으로는 해결되지 않음.
- **기존 접근의 한계:**
  - 기존 memory 시스템(MemGPT, Mem0, MemoryBank 등)은 저장·업데이트·검색에 집중 → personalization, persistent user state, cross-session recall에는 적합하지만, **"언제 개입할 것인가"** 문제를 다루지 않음
  - Reflection/critic/advisor 접근(Reflexion, Self-Refine, advisor models)은 broad strategic guidance에 가까움 — memory-grounded reminder로 제약되지 않음
  - Summarization은 "무엇을 보존할까"만 묻고, "보존된 상태가 다음 action에 활성화되어야 하는가"는 묻지 않음
  - 더 긴 context window 제공만으로는 해결 불가 (Lost in the middle 현상)

## Contributions

1. **개념 기여:** Behavioral state decay를 long-horizon language agent의 핵심 실패 모드로 식별
2. **방법론 기여:** Memory maintenance(Phase 1)와 action selection(Phase 2)을 분리하는 two-phase memory intervention 아키텍처 제안. Action agent를 수정하지 않는 plug-and-play 설계
3. **실증 기여:** Terminal-Bench 2.0 및 τ²-Bench에서 memory intervention이 weaker/stronger action agent 모두에 일관된 성능 향상. Ablation으로 selective intervention > passive exposure / always-on injection / advisor-only / general retrieval 입증
4. **학습 가능성 증명:** Qwen3.5-27B를 SFT + GRPO로 학습하여 intervention policy가 open-weight 모델에 부분적으로 distill 가능함을 초증명

## Method

### 전체 아키텍처

```
┌──────────────────────────────────────────────────────────────────────┐
│                    Memory-Intervention Architecture                    │
├──────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌──────────────┐                              ┌───────────────────┐  │
│  │  Action Agent │── action aₜ ──▶ Environment  │   Memory Agent    │  │
│  │  (unchanged)  │◀── observation oₜ ──────────  │   (π_M)           │  │
│  │               │                              │                   │  │
│  │  π_A(aₜ|x,τₜ) │                              │  observes:        │  │
│  │               │   ┌── transient ────────────▶│  • task desc x    │  │
│  │               │◀── reminder rₜ ──────────────│  • window wₜ(k=8) │  │
│  └──────────────┘   └                          │  • bank Bₜ₋₁      │  │
│                                                 └───────┬───────────┘  │
│                                                         │              │
│                    ┌────────────────────────────────────┘              │
│                    ▼                                                    │
│         ┌─────────────────────┐   ┌──────────────────────┐            │
│         │  Phase 1: Memory     │──▶│  Phase 2: Intervention│            │
│         │  Bank Management     │   │  Selection            │            │
│         │                      │   │                      │            │
│         │  tool calls:         │   │  rₜ (reminder)       │            │
│         │  • update_status     │   │  or ∅ (silence)      │            │
│         │  • save_knowledge    │   └──────────────────────┘            │
│         │  • save_procedural   │                                       │
│         │  • delete            │   ┌──────────────────────┐            │
│         └─────────────────────┘   │  Memory Bank Bₜ       │            │
│              │                    │  = (sₜ, Kₜ, Pₜ)      │            │
│              └───────────────────▶│  s: status (private)  │            │
│                                   │  K: knowledge mem.    │            │
│                                   │  P: procedural mem.   │            │
│                                   └──────────────────────┘            │
│                                                                        │
│  Trigger g(t): first step + fixed interval (every N steps)             │
└──────────────────────────────────────────────────────────────────────┘
```

### 핵심 설계 원칙: Memory = Intervention Policy

이 논문의 핵심 통찰은 memory를 **저장·검색 문제가 아닌 intervention 문제**로 재구성한 것. Memory agent는 "무엇을 기억할까"뿐 아니라 **"기억된 실행 상태가 언제 action loop에 진입해야 하는가"**를 결정해야 함.

이는 summarization보다 더 강력한 제어 질문:
- Summarizer: "무엇을 보존할까?" → 유지
- 본 방법: "보존된 실행 상태가 다음 action agent 결정에 활성화되어야 하는가?" → 개입 여부 판단

Task마다 실패 모드가 다름 (hard requirement vs environment fact vs failed command vs bug diagnosis vs unfinished subgoal) → 고정 summarization 정책으로는 "기억이 다음 행동을 중단시켜야 하는가"를 알 수 없음.

### Memory Bank 구조

세 컴포넌트로 구성된 구조화 표현:

| 컴포넌트 | 역할 | t시점 범위 | Action Agent 노출 |
| --- | --- | --- | --- |
| **Status** (sₜ) | memory agent의 private 진행/이슈/리스크 추적. task의 working model 유지 | 현재 t시점 상태 스냅샷 (매 step 덮어쓰기) | 노출 안 됨 (context 오염 방지) |
| **Knowledge** (Kₜ) | 안정적 사실: task requirements, environment properties, file paths, config, user/tool-verified facts | t까지 축적된 전체 entry 집합 | reminder로만 간접 전달 |
| **Procedural** (Pₜ) | 시도·결과: failed commands, successful fixes, ruled-out hypotheses, diagnostic signals, empirical improvements | t까지 축적된 전체 entry 집합 | reminder로만 간접 전달 |

**시점별 저장 방식:**
- **Status**는 축적이 아닌 현재 상태 스냅샷 — memory agent가 매 step `memory_update_status`로 덮어쓰기. 진행 중인 이슈·미해결 리스크의 working model.
- **Knowledge·Procedural**는 entry 단위로 축적 — 각 entry는 `memory_save_knowledge`/`memory_save_procedural`로 추가되며, identifier로 명시적 update/delete 가능. t시점까지 발견·시도한 모든 정보를 구조화하여 보존.

각 entry는 identifier + 자연어 content + metadata(생성 시각, access 통계). Status를 action agent에 노출하지 않는 설계가 핵심 — memory agent가 task 상태를 추적하되 action agent context를 오염시키지 않음.

### 검색 메커니즘 — 없음 (LLM 판단이 검색을 대체)

이 논문은 **별도 검색 메커니즘(vector search, BM25, 키워드 검색 등)을 제공하지 않음**. 대신:

1. Bank를 **"compact"**하게 유지 → memory agent가 매 step **전체 bank** Bₜ를 읽음
2. Phase 2에서 LLM이 전체 bank + recent trajectory를 보고 **어떤 entry가 다음 action에 관련 있는지 직접 판단** → retrieval step 없음
3. 관련 정보를 기존 entry 그대로 가져오지 않고, 상황에 맞게 **새 reminder 텍스트를 합성(synthesize)**하여 주입

기존 vector/keyword retrieval이 "저장된 record를 가져오는" 방식이라면, 본 방법은 LLM이 전체 bank를 읽고 현재 상황에 맞는 reminder를 **compose**하는 방식. Ablation에서 Mem0(vector+BM25 top-10)과 명시적으로 대조됨.

> "Unlike embedding-based retrieval-augmented prompting, which retrieves fixed records from a store, this variant still uses the memory model to synthesize a reminder from the maintained bank." (§4.3)

### Phase 1: Memory Bank Management

Memory agent가 predefined **tool call 리스트**를 반환하면 system이 순서대로 실행. 자유 형식 summary가 아닌 **제약된 bank edit sequence**가 출력. Tool call이 없으면 bank 변경 없음.

| Tool Call | 용도 |
| --- | --- |
| `memory_update_status` | private progress tracking (action agent 비노출) |
| `memory_save_knowledge` | task requirements, env facts, paths, API details, constraints |
| `memory_save_procedural` | debugging experience, failed approaches, solutions, error patterns |
| `memory_delete` | outdated/incorrect entry 제거 |

Tool-call interface가 memory management를 explicit·constrained하게 만듦: Phase 1 출력은 trajectory의 free-form summary가 아닌 bank edit sequence. unsatisfied requirements, verified env facts, failed commands, successful fixes 등을 externalize하면서 bank를 long trajectory에 걸쳐 structured하게 유지.

### Phase 2: Intervention Selection

Updated bank + recent trajectory 조건으로 다음 action-agent 결정을 위한 intervention 선택. **Bank를 수정하지 않음.** 결과는 reminder rₜ 또는 null intervention ∅.

**Reminder 주입 방식:**
- rₜ를 next action-agent call에 **separate transient memory context**로 제공
- Action agent의 base instructions, tools, decoding procedure는 변경 없음
- 유일한 변화: call time에 optional memory context 추가

**Null intervention은 explicit action** — memory agent가 매 update마다 개입하는 것이 아니라, remembered item이 다음 결정에 영향을 줄 likely할 때만 개입. 불필요한 latency·token 소비·주의 분산 방지.

**유용한 intervention vs 비권장 intervention:**

| 유용한 intervention | 비권장 intervention |
| --- | --- |
| 위반 직전의 requirements | broad strategic advice |
| 현재 observation을 설명하는 env facts | 이미 visible한 정보 재진술 |
| 반복하지 않아야 할 previous attempts | action agent planning 대체 |
| 관련 diagnoses | |
| neglect된 open subgoals | |

상세 발췌 → [excerpt](../source/paper/Remember_When_It_Matters_Proactive_Memory_Agent_for_Long-Horizon_Agents_2026_Meta_AI.md)

### Triggering

Main implementation: 첫 step + 고정 interval. 더 선택적 trigger(tool errors, failed tests, repeated commands, large context shifts 후)도 가능하지만, **memory intervention policy 자체의 효과를 분리**하기 위해 고정 interval 사용.

### Learning Memory Intervention Policies

Main instantiation은 prompted model (training 불필요). 하지만 prompted memory agent는 (1) 추가 inference cost, (2) 불완전한 intervention calibration 문제. 이에 학습 가능성을 early exploration.

**학습 설계:**
- Action agent 고정, memory agent만 학습
- SFT: prompted memory agent trajectory distill → interface + memory management discipline (compact writing, stale state update, unnecessary reminder 회피)
- GRPO: sparse task-level verifier rewards → **pivot turns**(downstream success에 영향을 줄 likely한 turns, offline rollout에서 식별)에 update 집중

## Experiments & Results

### Benchmark Datasets

- **Terminal-Bench 2.0:** 자율 command-line 실행 환경. 파일 검사·명령 실행·코드 편집·디버깅·hidden verifier test 만족. 89 tasks 중 85 paired tasks (4 docker failures 제외). 환경 grounding, debugging continuity, procedural memory 강조.
- **τ²-Bench:** 대화형 tool-use 환경. airline(50), retail(114), telecom(114) = 278 tasks. user simulator와의 multi-turn 대화, 도메인 policy 준수, 환경 tool 조작. policy adherence, user-state tracking, conversation-level execution state 강조.

### Setup

- **Action agents:** Claude Sonnet 4.5 (weaker), Claude Opus 4.6 (stronger)
- **Memory agent:** Claude Opus 4.6 (main), Qwen3.5-27B (training study)
- **Memory config:** 첫 step + 매 step 호출. trajectory window k=8. Phase 1 후 Phase 2.
- **Metric:** pass@1 (Terminal-Bench: verifier 통과율, τ²-Bench: task evaluator 통과 대화)

### Results

**Main results (Table 1):**

| Benchmark | Action Model | Baseline | + Memory | Δ |
| --- | --- | --- | --- | --- |
| Terminal-Bench 2.0 | Sonnet 4.5 | 37.6% | 45.9% | **+8.3 pp** |
| Terminal-Bench 2.0 | Opus 4.6 | 43.5% | 45.9% | +2.4 pp |
| τ²-Bench (avg) | Sonnet 4.5 | 55.0% | 61.8% | **+6.8 pp** |
| τ²-Bench (avg) | Opus 4.6 | 66.2% | 68.7% | +2.5 pp |

- Weaker action agent(Sonnet)에서 gain이 크지만, stronger agent(Opus)에서도 사라지지 않음 → 단순 capacity 보상이 아님
- τ²-Bench 도메인별: airline·retail에서 큰 gain(+10.0, +9.6 pp), telecom은 작음(+2.6 pp) → domain-sensitive intervention policy의 증거

### Ablations (Table 2)

Full memory agent 대비 각 capability를 하나씩 제거:

| Variant | Phase 1 | Phase 2 | Macro | vs Full |
| --- | --- | --- | --- | --- |
| **Full memory agent** | bank mgmt | selective/silence | **64.3** | — |
| Full-bank context | bank mgmt | expose full bank | 61.5 | -2.8 |
| Always inject | bank mgmt | forced every step | 63.5 | -0.8 |
| Injection-only (no bank) | skipped | selective guidance | 61.0 | -3.3 |
| Mem0 | Mem0 ADD | vector+BM25 top-10 | 62.1 | -2.2 |

**Ablation 핵심 발견:**
- **Full-bank context** (-2.8 macro): bank 유지만으로는 부족, selective intervention 필요
- **Always inject** (-0.8 macro): 매 step 주입은 토큰·latency 비용 없이는 경쟁력 있으나 macro에서 열세, 특히 airline에서 silence의 가치 확인
- **Injection-only** (-3.3 macro): persistent memory 없는 advisor-style은 불안정 (airline에서 baseline 대비 하락)
- **Mem0** (-2.2 macro): general memory retrieval은 유용하지만 "언제 개입할까" 모델링 부재 → airline에서 0 gain

→ "passive memory exposure, always-on reminders, generic auxiliary guidance 모두 충분치 않음. 유지된 execution-state memory + selective intervention policy의 결합이 가장 균형 잡힌 gain"

### Qualitative Analysis (Table 3)

성공적인 intervention의 5가지 메커니즘:

| Mechanism | 예시 |
| --- | --- |
| **Requirement/policy reactivation** | τ² airline 보상 규칙, retail 수정 규칙 |
| **Environment grounding** | Terminal-Bench Git server setup, ARS file-write failure |
| **Failure-loop avoidance** | adaptive rejection sampling 반복 실패, telecom diagnostic retries |
| **Diagnostic carryover** | regex edge cases, SQLite gcov configuration |
| **Progress/entity tracking** | τ² telecom line lookup, retail authentication state |

**구체적 동작 예시:**

**Terminal-Bench — debugging continuity:**
- **regex-log:** memory agent가 현재 regex가 task boundary condition을 위반하고 single-digit IPv4 octet을 놓치고 있음을 지적 (requirements + diagnoses 동시 reactivation)
- **adaptive-rejection-sampler:** 반복된 file edit 실패를 procedural memory에 축적 → 나중에 environment-specific workaround를 surface하여 동일 실패 반복 방지

**τ²-Bench — policy/interaction state:**
- **airline case 1:** user가 Gold status 주장, tool output은 Regular member. baseline은 user 주장 기반 compensation(실패), memory-enabled는 "검증된 기록 의존" reminder(성공)
- **airline case 2:** basic-economy flight 수정 시도 → memory가 "수정 불가" policy clause를 reactivate하여 invalid modification 방지
- **공통 패턴:** 성공 intervention은 state-changing tool call 직전에 발생 (authentication, eligibility, one-shot tool limits, policy clauses)

성공 intervention은 specific·grounded·timely — execution state의 인과적 영향을 복원. 실패는 storage 실패가 아닌 **calibration error** (투사적 추론 과신, 중복 정보, 불필요한 concern으로 extra verification 유발). → "언제 침묵할까" 결정이 핵심 과제.

### Training Open-Weight Memory Agent (Table 4)

Qwen3.5-27B memory agent, Qwen3.5-122B-A10B action agent(frozen). SETA 학습, Terminal-Bench held-out.

| Stage | SETA Avg. Reward | Δ |
| --- | --- | --- |
| Action only (no memory) | 0.709 | — |
| Base 27B memory (untrained) | 0.693 | -0.016 |
| + SFT | 0.720 | +0.011 |
| + GRPO | **0.734** | +0.025 |

| Transfer | Pass@1 | Δ |
| --- | --- | --- |
| Action only | 37.6% | — |
| + trained (GRPO) memory | **41.1%** | +3.5 pp |

- Untrained memory agent는 오히려 성능 하락 → interface 학습 필요
- SFT로 회복, GRPO로 intervention calibration 추가 개선
- SETA → held-out Terminal-Bench 부분 transfer 성공

상세 발췌 → [excerpt](../source/paper/Remember_When_It_Matters_Proactive_Memory_Agent_for_Long-Horizon_Agents_2026_Meta_AI.md)

### Findings & Implications

1. **Memory = intervention policy 입증:** Selective intervention이 passive exposure, always-on injection, advisor-only, general retrieval 모두에 우위. 단순히 memory를 더 보여주는 것만으로는 안 됨 — "언제 개입할까" 결정이 핵심.
2. **Stronger agent에게도 유효:** Opus 4.6에서도 +2.4~2.5 pp. 단순 capacity 보상이 아닌 질적 개선.
3. **Domain sensitivity:** 도메인마다 유용한 execution state가 다름 (Terminal-Bench: debugging continuity, τ²-Bench: policy adherence). 고정 summarization이 아닌 adaptive intervention이 필요한 이유.
4. **학습 가능성:** Open-weight 27B 모델로 intervention policy를 SFT+GRPO로 학습 가능. RL이 "언제 침묵할까" 결정을 개선.
5. **Status private 설계의 가치:** Memory agent의 private status가 action agent context를 오염시키지 않으면서 task tracking 유지.

## Analysis

### Strengths & Significance

- **Behavioral state decay 개념 정립:** Long-horizon agent 실패를 "정보 부재"가 아닌 "행동 제어력 상실"로 재구성한 것이 이 논문의 가장 큰 지적 기여. Lost in the middle 현상과 구별되는 실행 맥락의 decay를 명확히 정의.
- **Memory as intervention framing:** 기존 memory 연구의 저장·검색 패러다임에 "언제 개입할까"라는 새 축 추가. Ablation으로 각 설계 결정의 독립적 기여를 cleanly 분리.
- **Plug-and-play 설계:** Action agent를 변경하지 않으므로 frontier model에 바로 적용 가능. 기존 agent harness에 minimal 수정으로 통합.
- **Private status 분리:** Memory agent가 action agent context를 오염시키지 않으면서 task 상태를 추적하는 설계가 실용적.
- **학습 가능성 초증명:** Prompted policy가 아닌 learned policy로의 경로를 open-weight 모델로 제시. RL pivot-turn focusing이 sparse reward 문제에 대한 실용적 해법.

### Limitations

- **고정 interval trigger:** tool error, failed test 등 선택적 trigger를 사용하지 않아 intervention policy 자체 효과는 분리했으나, 실제 배치에서는 adaptive trigger가 더 효율적일 수 있음 (논문도 인정).
- **추가 inference cost:** Memory agent가 매 step 호출되어 frontier model call 추가. 저자는 training으로 완화 가능하다고 주장하지만, prompted setup에서는 비용 부담.
- **도메인 의존성:** τ²-Bench telecom(+2.6 pp), Opus airline(+0.0 pp) 등 gain이 불균일. 특정 도메인에서는 memory 개입의 한계.
- **Training study의 예비성:** SETA → Terminal-Bench transfer가 +3.5 pp로 Claude Opus memory agent(+8.3 pp)에 비해 작음. "partial transfer"로 명시되어 있으며, 완전한 open-weight 대체에는 미치지 못함.
- **단일 memory agent 모델:** 실험의 대부분이 Claude Opus 4.6에 의존. 다른 memory agent 모델에서의 일반성 미검증.
- **정성 분석의 한계:** 실패 case가 calibration error로 분류되었으나, 정량적 failure rate 분석 부족.

### Future Work / Improvements

- **Joint training:** Memory agent와 action agent를 jointly 학습 (논문의 open direction).
- **Adaptive trigger 학습:** 고정 interval 대신 "memory를 언제 호출할까" 자체를 학습.
- **Reminder 형태 학습:** Verbatim reminder vs task-specific abstraction 중 어느 것이 효과적인지 식별.
- **Pivot turn 식별 개선:** Offline rollout 기반 pivot turn 식별을 online adaptive로 발전.
- **다중 memory agent:** 도메인별 특화된 memory agent 또는 계층적 memory 구조.
- **비용-성능 분석:** Memory agent 모델 크기·호출 빈도에 따른 비용-성능 trade-off 정량화.

## References

- Code: https://github.com/yifannnwu/proactive-memory-agent
- arXiv: https://arxiv.org/abs/2607.08716
- Terminal-Bench 2.0 [Merrill et al., 2026]: https://arxiv.org/abs/2601.11868
- τ²-Bench [Barres et al., 2025]: https://arxiv.org/abs/2506.07982
- SETA [Shen et al., 2026]: https://github.com/camel-ai/seta
- Mem0 [Mem0 Team, 2026]: https://github.com/mem0ai/mem0
- Memory-as-Action [Zhang et al., 2025]: https://arxiv.org/abs/2510.12635
- Lost in the middle [Liu et al., 2024]: https://aclanthology.org/2024.tacl-1.9/
