> [paper] https://arxiv.org/abs/2607.10350

# ABot-AgentOS: A General Robotic Agent OS with Lifelong Multi-modal Memory

## Summary & Outline

Alibaba AMAP CV Lab이 제안한 **범용 로봇 Agent OS**. VLM/VLA 기반 perception-action 모델 위에 얹는 deliberative agent layer로서, (1) scene-conditioned planning, (2) context-isolated skill 실행, (3) multi-stage verification, (4) multi-modal graph memory, (5) edge-cloud collaboration을 통합 제공한다. 평가를 위해 16개 실내·외·혼합 씬, 4단계 난이도, 200+ 과제로 구성된 executable benchmark **EmbodiedWorldBench**를 제안하고, memory 전용 모듈은 LoCoMo·OpenEQA·Mem-Gallery·NExT-QA·EgoLife 5개 벤치마크에서 별도 검증한다. 논문 구성은 크게 Introduction → Agent Framework(§2, memory 포함) → EmbodiedWorldBench(§3) → 소형 모델 학습 파이프라인(§4) → Experiments(§5) → Conclusion 순.

## Problem & Motivation

- **연구 배경**: VLM/VLA는 로봇의 perception·action prediction을 크게 개선했지만, long-horizon embodied agent에는 reasoning·memory·tool use·verification·cross-embodiment execution을 아우르는 범용 runtime layer가 없었다.
- **풀고자 하는 문제**: embodied agent를 위한 agent operating system 설계 — task decomposition, skill delegation, 실행 검증(verification), 멀티모달 장기 메모리를 하나의 재사용 가능한 layer로 통합하는 것 (long-horizon task planning, embodied question answering, lifelong memory self-evolution).
- **기존 접근의 한계**: 저자들은 기존 연구를 3갈래로 나눠 한계를 짚는다.
  1. **로봇 파운데이션 모델 계열** (Galaxea G0, Hi Robot, RoboBrain, RoboMemory 등): perception-to-action mapping은 발전시켰지만 특정 모델 스택·embodiment·control interface에 종속됨.
  2. **범용 에이전트 계열** (ReAct, Toolformer, SWE-agent, OpenAI Operator, Claude computer use 등): tool use·추론 인터리빙을 보여줬지만 partial observability, actuation uncertainty, ambiguous completion signal 등 물리 실행 고유의 제약을 다루지 않음.
  3. **에이전트 메모리 계열** (Generative Agents, MemoryBank, MemGPT, Mem0): parametric memory를 넘어선 지속 메모리를 보여줬지만, dialogue·egocentric vision·object state·identity·공간/시간 관계·robot task trace를 함께 묶는 멀티모달·관계형 substrate는 부재.
  - 종합하면 **reasoning-execution gap**(중간 agent layer 부재), **embodiment-generalization gap**(하드웨어 종속), **persistent embodied-memory gap**(멀티모달·관계형·auditable 메모리 부재) 3가지 gap이 남는다.

## Contributions

- **시스템**: VLM과 물리 실행을 잇는 모듈형 Agent OS 아키텍처 — main LLM / Skill Runner / Verifier로 역할을 분리한 Agent Harness, edge Tiny LLM ↔ cloud Large LLM 협업 라우팅, plugin 방식 skill 통합으로 humanoid·quadruped 등 이형 embodiment에 재사용.
- **벤치마크**: EmbodiedWorldBench — 16개 씬(실내/실외/hybrid), 4단계 난이도, 200+ executable task, trace-grounded 채점으로 navigation·object search·NPC dialogue·dynamic event·partial observability를 하나의 시나리오에서 동시 평가.
- **메모리(핵심)**: Universal Multi-modal Graph Memory — dialogue·visual observation·spatial/temporal context·task trace를 typed node/edge로 변환하는 source-grounded 메모리 그래프 + failure-driven lifelong self-evolution loop (진단→제안→컴파일→게이트 승인된 evo-asset만 **차기 split**에 적용, 현재 split에는 절대 leakage되지 않음).
- **부가 기여(실증)**: 소형 배포용 모델을 위한 text-based sandbox 학습 파이프라인(teacher distillation → SFT → GiGPO 기반 RL) 및 self-evolving LLM-as-a-Judge reward engine(Meta-Judge + multi-agent prompt refinement).

## Method

### 1) Agent Framework — 3-역할 Harness

```
                          ┌──────────── Edge-Cloud Routing (learned policy) ────────────┐
                          │  Tiny LLM(edge): 매 턴 저지연 처리                          │
                          │  → 복잡 추론/장기계획/모호성 시 cloud Large LLM escalate     │
                          └────────────────────────────────────────────────────────────┘
User Instruction ─▶ Main LLM (scene-conditioned semantic planner)
                        │   관측 입력: 현재 위치 · 맵 구조 · 가시 evidence ·
                        │              recent history · available skills · robot state
                        │  ├─ 직접 tool 호출(단발) ──────────────▶ Skills/Tools
                        │  ├─ 서브태스크 위임 ──▶ Skill Runner (격리된 local context)
                        │  │                     subgoal · recent obs · skill state
                        │  │                     failed attempts · recovery strategy
                        │  │                         │  절차 연속성·국소 복구·정체 감지
                        │  │                         ▼
                        │  │              compact semantic summary 반환
                        │  │              (달성여부/실패원인/발견 씬 정보/복구시도/
                        │  │               main LLM 활용 가이드)
                        │  ▼
                        │  ◀── new obs · tool 결과 · skill summary · verifier 피드백
                        │        → plan 갱신 / goal 전환 / 종료 여부 판단
                        ▼
                    Verifier (supervisory signal, 단순 final evaluator 아님)
                        │   runtime: trajectory·skill state의 정체·반복충돌·local loop·
                        │            plan 불일치 감시
                        │   skill-level: subtask가 semantic objective를 실제로 달성했는지
                        │                 (정상 반환=성공 아님)
                        │   finish-time: 원 instruction·실행 history·skill summary·
                        │                 observation·추가 요구사항 종합 후 종료 허용
                        ▼
              반려 시 missing condition을 main LLM로 회귀 → 재계획 (closed loop)
```

#### 설계 동기 — 왜 단일 컨트롤러가 아닌가

저자들은 기존 ReAct(추론-행동 인터리빙), SWE-agent(컴퓨터 인터페이스), Toolformer(API tool use)가 "코드/디지털 작업은 외부 신호(테스트, API 응답, page state)로 중간 단계 완료를 검증할 수 있다"는 암묵적 전제 위에 있다고 본다. 반면 embodied agent는 그런 명시적 completion signal이 없다 — 로봇이 navigation 명령을 호출했지만 실제로는 현 위치에 머물거나, 회전하며 반복 충돌하면서도 **언어 수준에서는 "진행 중"이라고 믿는** 일이 발생한다. 따라서 ABot-AgentOS는 도구를 더 많이 부르는 식의 접근이 아니라 reasoning-execution-verification을 **명시적으로 역할 분리한 closed loop**로 구성한다.

#### Main LLM — Scene-Conditioned Semantic Planner

핵심은 "언어 해석"이 아니라 **"현재 scene에 대한 조건부 해석"**이라는 점이다. 같은 user instruction이라도 현재 위치·알려진 맵 구조·reachable regions·주변 objects·가시 visual evidence·interaction history에 따라 다른 실행 전략이 필요하므로, main LLM은 이들을 함께 관측한 뒤 아래를 판단한다.

- 이 instruction이 필요로 하는 것이 **navigation / search / human interaction / reporting / manipulation / additional observation** 중 어느 것인지 분류
- 이미 충분한 정보인지, 추가 관측/쿼리가 필요한지, 현재 상태에서 feasible한 skill은 무엇인지, **어떤 observable conditions이 completion인지**를 명시한 revisable high-level plan 수립
- 실행 중 new observation·tool result·skill summary·verifier feedback이 context에 들어올 때마다 plan을 갱신하고, 어떤 goal을 추구할지 / 직접 tool로 충분한지 / Skill Runner에 위임할지 / 종료할지를 결정

매 저수준 movement를 직접 발행하지 않는 이유는, 그런 local detail이 main reasoning thread로 들어오면 global task state가 procedural noise에 가려지기 때문이다. local 실행 디테일은 Skill Runner로 격리하고, main LLM은 global stage management에만 집중한다.

#### Skill Runner — Isolated Local Context Subagent

skill을 one-shot tool call이나 action macro로 보지 않고, **subgoal · recent observations · skill state · failed attempts · recovery strategy를 담은 isolated local context를 가진 subagent**로 취급한다. 이 격리가 필요한 이유는 embodied 환경에서 local 실행이 반복 이동·재관측·relocalization·view 조정·복구를 수반하기 때문 — 매 충돌·단거리 이동·실패 시도·시각 조정이 main LLM context에 append되면 global objective가 procedural detail에 묻힌다.

실행 중 procedural continuity를 유지하기 위해 (i) 진행 중인지, (ii) 추가 관측·국소 복구가 필요한 시점인지, (iii) control을 main LLM으로 반환할 시점인지를 self-check한다. 종료 시 **compact semantic summary**만 반환하며, 이 안에는 달성 여부 / 실패 원인 / 발견한 scene 정보 / 시도한 복구 / main LLM이 이 결과를 어떻게 활용해야 하는지가 포함된다. 즉 main LLM은 intermediate action sequence 전체가 아니라 이 압축 요약만 받는다.

#### Verifier — Multi-Stage Supervisory Signal

"성공 검증 외부 신호"가 없는 embodied 환경에서, agent declared progress · execution trajectory · observed scene state 사이의 consistency를 검사한다. 검증은 final evaluator가 아니라 **실행 before/during/after에 모두 작용하는 supervisory signal**이며, 반려 시 missing condition을 main LLM로 돌려 재계획을 유발한다.

| 단계 | 검사 대상 | 핵심 체크 |
|---|---|---|
| **Runtime** | recent trajectory + skill state | 유효 진행 여부, 아니면 stagnation / repeated collisions / local loop / 현재 plan과 불일치한 행동인지 |
| **Skill-level** | 위임된 subtask 결과 | tool/subagent가 정상 반환했다고 해서 success로 accept하지 않고, **semantic objective를 실제로 충족했는지** 별도 검사 |
| **Finish-time** | 종료 시도 시 전체 맥락 | 원 instruction · 현재 plan · execution history · skill summaries · observations · interaction 중 새로 생긴 요구사항을 종합한 뒤 종료 허용 |

디지털 작업의 verifier가 "테스트 통과/실패" 이진 신호라면, ABot-AgentOS의 verifier는 **"언어 수준의 믿음"과 "환경 근거"의 불일치를 단계별로 잡아내는** 구조다. 이것이 procedural drift(충돌 반복하면서도 진행 중이라 믿기)와 premature termination(조기 종료)을 줄이는 직접 메커니즘이다.

#### Edge-Cloud Collaborative Routing

on-device Tiny LLM이 매 턴을 먼저 처리(저지연 perception · instruction 이해 · state tracking · routine decision)하고, 복잡한 scene understanding / 장기 planning / skill execution / verification이 필요할 때만 cloud Large LLM으로 escalate한다. 라우팅 정책은 고정 규칙이 아니라 **training sample + execution feedback으로 학습된 policy**로, 무엇을 local tool로 풀지 / cloud planning이 필요한지 / routing 전 추가 관측이 필요한지를 결정한다.

#### 구체 예시 — 교외 주택가 compound task walkthrough

논문 Figure 5/§3.1이 드는 **대표 시나리오**로 3역할 harness가 어떻게 맞물리는지 보여준다. 아래는 논문에 명시된 시나리오 설정(§3.1 line 607-612, 쿼리 예시 §3.2.2 line 642, 가시-evidence/위치 혼동 실패 §5.1 line 1054-1059)과 §2.2.1~2.2.3 메커니즘을 결합해 **재구성한 walkthrough**다 (※ 단계별 메시지는 논문 원문이 아니라 메커니즘을 시나리오에 적용한 해설).

**환경 설정 (논문 원문 기반)**:
- 씬: 교외 주택가 — outdoor street · residential backyard(수영장) · indoor living room(가전)이 한 episode에서 traversing됨.
- compound instruction 예: "빨간 패딩 입은 노인을 찾아 위치를 보고하라" + 거리 점검 · 뒷마당 수영장 상태 확인 · 실내 가전 상태 검증 · 복귀 보고 4개 sub-goal.
- agent가 받는 입력: **filtered semantic map + 자연어 instruction만** — NPC 위치·평가 signal·예상 trajectory는 hidden(visibility isolation, §3.2.2).
- 동적 event: 고난이도 시나리오는 NPC relocation · object-state change가 중간에 발생해 plan 갱신을 요구함.

**단계별 동작**:

```
① Main LLM (scene-conditioned planning)
   관측: 현재 위치(실내 입구) · filtered semantic map(거리/뒷마당/실내 영역 + POI) ·
         가시 evidence(아직 한정적) · history(초기) · available skills(Navigate/Observe/Ask/Report)
   판단: instruction이 navigation + search + observation + reporting을 모두 요구.
         completion condition = 4개 sub-goal 모두 관측 근거 확보 + 복귀 후 보고.
   결정: sub-goal 1(거리 점검)은 Skill Runner에 위임, 직접 tool 호출이 아님.
         → "Skill Runner에 위임: Navigate to street, Observe elderly-in-red-jacket, 반환 시 발견 정보 포함"

② Skill Runner (격리된 local context)
   local context: subgoal(street 점검) · recent obs(실내 입구 시야) · skill state ·
                  failed attempts(없음) · recovery strategy(미정)
   실행: 반복 Navigate → relocalize → Observe(첫인칭 view를 VLM이 textual evidence로 변환)
         충돌/경로 이탈 시 국소 복구(local loop 안에서 해결, main context에 안 올림)
   종료: compact semantic summary 반환
         "sub-goal: 달성. 거리에서 빨간 패딩 노인 발견(위치: 교차로 근처).
          발견 scene 정보: 거리 끝에 공사 표지. 복구: 없음.
          main LLM 활용 가이드: sub-goal 2(뒷마당)로 진행 가능."

③ Main LLM (plan 갱신)
   new input: ② skill summary + new observation
   결정: sub-goal 2(뒷마당 수영장) 위임. completion condition 중 1个 충족 표시.

④ Skill Runner — 뒷마당 이동 중 정체 발생
   runtimeVerifier 감지: trajectory에 stagnation / repeated collision 패턴 → 진행 아님.
   피드백: Skill Runner가 local context 안에서 recovery(view 조정·경로 재시도) 시도.
   회복 불가 시 control을 main LLM으로 반환(실패 원인 포함 summary).
   → Main LLM은 plan 갱신 후 대체 경로/스킬 선택.

⑤ 신규 동적 event — NPC가 relocation 함
   Observe 결과가 기존 memory graph의 NPC 위치와 불일치.
   Main LLM: memory(context) 재참조 → "NPC가 이동했음" plan에 반영, search 범위 확장 결정.

⑥ Finish-time 검증 (Verifier) — belief vs. evidence 불일치 사례
   Main LLM이 "4개 sub-goal 모두 완료, 보고 준비됨"으로 종료 시도.
   Verifier가 원 instruction · plan · 관측 근거 교차 확인:
     - sub-goal 3(실내 가전): agent가 "이미 확인했다" 선언
     - 그러나 observation trace에 해당 가전에 대한 textual evidence 부재
     - 오히려 §5.1이 지적한 전형적 실패: 가시 영역 ↔ 자기 위치 혼동 —
       "실내에서 outdoor가 보였다" → "이미 outdoor로 이동 완료"로 해석한 흔적
   판정: 종료 반려, missing condition(가전 관측 evidence)을 main LLM로 회귀
   → Main LLM 재계획: 실내로 복귀 후 가전 관측 재수행

⑦ 최종 종료
   모든 sub-goal이 observation-grounded evidence로 충족된 경우에만 finish-time 통과 → 보고.
```

이 walkthrough가 보여주는 핵심: 단일 ReAct 컨트롤러였다면 ④의 충돌·⑥의 가시-위치 혼동이 모두 main context에 append되어 global objective가 procedural detail에 가려졌을 것이다. 3역할 분리 + 3단 검증이 없었다면 ⑥처럼 "언어 수준 믿음"만으로 조기 종료(premature termination)가 일어났을 가능성이 높다(이것이 §2.2 설계 동기가 직접 겨냥하는 실패 모드).

### 2) Multi-modal Graph Memory

```
관측/대화/영상 ──▶ [Memory Writer: 멀티모달 정규화] ──▶ Typed Graph G=(V,E)
                                                          │  V: source container, evidence unit,
                                                          │     entity, place, session, event
                                                          │  E: temporal/spatial/identity/
                                                          │     interaction/provenance edge
                                                          ▼
Query ──▶ [Hybrid Seed Selection] ──▶ [Typed-edge 확장: local evidence subgraph]
              s(q,v) = λ_sem·s_sem + λ_lex·s_lex          (고정 depth·token budget)
                       + λ_meta·s_meta + λ_type·s_type          │
                                                                  ▼
                                          Evidence context 직렬화 + Retrieval Trace 기록
                                                                  │
                                                                  ▼
                                              Answerer (evidence-grounded 응답, 불확실 시 보류)
```

- 텍스트 chunk RAG와의 핵심 차이는 **seed node → 관계 확장**이라는 점: 단순 유사도 snippet이 아니라 identity·시간·위치·참여관계·provenance까지 함께 추론 가능한 subgraph를 만든다.
- Writer는 원본을 그대로 적재하지 않고 semantic compression을 우선(요약된 typed record). 중복 entity/event는 provenance·temporal context가 맞으면 병합하고, 오래된 state는 삭제 대신 temporal edge로 superseded 표시(현재 vs. 과거 상태 구분 유지).
- 상세 공식·표는 [발췌 파일](../source/paper/ABot-AgentOS_A_General_Robotic_Agent_OS_with_Lifelong_Multi-modal_Memory_2026_Alibaba.md) 참조.

#### 구체 예시 — "강아지 입양" 발화 저장 → 후속 질문 검색

논문 §2.3.2는 저장 예시로 "I adopted a Maltese dog yesterday"라는 발화 하나를 든다. 이를 §2.3.1/§2.3.3에서 설명한 노드 스키마·검색 공식에 대응시켜 재구성하면 다음과 같은 저장→검색 흐름이 된다 (※ 아래 JSON은 논문이 서술한 필드를 바탕으로 재구성한 예시이며 논문에 실제로 실린 원문 JSON은 아님).

**1. 저장(Writing) — 발화 1건이 typed graph로 변환**

```
입력: 대화 turn "I adopted a Maltese dog yesterday" (+ 첨부 이미지 1장)

출력 (semantic event 노드 + 연결 엣지):
Event #e1301
  ├─ type: "pet_ownership_event"
  ├─ evidence_summary: "speaker adopted a Maltese dog"
  ├─ time_ref: resolved("yesterday" → session_date - 1d)     ← 시간 정규화
  ├─ confidence: 0.9
  ├─ provenance: {source_id: session_042/turn_17, extractor: writer-v3}
  │
  ├──edge(participant)──▶ Entity #person_08 (speaker)
  ├──edge(identity_hypothesis)──▶ Entity #pet_dog_11 {breed: "Maltese"}
  └──edge(evidence)──▶ Image #img_301 (첨부 사진, supporting evidence)
```

- 원본 대화 문장을 그대로 저장하지 않고, **시간 정규화("yesterday"→실제 날짜)**, **entity 연결(발화자↔반려동물)**, **이미지 근거 연결**까지 마친 압축된 typed record로 적재한다.
- 이미지가 없었다면 identity_hypothesis edge의 confidence만 낮게 유지되고, 이후 이미지가 들어오면 같은 Entity에 evidence edge가 추가되는 방식으로 점진적으로 보강된다.

**2. 검색(Retrieval) — 후속 질문 "내가 키우는 강아지 종이 뭐였지?"**

```
Query: "내가 키우는 강아지 종이 뭐였지?"
   │
   ▼ 식(2) 적용
s(q, v) = λ_sem·s_sem(q,v)   ← "강아지"/"dog" 임베딩 유사도
        + λ_lex·s_lex(q,v)   ← 어휘 중첩 낮음(패러프레이즈)
        + λ_meta·s_meta(q,v) ← speaker=person_08 메타데이터 일치
        + λ_type·s_type(q,v) ← 질의가 entity/attribute 타입을 기대 → 가중 상향
   │
   ▼ top seed: Entity #pet_dog_11
   ▼ typed-edge 확장(depth 고정): identity_hypothesis edge → Event #e1301 → evidence edge → Image #img_301
   │
   ▼ evidence subgraph 직렬화 → answerer에 주입
   │
   ▼ 응답: "Maltese종 강아지를 키우고 있습니다" (+ retrieval trace: e1301, img_301 인용)
```

이 예시가 보여주는 핵심은, 단순 텍스트 chunk RAG였다면 "강아지 종"이라는 어휘가 일치하는 스니펫만 찾았을 것이지만, 그래프 메모리는 **entity 노드를 seed로 잡고 관련 event·evidence까지 함께 확장**해서 "누가 언제 어떤 근거로" 답했는지까지 추적 가능한 응답을 만든다는 점이다.

### 3) Failure-Driven Lifelong Self-Evolution

#### 핵심 관점 — "메모리는 content만 쌓는 게 아니라 pipeline 자체가 개선되어야 한다"

저자의 출발점 (§2.3.4 line 479-481): 메모리가 단순히 새 content를 accumulate하기만 하면, 같은 extraction/retrieval/temporal-grounding/visual-selection/answer-composition 오류가 반복된다. 그래서 **memory graph(content-level)와 evo-asset set(pipeline-level) 두 형태의 lifelong 지식**을 따로 accumulate한다. evo-asset은 Python 코드가 아니라 lifecycle-managed **선언적 JSON DSL record**로, target layer·triggering condition·permitted action·safety constraint·provenance·validation result·version id를 명시한다.

#### Split-wise 프로토콜 — 왜 이렇게 하는가

```
시간축: D₁ → D₂ → ... → Dₜ (순서가 있는 disjoint split)

split t 실행:
  G_{t-1} (누적 memory graph) + A_{<t} (이전 split까지 승격된 evo-asset들)
  │
  ▼ Run(D_t; G_{t-1}, A_{<t})  ← 현 split은 오직 이전 asset만 사용
  T_t (answer traces + retrieval traces + failure traces) + G_t (graph 갱신)
  │
  ▼ split t 평가 완료 후 (post-hoc):
  Diagnose(T_t) → Propose → Compile → Gate
  │
  ▼ ΔA_t (승인된 신규 evo-asset)
  │
  ▼ A_{≤t} = A_{<t} ∪ ΔA_t  → split t+1부터 사용 (split t 채점에는 절대 미적용)
```

- **No-leakage**: split t에서 생성된 evo-asset은 split t 평가에 절대 쓰이지 않는다. benchmark에서는 ground-truth 정답이 post-hoc correctness signal으로만 사용 (inference 중 아님, memory graph에 write 아님, 동일 split asset 생성 아님).
- **Cumulative lifelong**: 일회성 post-hoc repair가 아니라, 승인된 asset이 누적되어 이후 split에서 계속 작동. deployment에서는 사용자 교정·환경 성공/실패 신호로 ground-truth를 대체 (동일 메커니즘).
- **두 형태의 lifelong 지식**: content-level(memory graph) + pipeline-level(evo-asset set).

#### evo-asset 승격은 2단계 (매우 보수적)

```
① Candidate Gate (개별 후보 평가)
   Accept(a) = I[ΔS_target(a) ≥ τ_gain ∧ ΔS_reg(a) ≥ −τ_reg]   ... (6)
   - target subset 개선 + regression subset 악화 억제
   - 추가 검사: protected-category regression, low-score count 증가 금지

② Stack Confirmation (승인된 후보 전체 재평가)
   - 개별 Gate를 통과한 asset들을 stack으로 묶어 현 incumbent와 다시 대결
   - stack이 incumbent를 이길 때만 최종 승격, 아니면 해당 round의 신규 asset 전부 deprecated
```

**8-split 실제 결과 (Appendix B Table 9)**: Gate 후보를 낸 split은 7개, 그 중 candidate gate를 통과한 split은 3개(split_00, split_03, split_05, split_07), 최종 stack confirmation까지 통과해 승격된 것은 **split_00 단 1개**뿐. 나머지는 전부 deprecated. 즉 self-evolution은 매 split 후보를 제안하되 실제 승격률은 극히 낮은 보수적 필터링 과정이다.

#### 실패 유형은 "검색 결과 없음"만이 아니다 — 대상 layer 6종

> 현재 아래 split_00 사례가 **검색 실패** 하나만 상세히 다뤄서 "모든 실패가 검색 실패"처럼 보일 수 있으나, 원문은 이를 명시적으로 부정한다. §2.3.4 line 500-501, 540-541이 나열하는 evo-asset **대상 layer는 6종**이다:

| 대상 layer | 개선 내용 | 전형 실패 모드 |
|---|---|---|
| **Memory writing** | adapter가 관측한 keyframe mention을 typed object observation으로 materialize, alias/state/room 연결/last-seen/source ref 보존 | derived-fact missing (관측은 있었으나 graph에 typed node로 승격 안 됨 — split_00이 이 사례) |
| **Evidence selection** | 검색된 evidence subgraph 중 answerer에 주입할 record 우선순위 | 이미지 근거는 있으나 **breed-specific cue가 빠져서** identity 검색은 성공했는데 답이 불완전 (Figure 4 좌측) |
| **Retriever ranking** | room/place anchor·last-seen·object identity·directed spatial grounding 기준 우선 랭킹 | broad scene-level 어휘 유사도가 관측 기반 object record를 이겨서 관련 없는 record 반환 (split_00) |
| **Frame selection** | video/egocentric에서 어떤 frame을 memory에 넣을지 | EgoLife 등 장기 egocentric에서 핵심 frame 누락 |
| **Temporal normalization** | "yesterday" / 상대 날짜를 session metadata로 실제 날짜로 정규화 | session metadata 기반 상대 날짜 해석 오류 (Figure 4 우측, LoCoMo의 주요 실패) |
| **Answerer calibration** | evidence가 불충분할 때 보류/거부, evidence-grounded 응답 보정 | Mem-Gallery conflict detection/refusal 카테고리 |

#### 벤치마크별로 잡히는 "systematic pipeline error"가 다르다

§5.2.4 line 1326-1328가 명시하는 벤치마크별 주요 개선 대상 (실패 유형이 한 곳에 편중되지 않음):

| 벤치마크 | self-evolution이 개선한 주요 실패 유형 | 개선 폭 |
|---|---|---|
| **LoCoMo** | temporal normalization (상대 날짜 해석) | 87.5 → 88.7 (+1.2) |
| **OpenEQA** | embodied scene disambiguation (object-room materialization) | 59.9 → 60.4 (+0.5, primary +1.2) |
| **Mem-Gallery** | relation/conflict handling (multi-entity, conflict detection/refusal) | 88.6 → 89.0 (+0.4, 카테고리별 집중) |
| **NExT-QA** | causal-temporal reasoning (인과/시간 비디오 QA) | Acc@All 최대 +4.1 |
| **EgoLife** | TaskMaster 카테고리 (egocentric 장기 일상) | 65.4 → 66.2 (+0.8) |

즉 "검색 실패"는 OpenEQA split_00 한 사례일 뿐, 다른 벤치마크에서는 temporal-normalization·relation-conflict·causal-temporal·evidence-selection 등 **전혀 다른 pipeline layer**의 오류가 주로 진단된다. 이게 저자가 "self-evolution이 파이프라인 자체를 개선한다"고 주장하는 근거다.

#### Diagnoser가 실패 원인을 어떻게 detect하는가

Appendix B procedure (line 1900-1946) 기준:

1. **Incumbent 평가**: 현 memory 시스템으로 split t 실행 → answer/retrieval/execution trace 확보
2. **Diagnose**: failed QA rows + retrieval traces + graph evidence를 함께 검사 → **memory root cause별 클러스터링** (답변 포맷 오류로 분류하지 않음)
   - split_00 예시 진단명: *"derived-fact missing / adapter-to-graph materialization failure"* — "관측 evidence는 있었으나 graph에 typed node로 materialize 안 됨 → 검색이 관련 없는 record를 반환 → answerer가 unsupported/missing 답" → 답변 포맷 패치가 아니라 memory system 자체 실패로 분류
3. **Propose**: HypothesisGenerator가 여러 layer(writer/retriever/answerer/frame-policy)의 후보 JSON DSL asset 제안
4. **Compile & safety review**: CompilerCritic이 실행 코드·schema migration·direct answer resolver·gold-answer table을 reject
5. **Gate analysis**: GateAnalyst이 protected-category constraint·activation guard·cost/lifecycle constraint 추가
6. **Candidate evaluation**: 각 asset을 incumbent 대비 평가 (target 개선 + global regression 억제 + low-score-count 증가 금지 + protected-category regression 금지)
7. **Stack confirmation**: 통과 asset 전체를 stack으로 재평가, stack이 incumbent를 이겨야 최종 승격, 아니면 전부 deprecated

#### 구체 사례 — OpenEQA `split_00` (Appendix B, 논문 원문 기반)

아래는 6종 layer 중 **retriever ranking + memory writing**이 결합된 한 사례다 (검색 실패로 보이지만 진단은 writer의 materialization 실패 + retriever ranking 개선으로 해결).

논문 부록 B는 실제 8-split self-evolution 실행 로그 중 `split_00`을 상세히 공개한다. "검색 실패가 어떻게 evo-asset으로 바뀌는지"를 보여주는 가장 구체적인 예시다.

**① 실패 진단 (Diagnoser)**

```
관측 evidence: "bed 위에 throw blanket이 있다" (scene evidence로 존재)
      │
      ▼ 그러나 그래프에는...
그래프 상태: 해당 object에 대응하는 typed object node / directed object-room 관계가 없음
      │
      ▼ 검색 결과
검색 결과: 관련 없는 다른 object record가 반환됨
      │
      ▼ 결과
답변: unsupported 또는 missing answer
```
→ 진단명: **"derived-fact missing / adapter-to-graph materialization failure"** (답변 포맷 문제가 아니라 메모리 시스템 자체의 실패로 분류).

**② 후보 evo-asset 3종 (HypothesisGenerator → Table 7)**

| 후보 | 대상 레이어 | 내용 |
|---|---|---|
| Writer materialization | Writer | adapter의 keyframe object mention을 alias·attribute·상태·room 연결·last-seen evidence·source reference·confidence를 갖춘 typed object observation으로 승격 |
| **Retriever room-anchor/last-seen focus** | Retriever | object anchor·room/place cue가 있으면 폭넓은 scene-level 어휘 유사도보다 room/place·last-seen evidence·object identity·spatial grounding 기준으로 우선 검색 |
| Writer directed support relations | Writer | 양 끝점과 관계 방향이 관측으로 명시적으로 grounding된 경우에만 spatial support 관계를 기록 |

**③ 안전 가드 (CompilerCritic/GateAnalyst)** — 예: 질문 텍스트만으로 object를 쓰지 않기, 관측되지 않은 state/affordance/relation을 추론하지 않기, spatial relation 방향을 뒤집지 않기, source reference·confidence·last-seen evidence를 감사 가능하도록 보존하기 등.

**④ Gate 결과 (Table 8)**

| 후보 | 결과 | 이유 |
|---|---|---|
| Writer materialization | **거부** | target·global 점수는 개선됐지만 protected object-state recognition이 허용 오차 이상 악화 |
| **Retriever room-anchor/last-seen** | **승인** | target Δ = **+0.800**, global Δ = **+0.044**, low-score count 91→77(−10) |
| Writer directed support relations | 거부 | target Δ가 음수 + protected object-state recognition 악화 |

**⑤ 승인된 최종 runtime policy** (논문 원문 인용):

> "When an OpenEQA retrieval query contains an object anchor, room/place cue, or spatial phrase, rank observation-grounded object memories ahead of broad scene summaries. Prefer records with matching room/place, explicit last_seen, source_ref/frame evidence, object identity, and directed spatial relation evidence. Do not infer room priors, reverse relation direction, or use ungrounded scene summaries as object evidence."

이 policy는 **정답을 담은 패치가 아니라 범용 검색 랭킹 규칙**이며, gold answer나 question-specific 단축 로직을 포함하지 않는다.

**⑥ 8-split 전체 트레이스 (Table 9)의 시사점** — `split_00`처럼 승인까지 간 경우는 소수. 나머지 7개 split(`split_01~07`) 대부분은 "candidate 단계는 통과했지만 stack 전체 재확인에서 실패"하거나 "target/global regression으로 거부"되어 새 asset이 승격되지 못했다. 즉 self-evolution은 **매 split마다 후보를 제안하지만 실제 승격 비율은 낮은, 보수적인 필터링 과정**이라는 점을 이 실제 로그가 보여준다.

- **Figure 4가 보여주는 다른 두 실패-개선 사례** — split_00(검색/ materialization)과 달리, 검색 자체는 성공했지만 다른 layer에서 실패한 경우:
  - **좌측 (visual memory QA)**: 이미지 근거로 identity 검색은 성공했으나, **breed-specific cue가 빠져서** 답이 불완전. → evidence-selection / frame-selection / memory-writing 개선으로 해결 (retriever가 아님).
  - **우측 (temporal text memory QA)**: "yesterday" 등 상대 날짜를 session metadata로 정규화하다가 **temporal-normalization 오류** 노출. → temporal-normalization policy 개선으로 해결 (역시 retriever가 아님).
  - 이 두 예시가 핵심: self-evolution이 "검색 결과 없음"만 잡는 게 아니라, identity/evidence/temporal 각 layer의 **systematic pipeline 오류**를 잡는다는 것을 Figure 4가 직접 보여준다.
- 상세 발췌 → [source/paper 발췌 파일](../source/paper/ABot-AgentOS_A_General_Robotic_Agent_OS_with_Lifelong_Multi-modal_Memory_2026_Alibaba.md)의 "Appendix B" 섹션.

### 4) Edge-Cloud Collaborative Memory

로봇별 private memory(지도·개인 히스토리 등)와 클라우드 공용 memory(장애물·랜드마크 등 공개 정보)를 분리. 신규 메모리 아이템마다 privacy gating을 적용해 개인식별정보 포함 시 업로드 차단(자체 구축 privacy 분류셋에서 99%+ 정확도).

### 5) 소형 모델 배포 파이프라인 (§4, 부가 기여)

Text-based sandbox(Easy/Medium/Hard, env_state·failure_triggers·human_persona 구조) → teacher 모델의 ReAct 궤적 distillation → LLM-judge 필터링 → SFT → GiGPO 기반 온라인 RL(episode-advantage + step-advantage) → self-evolving LLM-as-a-Judge reward engine(Meta-Judge로 저품질 judge 사례를 진단 → Cluster/Analyzer/Refiner/Validator 멀티에이전트로 reward prompt를 국소 수정). Judge 인간 정합도가 약 60%→90%+로 개선.

## Experiments & Results

### Benchmark Datasets
- **EmbodiedWorldBench**(자체 제안, UnrealZoo 기반 UE5 씬): 16 씬(병원·박물관·마트·교외 주택가 등), 4 난이도, 300+ waypoint, 200+ task. 각 task는 `⟨M, S₀, O, N, C⟩`(맵·초기상태·관측규칙·NPC 행동·성공기준) 형태의 실행 가능한 시나리오.
- **메모리 벤치마크 5종**: LoCoMo(초장기 멀티세션 대화), OpenEQA EM-EQA(open-vocab embodied QA), Mem-Gallery(멀티모달 장기 대화+비전 의존), NExT-QA(시간/인과 비디오 QA), EgoLife(초장문 egocentric 일상 QA). 모달리티·시간축·평가 프로토콜이 서로 달라 점수를 통합하지 않고 개별 리포트.

### Setup
- Agent 평가: baseline은 단일 컨트롤러 ReAct, 비교축은 동일 backbone(Qwen3.6-Plus)에서의 ABot-AgentOS 구조 효과, 그리고 backbone 교체(DeepSeek-V4-Pro) 효과.
- Memory 평가: 모든 실험이 **동일한 hybrid graph retriever**를 고정 사용(검색 자체는 변수 아님). writer/answerer/judge는 데이터셋별로 Qwen3.6-Plus, GPT-5.4, Qwen3.5-Flash 등을 벤치마크 프로토콜에 맞게 선택. gold 정답/근거는 표준 메모리 그래프 구축에는 절대 사용하지 않고, self-evolution에서만 **해당 split 평가 이후**에 한해 사용.

### Results
- **Agent (EmbodiedWorldBench subset)**: 동일 모델(Qwen3.6-Plus)에서 ABot-AgentOS가 ReAct baseline 대비 TSR +11.99pp, GCR +10.84pp. Backbone을 DeepSeek-V4-Pro로 바꾸면 추가로 TSR +6.22pp, GCR +5.83pp.
- **Memory (static)**: LoCoMo 87.5(Mem0 대비 +1.9, human 87.9에 근접) / OpenEQA EM-EQA 59.9(3D snapshot·Gaussian splatting 등 memory baseline 상회) / Mem-Gallery 88.6 / NExT-QA Acc@All 76.5(GraphVideoAgent +3.2) / EgoLife 65.4%(단 1개 프레임만 검색해도 최고 평균 정확도).
- **Self-evolution**: LoCoMo 87.5→88.7, OpenEQA 59.9→60.4, Mem-Gallery 88.6→89.0, NExT-QA는 카테고리 최대 +4.1, EgoLife 65.4→66.2 — 모두 **현재 split의 정답 leakage 없이** 이전 split에서 승격된 evo-asset만으로 달성.

### Findings & Implications
- Hierarchical 구조(작업 메모리 + skill-level feedback + finish-time verification)가 단일 컨트롤러보다 TSR·GCR을 **동시에** 끌어올려, 단순 부분진행이 아니라 완전 성공으로의 전환을 늘린다.
- Graph memory는 "단일 스니펫으로 답할 수 없는" 질문(정체성 연속성, 시간 정규화, 인과/시간관계, 시각 근거 필요)에서 우위가 두드러진다 — 특히 Mem-Gallery의 conflict detection·answer refusal, NExT-QA의 causal/temporal 카테고리.
- Self-evolution의 이득은 벤치마크마다 **다른 실패 유형**(LoCoMo: 시간 정규화, OpenEQA: 씬 disambiguation, Mem-Gallery: 관계/충돌 처리, NExT-QA: 인과-시간 추론)에 집중되어, "파이프라인 자체를 개선"하는 효과임을 시사.
- Agent 평가에서 남은 실패 요인: 세밀한 능동 관측/피드백 부족, VLM의 사람-사물 혼동, 실내-실외 연결 지점에서의 위치 오판.

## Analysis

### Strengths & Significance
- Reasoning-execution-verification을 명시적으로 3분할한 Harness 설계가, digital agent(ReAct/Toolformer류)와 달리 "성공 검증 신호가 없는" embodied 환경의 핵심 난제(procedural drift, 조기 종료)를 직접 타깃한다.
- 메모리를 raw transcript/video가 아닌 **typed, source-grounded graph**로 강제한 설계가 검증 가능성(auditability)과 실패 원인 추적을 동시에 확보 — self-evolution의 "diagnose 가능성" 자체가 이 표현 방식에서 나온다.
- Self-evolution의 no-leakage 보장(split t의 asset은 t+1부터만 적용)이 실험 설계상 매우 엄격하며, 이는 논문 신뢰도를 높이는 요소.
- EmbodiedWorldBench의 4대 설계원칙(executable scenario, 복합 능력 동시 평가, trace-grounded 채점, 재현성)이 기존 벤치마크의 환경 분절·과제 단일화·정적 평가 한계를 정확히 짚는다.

### Limitations
- (저자 인정) 실제 로봇 환경(노이즈 perception, 액추에이션 불확실성, 네트워크 지연, 안전 제약, 이형 embodiment)에서의 대규모 실증이 아직 없음 — 시뮬레이터(UnrealZoo/UE5) 중심.
- (저자 인정) EmbodiedWorldBench는 현재 subset만 평가했고 전체 벤치마크·리더보드는 미공개(future work로 공개 예정).
- (저자 인정) self-evolution이 구조화된 trace와 post-hoc 피드백(벤치마크에서는 ground truth)에 의존 — 실배포에서는 사용자 교정/환경 신호로 대체해야 하는데, 그 신호의 노이즈·희소성에 대한 검증은 없음.
- (분석자) memory 실험은 "동일 hybrid retriever 고정" 하에서만 비교되므로, 그래프 표현 자체의 이득과 특정 retriever 구현의 이득이 완전히 분리되지 않았다.
- (분석자) 33인 저자, 산업 리포트 성격의 논문으로 재현을 위한 코드/데이터 공개 범위가 불명확(Project Page만 언급, public release 시점 미확정).

### Future Work / Improvements
- 저자들이 언급: fine-grained active observation/피드백 정책, 시각 근거 검증, 공간 영역에 대한 추론 강화, EmbodiedWorldBench 전체 공개, 소형모델 파이프라인의 시각 관측/멀티모달 피드백 확장.
- 분석자 제안: self-evolution gate의 τ_gain/τ_reg 민감도 분석, retriever를 변수로 둔 ablation(그래프 구조 자체의 기여도 분리), 실제 로봇 하드웨어에서의 edge-cloud 라우팅 지연/비용 실측.

## References
- 발췌: [source/paper/ABot-AgentOS_..._2026_Alibaba.md](../source/paper/ABot-AgentOS_A_General_Robotic_Agent_OS_with_Lifelong_Multi-modal_Memory_2026_Alibaba.md)
- Project Page: https://amap-cvlab.github.io/ABot-AgentOS
- 비교 대상 memory 프레임워크: Mem0(arXiv:2504.19413), MemGPT(EMNLP 2025), A-MEM(arXiv), GraphVideoAgent(ACM MM 2025) 등 — 논문 References 참조
