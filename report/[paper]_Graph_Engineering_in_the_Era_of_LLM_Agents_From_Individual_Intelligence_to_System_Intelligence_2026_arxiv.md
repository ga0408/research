> [paper] https://arxiv.org/abs/2608.21156

# Graph Engineering in the Era of LLM Agents: From Individual Intelligence to System Intelligence

## Summary & Outline

본 논문은 대규모 언어 모델(LLM) 기반 자율 에이전트 연구가 단일 에이전트의 능력 증강(Individual Intelligence)에서 다중 에이전트 협업 및 시스템 차원의 구조화(System Intelligence)로 진화하는 패러다임 전환을 선언하고, 이를 실현하기 위한 핵심 방법론으로 **그래프 엔지니어링(Graph Engineering)**을 체계화한 63페이지 분량의 종합 서베이 논문이다. 

기존의 프롬프트 엔지니어링(Prompt Engineering), 컨텍스트 엔지니어링(Context Engineering), 하네스 엔지니어링(Harness Engineering), 루프 엔지니어링(Loop Engineering)은 단일 에이전트의 국소적 추론과 실행 역량을 극대화하는 데 기여했으나, 장기 실행(Long-Horizon), 이종 전문성(Heterogeneous Expertise), 병렬 의존성(Interdependent Workflows), 독립적 검증(Independent Verification), 지속적 상태 보존(Persistent State)이 요구되는 복잡한 실세계 과제에서는 본질적인 '개별 지능의 한계'에 직면한다. 논문은 이러한 한계를 극복하기 위해 과제(Task), 행위자(Agent), 런타임 상태(Runtime State) 간의 상호관계를 명시적 그래프 구조로 외재화(Externalize)하고 제어·최적화하는 **그래프 엔지니어링 3대 핵심 축(Task Organization, Agent Coordination, Runtime State Management)**과 **시스템 진화(System Evolution)** 메커니즘을 정식화한다. 나아가 구조적 연결성을 넘어 공유 의미론과 가치 정렬을 제공하는 차세대 패러다임으로 **온톨로지 엔지니어링(Ontology Engineering)**을 제시한다.

```
                    ┌────────────────────────────────────────────────────────┐
                    │                   SYSTEM INTELLIGENCE                  │
                    └───────────────────────────┬────────────────────────────┘
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
┌───────────────────────────────────┐                        ┌───────────────────────────────────┐
│        GRAPH ENGINEERING          │                        │       ONTOLOGY ENGINEERING        │
│    (Structural Substrate)         │                        │     (Shared Semantics & Logic)    │
├───────────────────────────────────┤                        ├───────────────────────────────────┤
│• Task Organization (G_task)       │ ──(Formal Grounding)──►│• Shared Semantic Classes & Axioms │
│• Agent Coordination (G_cap/team)  │                        │• Goal Formation & Value Alignment │
│• State Management (G_state/exec)  │◄──(Rule Constraints)───│• Cross-Framework Interoperability │
│• System Evolution (Graph Mutation)│                        │• System Intelligence Measurement  │
└───────────────────────────────────┘                        └───────────────────────────────────┘
                 ▲
                 │ (Scaffolding & Structural Scaling)
┌────────────────┴──────────────────┐
│       INDIVIDUAL INTELLIGENCE     │
│   Ai = Loop(Fi, Hi; s_i^t)        │
├───────────────────────────────────┤
│• Harness Engineering (Tools/Mem)  │
│• Loop Engineering (StateFlow/Fdbk)│
└───────────────────────────────────┘
                 ▲
                 │ (Elicitation & Augmentation)
┌────────────────┴──────────────────┐
│         MODEL INTELLIGENCE        │
├───────────────────────────────────┤
│• Foundation Models (Pre/Post-train│
│• Prompt & Context Engineering     │
└───────────────────────────────────┘
```

---

## Problem & Motivation

### 연구 배경
인공지능 연구는 기초 모델(Foundation Model)의 매개변수 지능(Model Intelligence)을 활용하여 도구 사용, 장기 기억, 상태 전이를 수행하는 단일 자율 에이전트(Individual Intelligence)로 빠르게 발전해 왔다. 이를 지탱한 4대 선행 엔지니어링 패러다임은 다음과 같다:
1. **Prompt Engineering**: 모델의 잠재 추론 경로를 유도(CoT, ToT, GoT, APE, TextGrad).
2. **Context Engineering**: 동적 검색과 컨텍스트 압축을 통해 유효 정보를 주입(RAG, GraphRAG, MemGPT, ACE).
3. **Harness Engineering**: 도구 인터페이스, 메모리 뱅크, 스킬 라이브러리, 보안 샌드박스를 결합(MCP, SWE-agent, OpenHands, Mem0).
4. **Loop Engineering**: 행동-관찰-피드백-상태 갱신을 반복하는 제어 루프 구성(StateFlow, Magentic-One, AIOS).

그러나 단일 모델의 컨텍스트 윈도우 확장과 하네스 고도화만으로는 소프트웨어 엔지니어링, 과학적 발견, 복합 디지털 자동화 등 현대 산업 과제가 요구하는 거대한 복잡성을 감당할 수 없다.

### 풀고자 하는 문제 (Target Problem)
본 연구가 집중하는 핵심 과제는 **다중 이종 에이전트 시스템 조율(Multi-Agent System Orchestration)**과 **장기 다단계 과제 실행(Long-Horizon Heterogeneous Execution)**이다. 복잡한 목표를 수행하기 위해서는 다수의 상호의존적인 하위 과제를 분해·스케줄링하고, 서로 다른 전문 지식과 권한을 가진 에이전트 팀을 구성하며, 분산 실행 중 발생하는 부분 오류를 격리·복구하고, 누적된 궤적으로부터 시스템 구조 자체를 지속 개선해야 한다.

### 기존 접근의 한계: 개별 지능의 4대 본질적 병목
단일 에이전트 중심의 아키텍처는 다음과 같은 구조적 불일치(Architectural Mismatch)를 겪는다:

| 병목 요인 | 메커니즘 및 증상 | 시스템에 미치는 영향 |
|---|---|---|
| **1. 용량-워크로드 불일치 (Capacity vs. Workload Mismatch)** | 컨텍스트가 길어질수록 주의력 희석(Attention Dilution)과 'Lost-in-the-Middle' 현상 발생, 연산량 및 지연 시간의 2차 곡선형($O(N^2)$) 증가 | 단일 에이전트에 수십 개의 도구와 수만 줄의 코드베이스, 방대한 실행 로그를 동시 주입 시 추론 정밀도 급락 |
| **2. 컨텍스트 오염 및 오류 연쇄 (Context Pollution & Error Cascading)** | 단일 실행 루프 내에서 발생한 하나의 환각(Hallucination)이나 잘못된 도구 반환값이 컨텍스트 히스토리에 영구 보존 | 후속 추론 단계가 오염된 사전 지식에 조건화되어 복구 불가능한 연쇄 실패(Catastrophic Drift) 초래 |
| **3. 전문화 부족과 인지 과부하 (Specialization vs. Cognitive Overload)** | 범용 프롬프트를 통해 단일 모델이 계획자, 도구 실행자, 도메인 전문가, 비평가(Critic) 역할을 동시 수행 | 역할 간 간섭(Role Interference)으로 인해 전반적 추론 성능 저하 및 얕은 의사결정 고착화 |
| **4. 단일 장애점과 독립 검증 부재 (Single Point of Failure & Lack of Verification)** | 단일 에이전트 루프의 런타임 크래시, 무한 루프, 도구 타임아웃 발생 시 복구 경계 없이 전체 태스크 중단 | 자기 자신의 추론 결함을 스스로 비판하는 데 내재적 한계(Self-Preference Bias) 존재 |

```
[단일 에이전트의 컨텍스트 오염 및 실패 연쇄]
Task Start ──► Step 1 (OK) ──► Step 2 (도구 오작동/환각) ──► Step 3 (오염된 상태 조건화) ──► Total Failure
                                         │
                                [컨텍스트 윈도우 오염]
                                [독립 검증 부재 / 롤백 불가]

[그래프 엔지니어링 기반 분산 격리 및 복구]
Task Start ──► Subtask A (Agent 1) ──► Validation Gate ──► Subtask B (Agent 2) ──► Success
                     │                        │ (실패 검출)
                     └──────────────── Causal Rollback & Re-dispatch
```

---

## Contributions

1. **지능 진화 4단계 계층화 (Four-Level Intelligence Hierarchy)**:
   - 인공지능 엔지니어링 패러다임을 **Model Intelligence $\rightarrow$ Individual Intelligence $\rightarrow$ System Intelligence $\rightarrow$ Next-Gen Ontology Intelligence**로 명확히 정립하고, 이를 설명하는 직관적 **시험 비유(Exam Analogy)**를 구축.
2. **시스템 지능(System Intelligence) 및 에이전트 시스템의 수학적 형식화**:
   - 개별 에이전트 $\mathcal{A}_i = \text{Loop}(\mathcal{F}_i, \mathcal{H}_i; s_i^t)$와 에이전트 시스템 $\mathcal{S}_t = \langle \mathcal{A}_t, \mathcal{R}_t, \mathcal{E}_t, \Pi_t, x_t \rangle$을 정의하여, 개별 구성요소의 단순 합산과 시스템 차원의 구조적 조율 간의 본질적 차이를 이론적으로 규명.
3. **그래프 엔지니어링 3대 핵심 기둥 및 시스템 진화 체계화**:
   - **Task Organization** ($\mathcal{G}_{\text{task}}$: Goal Decomposition & Workflow Optimization), **Agent Coordination** ($\mathcal{G}_{\text{cap}}, \mathcal{G}_{\text{team}}, \mathcal{G}_{\text{comm}}$: Capability Modeling, Team Topology, Dynamic Communication), **Runtime State Management** ($\mathcal{G}_{\text{state}}, \mathcal{G}_{\text{exec}}$: State Recording, Fault Localization, Failure Recovery)의 3대 추상화와 크로스 런(Cross-run) 구조 최적화를 위한 **System Evolution**을 정식화.
4. **차세대 온톨로지 엔지니어링(Ontology Engineering) 비전 제시**:
   - 그래프 구조의 위상적 연결성을 넘어, 도메인 불변 제약, TBox/ABox 형식 의미론, 가치 정렬 및 범조직적 상호운용성을 제공하는 온톨로지 기반 차세대 시스템 지능 청사진 제시.
5. **20+ 대표 서베이 비교, 50+ 라이브러리 및 6대 산업 응용 분석**:
   - 선행 서베이들과의 차별성을 8대 평가 축으로 명시하고(Table 4), 현대 에이전트 에코시스템(LangGraph, AutoGen, CrewAI, MetaGPT, AIOS 등) 및 소프트웨어 공학, 과학 발견, 헬스케어, 엔터프라이즈 워크플로 응용을 망라.

---

## Method

### 1. 지능 진화의 전개와 시험 비유 (Exam Analogy)

![From Model Intelligence to System Intelligence](../source/paper/figures/fig1_model_to_system_intelligence.png)

![A Taxonomy of Evolving Techniques in the Era of LLM Agents](../source/paper/figures/fig2_taxonomy_of_evolving_techniques.png)

논문은 LLM 기술의 진화를 개별 수험생이 시험을 치르는 과정에 빗대어 설명한다(Figure 4 참조):
- **Model Intelligence (기초 지능)**: 수험생이 머릿속에 암기하고 있는 기본 지식과 지적 잠재력.
- **Prompt & Context Engineering (조건화)**: 시험지 질문을 명확하게 다듬고(Prompt), 오픈북 시험처럼 관련 참고 자료와 요약본(Context/Cheat-sheet)을 책상에 제공하는 과정.
- **Harness Engineering (도구 보강)**: 계산기, 사전, 그래프 용지 등 외부 보조 도구(Tools, Memory, Sandboxes)를 손에 쥐여주는 과정.
- **Loop Engineering (반복 실행)**: 초안을 작성한 후 다시 읽어보고 오류를 수정하며 정답을 다듬는 자기 반성 루프(Draft $\rightarrow$ Verify $\rightarrow$ Refine).
- **Graph Engineering (팀 협업 및 시스템 조율)**: 단일 수험생의 한계를 넘어, 수학 전문가, 코딩 전문가, 검토자로 구성된 **수험 팀**을 조직하고, 문제 분해 DAG를 작성하여 역할을 배분하며, 중간 풀이 과정을 공유 칠판에 기록하고 오답 발생 시 롤백하는 구조적 협업 체계.
- **Ontology Engineering (표준 의미론 및 가치 정렬)**: 서로 다른 팀과 연구실 간에 수학 기호, 과학적 정의, 윤리적 기준, 검증 규칙을 통일하여 범용적으로 소통하고 검증하는 표준 의미 체계.

![An Illustrative Conceptualization of System Intelligence](../source/paper/figures/fig4_exam_analogy_progression.png)

---

### 2. 개별 지능과 에이전트 시스템의 수학적 정식화

![From Model Intelligence to Individual Intelligence](../source/paper/figures/fig3_model_to_individual_intelligence.png)

#### 2.1 개별 에이전트 (Individual Agent)
개별 에이전트는 환경을 인식하고, 결정을 내리며, 행동을 취하고, 피드백에 따라 적응하는 자율적 계산 단위이다:

$$\mathcal{A}_i = \text{Loop}\left(\mathcal{F}_i, \mathcal{H}_i; s_i^t\right)$$

- $\mathcal{F}_i$: 인지 핵심(Cognitive Core) 역할을 수행하는 Foundation Model.
- $\mathcal{H}_i$: 모델의 내재적 역량을 확장하는 Agent Harness (인식, 컨텍스트 조립, 메모리 접근, 도구 호출, 스킬 합성, 런타임 제어 인터페이스).
- $s_i^t$: 시간 $t$에서의 에이전트 $i$의 국소 런타임 상태(Local Runtime State).
- $\text{Loop}$: 지각(Perception) $\rightarrow$ 추론(Reasoning) $\rightarrow$ 행동(Action) $\rightarrow$ 피드백 처리(Feedback) $\rightarrow$ 상태 갱신(State Update)을 반복하는 실행 제어기.

#### 2.2 에이전트 시스템 (Agent System)
에이전트 시스템은 공유 자원, 외부 환경, 조율 메커니즘을 통해 상호작용하는 다중 에이전트의 결합체이다:

$$\mathcal{S}_t = \left\langle \mathcal{A}_t, \mathcal{R}_t, \mathcal{E}_t, \Pi_t, x_t \right\rangle$$

- $\mathcal{A}_t = \{\mathcal{A}_1, \dots, \mathcal{A}_n\}$: 에이전트 팀(Agent Team). 각 에이전트는 독립된 $\mathcal{F}_i, \mathcal{H}_i, s_i^t$를 보유.
- $\mathcal{R}_t$: 공유 자원(Shared Resources: 도구 레지스트리, 공용 메모리/KB, 독립 검증기, 인간 개입 채널).
- $\mathcal{E}_t$: 외부 환경(External Environment: 관측값 제공 및 행동 효과 반영).
- $\Pi_t$: 조율 메커니즘(Coordination Mechanisms: 과제 할당 규칙, 메시지 라우팅, 합의 알고리즘, 오류 격리 정책).
- $x_t$: 글로벌 시스템 상태(Global System State: 전체 태스크 진행도, 산출물 버전, 자원 잠금, 실패 이력).

---

### 3. 그래프 엔지니어링 아키텍처: 핵심 3대 기둥

![Overview of Graph Engineering](../source/paper/figures/fig5_graph_engineering_overview.png)

그래프 엔지니어링은 복잡한 시스템 수준의 상호관계를 명시적 그래프 객체로 모델링하여 제어하는 구조 중심 인프라이다.

```
                    ┌──────────────────────────────────────────────────────────┐
                    │                 GRAPH ENGINEERING TRIAD                  │
                    └─────────────────────────────┬────────────────────────────┘
                                                  │
         ┌────────────────────────────────────────┼────────────────────────────────────────┐
         ▼                                        ▼                                        ▼
┌─────────────────────────────────┐      ┌─────────────────────────────────┐      ┌─────────────────────────────────┐
│       Task Organization         │      │       Agent Coordination        │      │    Runtime State Management     │
│       G_task = (V_t, E_t)       │      │   G_cap, G_team, G_comm         │      │       G_state, G_exec           │
├─────────────────────────────────┤      ├─────────────────────────────────┤      ├─────────────────────────────────┤
│• Goal Decomposition             │      │• Agent Capability Modeling      │      │• State Recording                │
│  - Semantic Subgoal Parsing     │      │  - Skill/Tool/Permission Nodes  │      │  - Global Event Stream          │
│  - Dependency DAG Construction  │      │  - Reliability & Cost Attributes│      │  - Artifact Lineage DAG         │
│• Workflow Optimization          │      │• Agent Team Organization        │      │• Fault Localization             │
│  - Critical Path Scheduling     │      │  - Hierarchical / Mesh Topology │      │  - Causal Dependency Tracing    │
│  - Parallel Branch Dispatch     │      │  - Authority & Review Gates     │      │  - Anomaly Root-Cause Audit     │
│  - Dynamic Path Pruning         │      │• Multi-Agent Communication      │      │• Failure Recovery               │
│  - Topological Compilation      │      │  - Bandwidth Control            │      │  - Checkpoint & State Rollback  │
│    (AFlow, GPTSwarm, LLMCompiler│      │  - Semantic Message Filtering   │      │  - Subgraph Re-execution        │
│     ReWOO, HuggingGPT)          │      │  - Topology Pruning (DyLAN,     │      │    (LangGraph, Burr, MemTX,     │
│                                 │      │    AgentPrune, ChatDev, MetaGPT)│      │     SagaLLM, PatchBoard)        │
└────────────────┬────────────────┘      └────────────────┬────────────────┘      └────────────────┬────────────────┘
                 │                                        │                                        │
                 └────────────────────────────────────────┼────────────────────────────────────────┘
                                                          ▼
                                    ┌───────────────────────────────────────────┐
                                    │             System Evolution              │
                                    │    Cross-Run Graph Rewriting & Plasticity │
                                    │ (EvoFlow, CARD, TDAG, DynTaskMAS, Flow)   │
                                    └───────────────────────────────────────────┘
```

---

#### 3.1 Task Organization (과제 조직화: 무엇을 할 것인가)

![Overview of Task Organization](../source/paper/figures/fig6_task_organization.png)

Task Organization은 비구조화된 고수준 목표를 실행 가능하고 스케줄링 가능한 작업 그래프 $\mathcal{G}_{\text{task}} = (\mathcal{V}_{\text{task}}, \mathcal{E}_{\text{task}})$로 변환한다.

1. **목표 분해 (Goal Decomposition)**:
   - 복잡한 사용자 의도를 원자적 하위 과제 노드 $\mathcal{V}_{\text{task}}$로 분해하고, 데이터 의존성 및 선후 관계를 방향성 엣지 $\mathcal{E}_{\text{task}}$로 연결.
   - 대표 사례: HuggingGPT(멀티모달 모델 라우팅), ReWOO(추론과 실행의 분리 및 변수 참조 체계), TDAG(실행 중 동적 트리 확장).
2. **워크플로 최적화 (Workflow Optimization)**:
   - 생성된 작업 그래프를 정적 컴파일하거나 런타임 피드백에 따라 동적으로 최적화.
   - **위상 정렬 및 임계 경로 분석**: 독립적인 하위 브랜치를 병렬 디스패치하고(LLMCompiler), 병목 경로를 우선 할당.
   - **구조적 토폴로지 탐색**: 유전 알고리즘이나 MCTS를 활용하여 최적의 에이전트 실행 흐름 그래프를 탐색(AFlow, GPTSwarm, EvoFlow, MermaidFlow).

---

#### 3.2 Agent Coordination (에이전트 조율: 누가 일할 것인가)

![Overview of Agent Coordination](../source/paper/figures/fig7_agent_coordination.png)

Agent Coordination은 이종 전문성을 가진 다중 에이전트의 역량, 팀 구조, 통신 채널을 그래프로 구조화하여 조율한다.

1. **에이전트 역량 모델링 ($\mathcal{G}_{\text{cap}}$)**:
   - 에이전트, 스킬, 도구, 자원 권한을 노드로 정의하고, 소유·접근·신뢰도 관계를 엣지로 매핑(DyLAN, SkillGraph, MasRouter).
   - 작업 요구사항과 에이전트 역량 간의 최적 매칭을 수행하며, 자원 결손 시 대체 에이전트를 동적 재할당.
2. **에이전트 팀 조직화 ($\mathcal{G}_{\text{team}}$)**:
   - 계층형(Hierarchical: 관리자-작업자), 탈중앙 메시형(Decentralized Mesh: 수평적 피어 협업), 위원회/투표형(Committee/Voting) 등 협업 토폴로지 구성(MetaGPT, ChatDev, Magentic-One, SwarmAgentic).
   - 권한 경계와 중간 산출물 리뷰 게이트(Review Gates)를 강제하여 역할 간섭을 방지.
3. **다중 에이전트 통신 조율 ($\mathcal{G}_{\text{comm}}$)**:
   - 모든 에이전트가 무제한 메시지를 교환하는 $O(N^2)$ 통신 오버헤드와 정보 범람(Context Flooding)을 차단.
   - 동적 통신 그래프 가지치기(AgentPrune, AgentDropout, DyTopo)를 통해 필수적인 데이터 교환 채널만 활성화하고 의미론적 필터링 적용.

---

#### 3.3 Runtime State Management (런타임 상태 관리: 시스템이 어떻게 작동하는가)

![Overview of Runtime State Management](../source/paper/figures/fig8_runtime_state_management.png)

분산된 다중 에이전트 환경에서 "무슨 일이 일어났는가", "무엇이 잘못되었는가", "어떻게 안전하게 재개할 것인가"를 보증하는 런타임 인프라이다.

1. **상태 기록 (State Recording)**:
   - 개별 에이전트의 로컬 로그를 넘어, 시스템 전체의 이벤트 스트림, 산출물 버전, 외부 도구 사이드이펙트를 추적하는 글로벌 상태 그래프 $\mathcal{G}_{\text{state}}$ 유지(LangGraph, Burr, MemTX, SagaLLM, PatchBoard, AgentGit).
   - 엄격한 상태 업데이트 게이트(Schema Check, Permission Check, Invariant Check, Conflict Check)를 두어 공유 상태의 일관성 보장.
2. **결함 국소화 (Fault Localization)**:
   - 실행 실패 발생 시 원인과 결과를 추적할 수 있는 인과 그래프(Causal Execution Graph)를 기반으로 장애의 근본 원인(Root Cause) 진단(CausalFlow, ReflexGrad, TDAD, MAST).
   - 상위 요구사항 정의 오류인지, 코드 생성 논리 결함인지, 환경 테스트 실패인지를 명확히 분리.
3. **장애 복구 (Failure Recovery)**:
   - 전체 시스템을 처음부터 재시작하지 않고, 검증된 체크포인트(Recovery Boundary)로 롤백하거나 실패한 하위 서브그래프만 동적으로 재계획·재실행(Atomix, SagaLLM, Cordon).

---

#### 3.4 System Evolution (시스템 진화: 경험을 통한 지속적 구조 개선)

단일 실행에서 얻은 피드백과 오류 증거를 바탕으로 시스템 그래프를 크로스 런(Cross-run) 차원에서 지속 최적화한다:
- **과제 구조 진화 ($\Delta \mathcal{G}_{\text{task}}$)**: 자주 반복되는 성공적인 하위 태스크 묶음을 템플릿화하거나 비효율적인 분해 패턴을 재작성(EvoFlow, Flow, DynTaskMAS).
- **역량 및 팀 구조 진화 ($\Delta \mathcal{G}_{\text{cap}}, \Delta \mathcal{G}_{\text{team}}$)**: 에이전트의 과거 성공률에 따라 기여도 가중치를 갱신하고, 비효율적인 통신 링크를 영구 제거(CARD, Meta-Team, DyTopo).
- **상태 및 스킬 자산 축적**: 검증된 서브그래프 실행 결과를 모듈식 스킬(SkillDAG, Graph of Skills)로 승격하여 시스템 메모리에 영속화.

---

### 4. 차세대 패러다임: 온톨로지 엔지니어링 (Ontology Engineering)

그래프 엔지니어링은 노드와 엣지의 **위상적 연결성(Topology)**을 효과적으로 제어하지만, 노드 라벨의 엄밀한 의미, 복합 도메인 규칙, 다중 조직 간 상호운용성, 시스템 자체의 목표 형성 메커니즘을 정의하는 데는 한계가 있다. 이를 해결하기 위해 논문은 지식공학(Knowledge Engineering)의 온톨로지 개념을 접목한 **Ontology Engineering**을 제시한다.

| 비교 차원 | Graph Engineering (현행) | Ontology Engineering (차세대) |
|---|---|---|
| **핵심 초점** | 구조적 연결성, 실행 라우팅, 런타임 제어 | 정형 의미론(Formal Semantics), 공리(Axioms), 가치 정렬 |
| **개념 표현** | 임의 텍스트 라벨, 태스크 식별자, 데이터 참조 | TBox(개념·공리 스키마) / ABox(인스턴스 사실), OWL/RDF 표준 |
| **제약 모델** | 그래프 토폴로지 (DAG, 상태 머신, 트리) | 1차 논리 규칙, 불변성 공리, 도메인 무결성 제약 |
| **시스템 경계** | 단일 프레임워크 / 격리된 실행 그래프 | 이종 에이전트 프레임워크 및 기업 시스템 간 상호운용성 |
| **목표 형성** | 사용자 프롬프트 기반 분해에 의존 | 조직 가치와 규범적 기준에 부합하는 자율적 목표 형성 |

- **목표 형성과 가치 정렬 (Goal Formation & Value Alignment)**: 시스템이 해결해야 할 목표가 기업의 정책, 윤리적 기준, 안전 공리에 부합하는지 정형 규칙으로 검증.
- **공유 의미론과 세계 접지 (Shared Semantics & World Grounding)**: 서로 다른 에이전트와 도구들이 동일한 엔티티와 동작에 대해 오차 없는 공통 정의를 공유(OntoCodex, CoA-Text2OWL, AgentO, Palantir Ontology).
- **시스템 지능의 객관적 측정 (Measuring System Intelligence)**: 기초 모델의 연산량이나 파라미터 크기에 의한 성능 향상과 시스템 조직화 구조에 의한 순수 기여도를 분리하여 정량화.

상세 기술 발췌 원문 → [excerpt](../source/paper/Graph_Engineering_in_the_Era_of_LLM_Agents_From_Individual_Intelligence_to_System_Intelligence_2026_arxiv.md)

---

## Experiments, Benchmarks & Ecosystem

### 1. 지능 수준별 평가 벤치마크 (Table 1 기반 분류)

| 지능 수준 | 주요 평가 영역 | 대표 벤치마크 및 데이터셋 | 주요 특징 및 평가 척도 |
|---|---|---|---|
| **Model Intelligence** | 기초 언어 이해, 지식 회상, 단일 단계 추론 | MMLU, GSM8K, MATH, HumanEval, ARC | 파라미터 내재 지식 및 단일 프롬프트 완결형 정확도(Accuracy, Pass@1) |
| **Individual Intelligence** | 도구 호출, 단일 에이전트 환경 상호작용, 장기 실행 | ToolBench, WebArena, OSWorld, GAIA, SWE-bench Verified, Terminal-Bench 2.0 | 환경 피드백 처리, 도구 인자 정확도, 단일 에이전트 태스크 완료율 |
| **System Intelligence** | 다중 에이전트 협업, 역할 분담, 통신 효율, 결함 복구 | AgentBench, Collaborative Gym, MAS-Bench, Mind2Web-Multi, ITBench, LoopBench | 이종 에이전트 조율 성공률, 통신 토큰 오버헤드, 결함 격리 및 롤백 회복력 |

---

### 2. 오픈소스 라이브러리 및 엔지니어링 에코시스템 (Table 2 요약)

```
[System Layer]    LangGraph ── AutoGen ── CrewAI ── MetaGPT ── ChatDev ── AIOS
                      │           │          │         │          │        │
[Runtime/State]   Burr ────── MemTX ──── SagaLLM ── PatchBoard ── PydanticAI
                      │           │          │         │          │
[Model/Harness]   vLLM ────── SGLang ─── verl ───── OpenHands ── SWE-agent
```

- **시스템 지능 오케스트레이션**:
  - **LangGraph**: 명시적 상태 그래프(StateGraph)와 순환 엣지, 체크포인트 영속성을 제공하는 표준 런타임.
  - **AutoGen / CrewAI**: 다중 에이전트 대화 및 역할 기반 태스크 위임 프레임워크.
  - **MetaGPT / ChatDev**: 소프트웨어 개발 SOP(표준 운영 절차)를 구조화된 에이전트 팀으로 구현.
  - **AIOS**: LLM 에이전트를 위한 커널 수준의 스케줄링, 메모리, 도구 접근 제어 OS.
- **런타임 상태 및 트랜잭션 엔진**:
  - **Burr / PydanticAI**: 상태 머신 제어 및 엄격한 타입 검증 런타임.
  - **SagaLLM / MemTX / PatchBoard**: 분산 에이전트 트랜잭션 보장, 롤백 및 충돌 방지 게이트.

---

### 3. 실세계 산업 응용 분야 (Table 3 분석)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GRAPH ENGINEERING APPLICATION DOMAINS                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Software Engineering & IT Ops │ MetaGPT, SWE-agent, OpenHands, Cline,    │
│                                  │ Codex, Claude Code, Project ALICE        │
├──────────────────────────────────┼──────────────────────────────────────────┤
│ 2. Scientific Discovery          │ SciAgents, The AI Scientist, Virtual Lab,│
│                                  │ Co-Scientist, Robin                      │
├──────────────────────────────────┼──────────────────────────────────────────┤
│ 3. Healthcare & Clinical Support │ Multi-Agent Consultation (MAC), MedAgent │
├──────────────────────────────────┼──────────────────────────────────────────┤
│ 4. Enterprise Digital Automation │ OpenClaw, Hermes Agent, Enterprise Graph │
├──────────────────────────────────┼──────────────────────────────────────────┤
│ 5. Social & Economic Simulation  │ AgentSociety, EconAgent, TwinMarket      │
└──────────────────────────────────┘
```

1. **소프트웨어 공학 및 IT 운영**:
   - 요구사항 분석, 아키텍처 설계, 코드 작성, 단위 테스트, 코드 리뷰를 전문화된 에이전트 팀에 분배하고 Git 작업 트리와 연계(MetaGPT, OpenHands, Codex, Project ALICE).
2. **과학적 발견 및 실험실 자동화**:
   - 가설 생성 $\rightarrow$ 문헌 검토 $\rightarrow$ 실험 설계 $\rightarrow$ 로봇 실험실 물리 실행 $\rightarrow$ 데이터 검증의 전주기를 다중 에이전트 루프로 연결(SciAgents, The AI Scientist, The Virtual Lab, Co-Scientist).
3. **헬스케어 및 임상 의사결정**:
   - 다학제 진료팀(내과, 영상의학과, 병리과 등)을 에이전트 팀으로 모사하여 진단 오류를 줄이고 근거 추적성을 확보.
4. **개인 및 기업 디지털 자동화**:
   - 다채널 게이트웨이와 영속적 스킬 진화를 결합한 시스템(Hermes Agent, OpenClaw).
5. **사회 및 거시경제 시뮬레이션**:
   - 대규모 이종 에이전트 집단의 상호작용을 통해 시장 동학 및 정책 효과 시뮬레이션(AgentSociety, EconAgent, TwinMarket).

---

## Analysis

### Strengths & Significance
1. **패러다임의 명확한 구조화 및 이론적 정립**:
   - 단편적으로 분산되어 있던 프롬프트, RAG, 하네스, 멀티에이전트 연구들을 **Model $\rightarrow$ Individual $\rightarrow$ System $\rightarrow$ Ontology Intelligence**라는 거대한 지능 진화의 틀 안에서 일관되게 정렬함.
2. **인과적 추적성 및 회복 탄력성(Resilience) 중심의 설계**:
   - 기존 멀티에이전트 연구가 단순 대화(Chat)에 머물렀던 것과 달리, 상태 그래프($\mathcal{G}_{\text{state}}$), 인과적 장애 진단, 서브그래프 롤백 등 소프트웨어 공학적 신뢰성 요건을 1등 시민(First-class citizen)으로 격상시킴.
3. **체계적 비교 분석 및 방대한 문헌 포괄**:
   - 498편의 최신 문헌과 20여 편의 선행 서베이를 비교 분석하여, 그래프가 단순 데이터 구조가 아닌 '시스템 조직의 핵심 기저(Organizational Substrate)'임을 명확히 논증함.

### Limitations
1. **사전 정의된 정적 토폴로지 의존성**:
   - 현재 대다수의 실용적 에이전트 시스템(LangGraph, MetaGPT 등)은 인간 개발자가 설계한 정적 DAG 또는 상태 머신에 크게 의존하며, 진정한 의미의 자율적 토폴로지 진화(System Evolution)는 초기 연구 단계에 머물러 있음.
2. **그래프 간 결합 진화의 복잡성 (Coupled Cross-Graph Evolution)**:
   - 과제 그래프($\mathcal{G}_{\text{task}}$)의 변경이 에이전트 팀($\mathcal{G}_{\text{team}}$) 및 통신($\mathcal{G}_{\text{comm}}$) 요구사항을 변경시키고, 이로 인해 상태 관리의 불변성이 깨지는 상호 결합 문제에 대한 수학적 수렴성 보장이 부족함.
3. **높은 시스템 조율 오버헤드**:
   - 복잡한 그래프 검증 게이트, 분산 체크포인팅, 다중 에이전트 라우팅으로 인해 단순 단일 에이전트 대비 인프라 복잡도와 지연 시간(Latency)이 증가할 수 있음.

### Future Work / Improvements
1. **Graph-Native Agent Operating System (그래프 네이티브 에이전트 OS)**:
   - 개별 프레임워크마다 분편화된 스케줄러, 메모리, 권한 체계를 통합하여 태스크, 에이전트, 상태를 기본 커널 객체로 관리하는 그래프 네이티브 OS 구축.
2. **온톨로지 기반 정형 검증 (Formal Verification with Ontologies)**:
   - LLM 에이전트의 생성물이 도메인 공리 및 안전 규칙을 위반하지 않는지 TBox/ABox 추론 엔진과 실시간 연계하는 하이브리드 뉴로-심볼릭 검증 체계 구현.
3. **안전한 크로스 런 자율 진화 프로토콜 (Safe Cross-Run Evolution)**:
   - 시스템 구조 변경 시 안전성을 자동으로 평가하고 롤백할 수 있는 형상 관리 및 리그레션 테스트 프레임워크 개발.

---

## References

- **원문 논문**: [arXiv:2608.21156](https://arxiv.org/abs/2608.21156) — *Graph Engineering in the Era of LLM Agents: From Individual Intelligence to System Intelligence* (Feng et al., 2026)
- **오픈소스 프로젝트**: [Awesome-Graph-Engineering (DEEP-JLU)](https://github.com/DEEP-JLU/Awesome-Graph-Engineering)
- **핵심 발췌 파일**: [Graph_Engineering_..._2026_arxiv.md](../source/paper/Graph_Engineering_in_the_Era_of_LLM_Agents_From_Individual_Intelligence_to_System_Intelligence_2026_arxiv.md)
- **관련 저장소 내 연구 문서**:
  - [Hermes Agent 분석]([git]_hermes-agent_NousResearch.md) — 자기개선형 개인 AI 에이전트 루프 및 스킬 진화
  - [OpenClaw 분석]([git]_openclaw_openclaw.md) — 멀티채널 게이트웨이 및 임베디드 하이브리드 메모리 시스템
  - [ABot-AgentOS 분석]([paper]_ABot-AgentOS_A_General_Robotic_Agent_OS_with_Lifelong_Multi-modal_Memory_2026_Alibaba.md) — 다중 모달 그래프 메모리 및 평생 자기진화 로봇 에이전트 OS
  - [Self-Improvements in Modern Agentic Systems 분석]([paper]_Self-Improvements_in_Modern_Agentic_Systems_A_Survey_2026_arxiv.md) — 에이전트 자기개선 전주기 서베이
  - [ReflectWorld-MM 분석]([paper][git]_ReflectWorld-MM_An_Entity-Oriented_Multimodal_Memory_System_for_Open-Ended_Video_Streams_2026_Rightly_Robotics.md) — 엔티티 중심 계층형 멀티모달 메모리 시스템
