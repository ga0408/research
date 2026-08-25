# Graph Engineering in the Era of LLM Agents: From Individual Intelligence to System Intelligence — 핵심 발췌

> 출처: [분석 문서](../../report/[paper]_Graph_Engineering_in_the_Era_of_LLM_Agents_From_Individual_Intelligence_to_System_Intelligence_2026_arxiv.md) / 원본: [arXiv:2608.21156](https://arxiv.org/abs/2608.21156)

---

## 1. Preliminaries & Formal Definitions

### 1.1 Individual Agent
An Individual Agent is an autonomous computational entity that perceives its environment, makes decisions, executes actions, and adapts its behavior according to feedback. It consists primarily of a Foundation Model $F_i$, an Agent Harness $H_i$, and an Agent Loop:

$$A_i = \text{Loop}\left(F_i, H_i; s_i^t\right)$$

- $F_i$: Foundation Model serving as the cognitive core (language understanding, reasoning, planning, code/content generation).
- $H_i$: Agent Harness extending intrinsic capabilities through perception, context construction, memory/knowledge access, tool invocation, reusable skills, and runtime governance.
- $s_i^t$: Local runtime state of agent $i$ at time $t$.
- $\text{Loop}$: Iterative execution cycle governing continuous perception $\rightarrow$ reasoning $\rightarrow$ action $\rightarrow$ feedback processing $\rightarrow$ state update.

### 1.2 Agent System & System Intelligence
An Agent System extends the Individual Agent abstraction to a collection of agents operating via shared resources, environments, and coordination mechanisms at time $t$:

$$S_t = \left\langle A_t, R_t, E_t, \Pi_t, x_t \right\rangle$$

- $A_t = \{A_1, \dots, A_n\}$: Agent Team, where each agent possesses its own model, harness, loop, and local state.
- $R_t$: Shared Resources (tools, model endpoints, shared memory/KBs, verifiers, human-in-the-loop).
- $E_t$: External Environment providing observations and receiving actions.
- $\Pi_t$: Coordination Mechanisms governing task assignment, communication routing, consensus, and fault handling.
- $x_t$: Global System State describing system-wide task progress, artifact states, resource locks, and failure records.

**Definition of System Intelligence:**
> *"System Intelligence is the ability of an agent system $S_t$ to organize and coordinate multiple intelligent components into a coherent, adaptive whole that pursues a shared objective across changing conditions, heterogeneous expertise, parallel branches, and partial failures."*

---

## 2. The Four Precursor Engineering Paradigms

```
[ Foundation Model ]  ──(Prompt & Context Eng.)──►  [ Model Intelligence ]
         │
         ▼
[ Harness & Loop Eng. ] ──(Scaffolding & Loops)──►  [ Individual Intelligence ]
         │
         ▼
[ Graph Engineering ]  ──(Task/Team/State Graphs)──► [ System Intelligence ]
         │
         ▼
[ Ontology Engineering] ──(Shared Semantics/Rules)─► [ Next-Gen System Intelligence ]
```

1. **Prompt Engineering**: Elicits latent capabilities within model parameters through instruction tuning, in-context demonstrations, reasoning scaffolds (CoT, ToT, GoT), and automated optimization (APE, OPRO, Promptbreeder, TextGrad).
2. **Context Engineering**: Manages dynamic information access into the context window via dense retrieval (DPR), RAG, GraphRAG, context compression (LLMLingua), dynamic memory (MemGPT), and selective context curation (ACE, AdaCoM).
3. **Harness Engineering**: Wraps the model with external scaffolding—standardized tool interfaces (ReAct, MCP, CodeAct, OpenHands), structured memory banks (A-MEM, Mem0, Zep), reusable skill libraries (Voyager, CRAFT, HDSO), and security sandboxes/governance (ToolSandbox, CaMeL, Living-Harness).
4. **Loop Engineering**: Implements iterative feedback-driven execution architectures (StateFlow, Magentic-One, AIOS, ResearchLoop), communication protocols (Internet of Agents, Sovereign Loops, "The Log is the Agent"), and execution feedback hooks (CRITIC, LEVER, OSWorld).

---

## 3. Fundamental Limitations of Individual Intelligence & Core System Requirements

### 3.1 Five Inherent Requirements of Complex Tasks
1. **Long-Horizon Execution**: Multi-step reasoning across extended time horizons without context saturation.
2. **Heterogeneous Expertise**: Task specialization across distinct domain models, tool permissions, and prompt roles without role interference.
3. **Parallel & Interdependent Dependencies**: Concurrent branch execution and topological dataflow aggregation without serial bottlenecking.
4. **Independent Verification**: Architectural separation of creator and auditor to eliminate self-preference bias and confirmation bias.
5. **Persistent State Management**: Durable global provenance and state tracking beyond volatile context windows for transactional rollback.

### 3.2 Four Architectural Mismatches in Single-Agent Loops
1. **Capacity vs. Workload Mismatch**: As real-world tasks scale in duration and complexity, the volume of required context, tool definitions, intermediate artifacts, and verification traces exceeds the effective attention span and memory budget of any single model (attention dilution, lost-in-the-middle, quadratic compute/latency overhead).
2. **Context Pollution & Error Cascading**: In a single execution loop, erroneous intermediate outputs, invalid tool responses, or hallucinations contaminate the single context history. Subsequent reasoning steps condition on corrupted priors, causing irrevocable cascading failure.
3. **Specialization vs. Cognitive Overload**: General-purpose prompting forces one agent to simultaneously act as planner, domain specialist, tool operator, critic, and verifier. Role interference degrades performance compared to specialized agents with focused system prompts, isolated tools, and strict task boundaries.
4. **Single Point of Failure & Lack of Independent Verification**: A single agent cannot reliably audit its own internal reasoning flaws. A timeout, tool crash, or infinite loop in a monolithic harness halts the entire task with no recovery boundary.

---

## 4. Graph Engineering Architecture: Mathematical Formalization

```
                ┌──────────────────────────────────────────────┐
                │             GRAPH ENGINEERING                │
                └──────────────────────┬───────────────────────┘
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         ▼                             ▼                             ▼
┌──────────────────┐         ┌──────────────────┐         ┌──────────────────┐
│ Task Organization│         │Agent Coordination│         │State Management  │
│ G_task = (V, E)  │         │G_cap, G_team, ...│         │G_state, G_exec   │
├──────────────────┤         ├──────────────────┤         ├──────────────────┤
│• Goal Decomp.    │         │• Capability Graph│         │• State Recording │
│  (Tree/DAG/Hyper)│         │  (Skill/Resource)│         │  (Trace/Lineage) │
│• Workflow Optim. │         │• Team Topology   │         │• Fault Localize  │
│  (Schedule/Path) │         │  (Hierarchy/Mesh)│         │  (Root-cause DAG)│
│• Dynamic Pruning │         │• Comm. Routing   │         │• Failure Recovery│
│  (AFlow/GPTSwarm)│         │  (Filter/Bottle.)│         │  (Rollback/Tx)   │
└────────┬─────────┘         └────────┬─────────┘         └────────┬─────────┘
         │                            │                            │
         └────────────────────────────┼────────────────────────────┘
                                      ▼
                        ┌───────────────────────────┐
                        │     System Evolution      │
                        │ Cross-run Graph Rewriting │
                        │  (EvoFlow, CARD, TDAG)    │
                        └───────────────────────────┘
```

### 4.1 Task Organization ($G_{\text{task}}$)
과제 조직화는 비구조적 목표를 방향성 비순환 그래프(DAG) 또는 하이퍼그래프 $G_{\text{task}}$로 구조화하여 스케줄링 및 실행 흐름을 제어한다:

$$G_{\text{task}} = \left\langle V_{\text{task}}, E_{\text{task}}, \Phi_{\text{task}}, \Sigma_{\text{task}} \right\rangle$$

- $V_{\text{task}} = \{v_1, \dots, v_m\}$: 원자적 하위 과제(Subtask/Subgoal) 노드 집합. 각 노드 $v_k = \langle \text{spec}_k, \text{input}_k, \text{output}_k \rangle$.
- $E_{\text{task}} \subseteq V_{\text{task}} \times V_{\text{task}} \times T_{\text{dep}}$: 타입화된 의존성 엣지 집합:
  - **데이터 흐름 의존성 (Dataflow)**: $u \xrightarrow{\text{data}} v \iff \text{output}(u) \subseteq \text{input}(v)$
  - **선행 제약 (Precedence)**: $u \prec v \iff t_{\text{start}}(v) \ge t_{\text{finish}}(u)$
  - **조건부 분기 (Conditional)**: $u \xrightarrow{\text{cond}(c)} v$
  - **검증/승인 게이트 (Review Gate)**: $u \xrightarrow{\text{verify}} v$
- $\Sigma_{\text{task}}(v) \in \{\text{Pending}, \text{Ready}, \text{Running}, \text{Blocked}, \text{Committed}, \text{Failed}\}$: 런타임 노드 실행 상태.
- **Ready 상태 전이 조건**:
  $$\Sigma_{\text{task}}(v) \leftarrow \text{Ready} \iff \forall u \in \text{Parents}(v), \Sigma_{\text{task}}(u) = \text{Committed}$$
- **워크플로 최적화 (Workflow Optimization as Structural Search)**:
  $$G_{\text{task}}^* = \arg\max_{G \in \Omega(\text{Goal})} \mathbb{E}_{\tau \sim G}\left[ R(\tau) - \lambda_1 \cdot \text{Cost}(\tau) - \lambda_2 \cdot \text{Latency}(\tau) \right]$$
- **임계 경로 메이크스팬 (Critical Path Makespan)**:
  $$T_{\text{makespan}}(G_{\text{task}}) = \max_{p \in \text{Paths}(G_{\text{task}})} \sum_{v \in p} \text{Duration}(v)$$

---

### 4.2 Agent Coordination ($G_{\text{cap}}, G_{\text{team}}, G_{\text{comm}}^t$)
에이전트 조율은 이종 에이전트의 역량 매핑, 팀 위계, 동적 통신 채널을 3대 그래프로 정식화한다:

#### (1) 에이전트 역량 이분 그래프 (Agent Capability Bipartite Graph $G_{\text{cap}}$)
$$G_{\text{cap}} = \left\langle V_{\text{agent}} \cup V_{\text{res}}, E_{\text{cap}}, W_{\text{cap}} \right\rangle$$

- $V_{\text{agent}} = \{A_1, \dots, A_n\}$: 에이전트 집합.
- $V_{\text{res}} = K_{\text{skills}} \cup T_{\text{tools}} \cup D_{\text{data}}$: 자원(스킬, 도구, 데이터베이스) 노드 집합.
- $E_{\text{cap}} \subseteq V_{\text{agent}} \times V_{\text{res}}$: 소유 및 접근 권한 엣지.
- $W_{\text{cap}}(A_i, r) = \langle \text{proficiency}_{ir}, \text{permission}_{ir}, \text{reliability}_{ir} \rangle$: 역량 가중치 튜플.
- **에이전트-과제 최적 할당 함수 ($\mu^*: V_{\text{task}} \to V_{\text{agent}}$)**:
  $$\mu^*(v) = \arg\max_{A_i \in V_{\text{agent}}} \left[ \text{Match}\left(W_{\text{cap}}(A_i), \text{Req}(v)\right) \cdot \text{Avail}(A_i, t) \right]$$

#### (2) 에이전트 팀 조직 그래프 ($G_{\text{team}}$)
$$G_{\text{team}} = \left\langle V_{\text{agent}}, E_{\text{team}}, \text{Role} \right\rangle$$

- $E_{\text{team}}$: 위계 및 보고 관계($A_{\text{lead}} \xrightarrow{\text{supervise}} A_{\text{sub}}$), 피어 협업($A_i \xleftrightarrow{\text{peer}} A_j$), 독립 검토 게이트($A_{\text{worker}} \xrightarrow{\text{submit}} A_{\text{reviewer}}$).

#### (3) 동적 통신 그래프 ($G_{\text{comm}}^t$)
$$G_{\text{comm}}^t = \left\langle V_{\text{agent}}, E_{\text{comm}}^t, M^t \right\rangle$$

- **동적 통신 가지치기 (Communication Sparsification)**: $O(N^2)$ 메시지 범람을 차단하고 관련성 기반 활성 채널만 유지:
  $$E_{\text{comm}}^t = \left\{ (i, j) \in V_{\text{agent}}^2 \;\middle|\; \text{Score}\left(m_{i \to j}^t, \text{Context}_j^t\right) \ge \tau_{\text{comm}} \land \text{Perm}(i \to j) = 1 \right\}$$
- $m_{i \to j}^t$: 구조화된 교환 메시지 $\langle \text{Sender}, \text{Receiver}, \text{Type}, \text{Payload}, \text{ArtifactID}, t \rangle$.

---

### 4.3 Runtime State Management ($G_{\text{state}}^t, G_{\text{exec}}^t$)
분산 런타임 환경에서 글로벌 상태의 일관성, 인과적 결함 격리 및 부분 롤백을 제어한다:

#### (1) 글로벌 상태 & 인과 실행 그래프 ($G_{\text{state}}^t, G_{\text{exec}}^t$)
$$G_{\text{state}}^t = \left\langle V_{\text{event}}^t \cup V_{\text{art}}^t, E_{\text{causal}}^t, x_t \right\rangle$$

- $V_{\text{event}}^t$: 시간 $t$까지의 실행 이벤트 $e_k = \langle A_i, \text{action}, \text{args}, \text{result}, t \rangle$.
- $V_{\text{art}}^t$: 산출물 버전 $a_k$ (코드 diff, 테스트 로그, 중간 상태 문서).
- $E_{\text{causal}}^t$: 인과 의존성 엣지 ($e_1 \xrightarrow{\text{causes}} e_2$, $e \xrightarrow{\text{generates}} a$, $a \xrightarrow{\text{derives}} a'$).

#### (2) 거버넌스 상태 업데이트 게이트 (Governed State Update Gate)
임의의 에이전트 $A_i$가 제안한 상태 변이(State Mutation) $\Delta x$가 공유 시스템 상태 $x_t$에 영속적으로 반영되기 위해서는 4대 무결성 검증 게이트 $\Gamma_{\text{gate}}$를 반드시 통과해야 한다:

$$x_{t+1} = \begin{cases} x_t \oplus \Delta x & \text{if } \Gamma_{\text{gate}}(\Delta x; x_t) = \text{True} \\ x_t & \text{otherwise (Reject / Conflict Flag)} \end{cases}$$

$$\Gamma_{\text{gate}}(\Delta x; x_t) = \text{SchemaCheck}(\Delta x) \land \text{PermCheck}(\Delta x) \land \text{InvariantCheck}(\Delta x, x_t) \land \text{NoConflict}(\Delta x)$$

- **`SchemaCheck` (스키마 무결성 검증)**: $\Delta x$의 데이터 구조 및 타입 정의가 사전에 정의된 상태 스키마 규격을 충족하는지 검증.
- **`PermCheck` (접근 권한 검증)**: 수정을 시도한 에이전트 $A_i$가 대상 상태 객체에 대한 쓰기 권한($\text{permission}_{ir} = 1$)을 보유하고 있는지 확인.
- **`InvariantCheck` (불변성 제약 검증)**: $\Delta x$ 적용 후의 상태가 시스템 전역 불변성 규칙(예: 예산 한도, 데드락 방지, 비모순성 제약)을 만족하는지 검사.
- **`NoConflict` (동시성 충돌 검증)**: 병렬 실행 중인 타 에이전트의 상태 갱신과의 경합(Race Condition / Write-Write Conflict) 발생 여부 확인.

#### (3) 인과 결함 국소화 (Causal Fault Localization)
실행 장애 노드 $v_{\text{fail}}$ 발생 시 인과 조상 집합 $\text{Anc}(v_{\text{fail}})$로부터 근본 원인(Root Cause) 노드 $v_{\text{root}}$를 특정:

$$v_{\text{root}} = \arg\min_{u \in \text{Anc}(v_{\text{fail}})} \left\{ \text{Depth}(u) \;\middle|\; \neg \text{Valid}(u) \land \left( \forall w \in \text{Parents}(u), \text{Valid}(w) \right) \right\}$$

#### (4) 장애 복구 및 회복 경계 (Failure Recovery & Scoped Rollback)
복구 경계 $B_{\text{rec}}$를 산출하여 무효화된 서브그래프 $G_{\text{invalid}}$만 격리하고 부분 재실행:

$$B_{\text{rec}} = \left\{ u \in V \;\middle|\; \text{Valid}(u) = \text{True} \land \exists w \in \text{Children}(u), \neg \text{Valid}(w) \right\}$$

$$G_{\text{exec}}^{\text{recovered}} = \left( G_{\text{exec}} \setminus G_{\text{invalid}}(v_{\text{root}}) \right) \cup \text{RePlan}\left( v_{\text{root}}, \text{Context}(B_{\text{rec}}) \right)$$

---

### 4.4 System Evolution (시스템 진화: 크로스 런 메타 최적화)
시스템 수준의 영속적 구조 그래프 튜플 $\Theta_{\text{sys}} = \langle G_{\text{task}}^0, G_{\text{cap}}^0, G_{\text{team}}^0, G_{\text{comm}}^0 \rangle$에 대해, $k$번째 실행 세션 궤적 $T_k = \langle G_{\text{task}}^{(k)}, G_{\text{coord}}^{(k)}, G_{\text{state}}^{(k)}, Y_k \rangle$을 기반으로 크로스 런 갱신을 수행:

$$\Theta_{\text{sys}}^{(k+1)} = \Theta_{\text{sys}}^{(k)} \oplus U_{\text{sys}}\left( \Theta_{\text{sys}}^{(k)}, T_k \right)$$

1. **과제 워크플로 진화**:
   $$G_{\text{task}}^0 \leftarrow \text{TemplateInduction}\left( G_{\text{task}}^0, \left\{ G_{\text{task}}^{(k)} \;\middle|\; Y_k = \text{Success} \right\} \right)$$
2. **역량 프로필 갱신**:
   $$W_{\text{cap}}(A_i, r) \leftarrow (1-\alpha)W_{\text{cap}}(A_i, r) + \alpha \cdot \text{Feedback}_k(A_i, r)$$
3. **통신 위상 최적화**:
   $$E_{\text{comm}}^0 \leftarrow E_{\text{comm}}^0 \setminus \left\{ (i, j) \;\middle|\; \mathbb{E}_k[\text{Utility}(i \to j)] < \epsilon \right\}$$

---

## 5. Next-Generation Paradigm: Ontology Engineering

| Dimension | Graph Engineering | Ontology Engineering |
|---|---|---|
| **Primary Focus** | Structural Connectivity & Execution Routing | Semantic Formalism, Axioms & Shared Meaning |
| **Node/Edge Semantics** | Ad-hoc labels, task names, data references | Rigorous TBox/ABox classes, OWL/RDF ontologies, formal relations |
| **Constraint Model** | Graph topologies (DAGs, trees, state machines) | Logic rules, invariant axioms, domain consistency constraints |
| **System Boundary** | Single framework / isolated execution graph | Cross-framework, cross-organization semantic interoperability |
| **Goal Formation** | Pre-conditioned on prompt / task decomposition | Value-aligned, autonomous goal formulation and ethical boundaries |

- **Goal Formation & Value Alignment**: Formally specifies high-level organizational objectives, ethical guardrails, and compliance invariants.
- **Shared Semantics & World Grounding**: Bridges vocabulary gaps across heterogeneous multi-agent systems and external enterprise tools (OntoCodex, CoA-Text2OWL, AgentO, Ontology-to-Tools, Palantir Ontology).
- **Standardized Measurement**: Establishes formal semantic metrics for system intelligence, isolating organizational efficiency from raw foundation model compute.

---

## 6. Comprehensive Survey Comparison (Table 4 Excerpt)

| Survey / Study | Year | Harness | Loop | Planning | Workflow | MAS | State | Self-Evolution | Ontology |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| LLM Agents (Wang et al.) | 2026 | ◦ | – | ✓ | – | ✓ | – | – | – |
| Graphs Meet Agents (Chen et al.) | 2025 | ◦ | – | ✓ | ◦ | ✓ | – | – | – |
| Graph-Augmented Agents (Peng et al.) | 2025 | ◦ | – | ✓ | ◦ | ✓ | – | ◦ | – |
| Agent Harness (Huang et al.) | 2026 | ✓ | ✓ | ✓ | ◦ | ✓ | ✓ | ◦ | – |
| Runtime Graphs (Zheng et al.) | 2026 | ◦ | ◦ | ◦ | ✓ | ◦ | ◦ | ◦ | – |
| Multi-Agent Orchestration (Zhang et al.) | 2026 | ◦ | ◦ | ✓ | ✓ | ✓ | ✓ | – | – |
| Dynamic Graph Transformation (Xu et al.) | 2026 | ◦ | – | ◦ | ✓ | ✓ | ✓ | ✓ | ◦ |
| **Graph Engineering (Feng et al., Ours)** | **2026** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** |

*Notation: ✓: primary organizing axis / dedicated taxonomy; ◦: substantive secondary coverage; –: absent or incidental background.*
