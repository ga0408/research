# Graph Engineering in the Era of LLM Agents: From Individual Intelligence to System Intelligence — 핵심 발췌

> 출처: [분석 문서](../../report/[paper]_Graph_Engineering_in_the_Era_of_LLM_Agents_From_Individual_Intelligence_to_System_Intelligence_2026_arxiv.md) / 원본: [arXiv:2608.21156](https://arxiv.org/abs/2608.21156)

---

## 1. Preliminaries & Formal Definitions

### 1.1 Individual Agent
An Individual Agent is an autonomous computational entity that perceives its environment, makes decisions, executes actions, and adapts its behavior according to feedback. It consists primarily of a Foundation Model $\mathcal{F}_i$, an Agent Harness $\mathcal{H}_i$, and an Agent Loop:

$$\mathcal{A}_i = \text{Loop}\left(\mathcal{F}_i, \mathcal{H}_i; s_i^t\right)$$

- $\mathcal{F}_i$: Foundation Model serving as the cognitive core (language understanding, reasoning, planning, code/content generation).
- $\mathcal{H}_i$: Agent Harness extending intrinsic capabilities through perception, context construction, memory/knowledge access, tool invocation, reusable skills, and runtime governance.
- $s_i^t$: Local runtime state of agent $i$ at time $t$.
- $\text{Loop}$: Iterative execution cycle governing continuous perception $\rightarrow$ reasoning $\rightarrow$ action $\rightarrow$ feedback processing $\rightarrow$ state update.

### 1.2 Agent System & System Intelligence
An Agent System extends the Individual Agent abstraction to a collection of agents operating via shared resources, environments, and coordination mechanisms at time $t$:

$$\mathcal{S}_t = \left\langle \mathcal{A}_t, \mathcal{R}_t, \mathcal{E}_t, \Pi_t, x_t \right\rangle$$

- $\mathcal{A}_t = \{\mathcal{A}_1, \dots, \mathcal{A}_n\}$: Agent Team, where each agent possesses its own model, harness, loop, and local state.
- $\mathcal{R}_t$: Shared Resources (tools, model endpoints, shared memory/KBs, verifiers, human-in-the-loop).
- $\mathcal{E}_t$: External Environment providing observations and receiving actions.
- $\Pi_t$: Coordination Mechanisms governing task assignment, communication routing, consensus, and fault handling.
- $x_t$: Global System State describing system-wide task progress, artifact states, resource locks, and failure records.

**Definition of System Intelligence:**
> *"System Intelligence is the ability of an agent system $\mathcal{S}_t$ to organize and coordinate multiple intelligent components into a coherent, adaptive whole that pursues a shared objective across changing conditions, heterogeneous expertise, parallel branches, and partial failures."*

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

## 3. Fundamental Limitations of Individual Intelligence

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

### 4.1 Task Organization ($\mathcal{G}_{\text{task}}$)
과제 조직화는 비구조적 목표를 방향성 비순환 그래프(DAG) 또는 하이퍼그래프 $\mathcal{G}_{\text{task}}$로 구조화하여 스케줄링 및 실행 흐름을 제어한다:

$$\mathcal{G}_{\text{task}} = \left\langle \mathcal{V}_{\text{task}}, \mathcal{E}_{\text{task}}, \Phi_{\text{task}}, \Sigma_{\text{task}} \right\rangle$$

- $\mathcal{V}_{\text{task}} = \{v_1, \dots, v_m\}$: 원자적 하위 과제(Subtask/Subgoal) 노드 집합. 각 노드 $v_k = \langle \text{spec}_k, \text{input}_k, \text{output}_k \rangle$.
- $\mathcal{E}_{\text{task}} \subseteq \mathcal{V}_{\text{task}} \times \mathcal{V}_{\text{task}} \times \mathcal{T}_{\text{dep}}$: 타입화된 의존성 엣지 집합:
  - **데이터 흐름 의존성 (Dataflow)**: $u \xrightarrow{\text{data}} v \iff \text{output}(u) \subseteq \text{input}(v)$
  - **선행 제약 (Precedence)**: $u \prec v \iff t_{\text{start}}(v) \ge t_{\text{finish}}(u)$
  - **조건부 분기 (Conditional)**: $u \xrightarrow{\text{cond}(c)} v$
  - **검증/승인 게이트 (Review Gate)**: $u \xrightarrow{\text{verify}} v$
- $\Sigma_{\text{task}}(v) \in \{\text{Pending}, \text{Ready}, \text{Running}, \text{Blocked}, \text{Committed}, \text{Failed}\}$: 런타임 노드 실행 상태.
- **Ready 상태 전이 조건**:
  $$\Sigma_{\text{task}}(v) \leftarrow \text{Ready} \iff \forall u \in \text{Parents}(v), \Sigma_{\text{task}}(u) = \text{Committed}$$
- **워크플로 최적화 (Workflow Optimization as Structural Search)**:
  $$\mathcal{G}_{\text{task}}^* = \arg\max_{\mathcal{G} \in \Omega(\text{Goal})} \mathbb{E}_{\tau \sim \mathcal{G}}\left[ \mathcal{R}(\tau) - \lambda_1 \cdot \text{Cost}(\tau) - \lambda_2 \cdot \text{Latency}(\tau) \right]$$
- **임계 경로 메이크스팬 (Critical Path Makespan)**:
  $$T_{\text{makespan}}(\mathcal{G}_{\text{task}}) = \max_{p \in \text{Paths}(\mathcal{G}_{\text{task}})} \sum_{v \in p} \text{Duration}(v)$$

---

### 4.2 Agent Coordination ($\mathcal{G}_{\text{cap}}, \mathcal{G}_{\text{team}}, \mathcal{G}_{\text{comm}}^t$)
에이전트 조율은 이종 에이전트의 역량 매핑, 팀 위계, 동적 통신 채널을 3대 그래프로 정식화한다:

#### (1) 에이전트 역량 이분 그래프 (Agent Capability Bipartite Graph $\mathcal{G}_{\text{cap}}$)
$$\mathcal{G}_{\text{cap}} = \left\langle \mathcal{V}_{\text{agent}} \cup \mathcal{V}_{\text{res}}, \mathcal{E}_{\text{cap}}, \mathbf{W}_{\text{cap}} \right\rangle$$

- $\mathcal{V}_{\text{agent}} = \{\mathcal{A}_1, \dots, \mathcal{A}_n\}$: 에이전트 집합.
- $\mathcal{V}_{\text{res}} = \mathcal{K}_{\text{skills}} \cup \mathcal{T}_{\text{tools}} \cup \mathcal{D}_{\text{data}}$: 자원(스킬, 도구, 데이터베이스) 노드 집합.
- $\mathcal{E}_{\text{cap}} \subseteq \mathcal{V}_{\text{agent}} \times \mathcal{V}_{\text{res}}$: 소유 및 접근 권한 엣지.
- $\mathbf{W}_{\text{cap}}(\mathcal{A}_i, r) = \langle \text{proficiency}_{ir}, \text{permission}_{ir}, \text{reliability}_{ir} \rangle$: 역량 가중치 튜플.
- **에이전트-과제 최적 할당 함수 ($\mu^*: \mathcal{V}_{\text{task}} \to \mathcal{V}_{\text{agent}}$)**:
  $$\mu^*(v) = \arg\max_{\mathcal{A}_i \in \mathcal{V}_{\text{agent}}} \left[ \text{Match}\left(\mathbf{W}_{\text{cap}}(\mathcal{A}_i), \text{Req}(v)\right) \cdot \text{Avail}(\mathcal{A}_i, t) \right]$$

#### (2) 에이전트 팀 조직 그래프 ($\mathcal{G}_{\text{team}}$)
$$\mathcal{G}_{\text{team}} = \left\langle \mathcal{V}_{\text{agent}}, \mathcal{E}_{\text{team}}, \text{Role} \right\rangle$$

- $\mathcal{E}_{\text{team}}$: 위계 및 보고 관계($\mathcal{A}_{\text{lead}} \xrightarrow{\text{supervise}} \mathcal{A}_{\text{sub}}$), 피어 협업($\mathcal{A}_i \xleftrightarrow{\text{peer}} \mathcal{A}_j$), 독립 검토 게이트($\mathcal{A}_{\text{worker}} \xrightarrow{\text{submit}} \mathcal{A}_{\text{reviewer}}$).

#### (3) 동적 통신 그래프 ($\mathcal{G}_{\text{comm}}^t$)
$$\mathcal{G}_{\text{comm}}^t = \left\langle \mathcal{V}_{\text{agent}}, \mathcal{E}_{\text{comm}}^t, \mathbf{M}^t \right\rangle$$

- **동적 통신 가지치기 (Communication Sparsification)**: $O(N^2)$ 메시지 범람을 차단하고 관련성 기반 활성 채널만 유지:
  $$\mathcal{E}_{\text{comm}}^t = \left\{ (i, j) \in \mathcal{V}_{\text{agent}}^2 \;\middle|\; \text{Score}\left(\mathbf{m}_{i \to j}^t, \text{Context}_j^t\right) \ge \tau_{\text{comm}} \land \text{Perm}(i \to j) = 1 \right\}$$
- $\mathbf{m}_{i \to j}^t$: 구조화된 교환 메시지 $\langle \text{Sender}, \text{Receiver}, \text{Type}, \text{Payload}, \text{ArtifactID}, t \rangle$.

---

### 4.3 Runtime State Management ($\mathcal{G}_{\text{state}}^t, \mathcal{G}_{\text{exec}}^t$)
분산 런타임 환경에서 글로벌 상태의 일관성, 인과적 결함 격리 및 부분 롤백을 제어한다:

#### (1) 글로벌 상태 & 인과 실행 그래프 ($\mathcal{G}_{\text{state}}^t, \mathcal{G}_{\text{exec}}^t$)
$$\mathcal{G}_{\text{state}}^t = \left\langle \mathcal{V}_{\text{event}}^t \cup \mathcal{V}_{\text{art}}^t, \mathcal{E}_{\text{causal}}^t, \mathbf{x}_t \right\rangle$$

- $\mathcal{V}_{\text{event}}^t$: 시간 $t$까지의 실행 이벤트 $e_k = \langle \mathcal{A}_i, \text{action}, \text{args}, \text{result}, t \rangle$.
- $\mathcal{V}_{\text{art}}^t$: 산출물 버전 $a_k$ (코드 diff, 테스트 로그, 중간 상태 문서).
- $\mathcal{E}_{\text{causal}}^t$: 인과 의존성 엣지 ($e_1 \xrightarrow{\text{causes}} e_2$, $e \xrightarrow{\text{generates}} a$, $a \xrightarrow{\text{derives}} a'$).

#### (2) 거버넌스 상태 업데이트 게이트 (Governed State Update Gate)
공유 시스템 상태 $\mathbf{x}_t$로의 반영은 4대 불변성 검증 게이트 $\Gamma_{\text{gate}}$를 통과해야 한다:

$$\mathbf{x}_{t+1} = \begin{cases} \mathbf{x}_t \oplus \Delta \mathbf{x} & \text{if } \Gamma_{\text{gate}}(\Delta \mathbf{x}; \mathbf{x}_t) = \text{True} \\ \mathbf{x}_t & \text{otherwise (Reject / Conflict Flag)} \end{cases}$$

$$\Gamma_{\text{gate}}(\Delta \mathbf{x}) = \text{SchemaCheck}(\Delta \mathbf{x}) \land \text{PermCheck}(\Delta \mathbf{x}) \land \text{InvariantCheck}(\Delta \mathbf{x}, \mathbf{x}_t) \land \text{NoConflict}(\Delta \mathbf{x})$$

#### (3) 인과 결함 국소화 (Causal Fault Localization)
실행 장애 노드 $v_{\text{fail}}$ 발생 시 인과 조상 집합 $\text{Anc}(v_{\text{fail}})$로부터 근본 원인(Root Cause) 노드 $v_{\text{root}}$를 특정:

$$v_{\text{root}} = \arg\min_{u \in \text{Anc}(v_{\text{fail}})} \left\{ \text{Depth}(u) \;\middle|\; \neg \text{Valid}(u) \land \left( \forall w \in \text{Parents}(u), \text{Valid}(w) \right) \right\}$$

#### (4) 장애 복구 및 회복 경계 (Failure Recovery & Scoped Rollback)
복구 경계 $\mathcal{B}_{\text{rec}}$를 산출하여 무효화된 서브그래프 $\mathcal{G}_{\text{invalid}}$만 격리하고 부분 재실행:

$$\mathcal{B}_{\text{rec}} = \left\{ u \in \mathcal{V} \;\middle|\; \text{Valid}(u) = \text{True} \land \exists w \in \text{Children}(u), \neg \text{Valid}(w) \right\}$$

$$\mathcal{G}_{\text{exec}}^{\text{recovered}} = \left( \mathcal{G}_{\text{exec}} \setminus \mathcal{G}_{\text{invalid}}(v_{\text{root}}) \right) \cup \text{RePlan}\left( v_{\text{root}}, \text{Context}(\mathcal{B}_{\text{rec}}) \right)$$

---

### 4.4 System Evolution (시스템 진화: 크로스 런 메타 최적화)
시스템 수준의 영속적 구조 그래프 튜플 $\mathbf{\Theta}_{\text{sys}} = \langle \mathcal{G}_{\text{task}}^0, \mathcal{G}_{\text{cap}}^0, \mathcal{G}_{\text{team}}^0, \mathcal{G}_{\text{comm}}^0 \rangle$에 대해, $k$번째 실행 세션 궤적 $\mathcal{T}_k = \langle \mathcal{G}_{\text{task}}^{(k)}, \mathcal{G}_{\text{coord}}^{(k)}, \mathcal{G}_{\text{state}}^{(k)}, \mathcal{Y}_k \rangle$을 기반으로 크로스 런 갱신을 수행:

$$\mathbf{\Theta}_{\text{sys}}^{(k+1)} = \mathbf{\Theta}_{\text{sys}}^{(k)} \oplus \mathcal{U}_{\text{sys}}\left( \mathbf{\Theta}_{\text{sys}}^{(k)}, \mathcal{T}_k \right)$$

1. **과제 워크플로 진화**:
   $$\mathcal{G}_{\text{task}}^0 \leftarrow \text{TemplateInduction}\left( \mathcal{G}_{\text{task}}^0, \left\{ \mathcal{G}_{\text{task}}^{(k)} \;\middle|\; \mathcal{Y}_k = \text{Success} \right\} \right)$$
2. **역량 프로필 갱신**:
   $$\mathbf{W}_{\text{cap}}(\mathcal{A}_i, r) \leftarrow (1-\alpha)\mathbf{W}_{\text{cap}}(\mathcal{A}_i, r) + \alpha \cdot \text{Feedback}_k(\mathcal{A}_i, r)$$
3. **통신 위상 최적화**:
   $$\mathcal{E}_{\text{comm}}^0 \leftarrow \mathcal{E}_{\text{comm}}^0 \setminus \left\{ (i, j) \;\middle|\; \mathbb{E}_k[\text{Utility}(i \to j)] < \epsilon \right\}$$

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
