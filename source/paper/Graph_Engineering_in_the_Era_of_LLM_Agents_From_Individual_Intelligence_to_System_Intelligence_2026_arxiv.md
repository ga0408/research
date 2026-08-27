# Graph Engineering in the Era of LLM Agents: From Individual Intelligence to System Intelligence — 핵심 발췌

> 출처: [분석 문서](../../report/[paper]_Graph_Engineering_in_the_Era_of_LLM_Agents_From_Individual_Intelligence_to_System_Intelligence_2026_arxiv.md) / 원본: [arXiv:2608.21156](https://arxiv.org/abs/2608.21156)

---

## 1. Preliminaries & Formal Definitions

### 1.1 Individual Agent
An Individual Agent is an autonomous computational entity that perceives its environment, makes decisions, executes actions, and adapts its behavior according to feedback. It consists primarily of a Foundation Model `F_i`, an Agent Harness `H_i`, and an Agent Loop:

```
A_i = Loop(F_i, H_i; s_i^t)
```

- `F_i`: Foundation Model serving as the cognitive core (language understanding, reasoning, planning, code/content generation).
- `H_i`: Agent Harness extending intrinsic capabilities through perception, context construction, memory/knowledge access, tool invocation, reusable skills, and runtime governance.
- `s_i^t`: Local runtime state of agent `i` at time `t`.
- `Loop`: Iterative execution cycle governing continuous perception → reasoning → action → feedback processing → state update.

### 1.2 Agent System & System Intelligence
An Agent System extends the Individual Agent abstraction to a collection of agents operating via shared resources, environments, and coordination mechanisms at time `t`:

```
S_t = <A_t, R_t, E_t, Π_t, x_t>
```

- `A_t = {A_1, ..., A_n}`: Agent Team, where each agent possesses its own model, harness, loop, and local state.
- `R_t`: Shared Resources (tools, model endpoints, shared memory/KBs, verifiers, human-in-the-loop).
- `E_t`: External Environment providing observations and receiving actions.
- `Π_t`: Coordination Mechanisms governing task assignment, communication routing, consensus, and fault handling.
- `x_t`: Global System State describing system-wide task progress, artifact states, resource locks, and failure records.

**Definition of System Intelligence:**
> *"System Intelligence is the ability of an agent system `S_t` to organize and coordinate multiple intelligent components into a coherent, adaptive whole that pursues a shared objective across changing conditions, heterogeneous expertise, parallel branches, and partial failures."*

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

### 4.1 Task Organization (`G_task`)
과제 조직화는 비구조적 목표를 방향성 비순환 그래프(DAG) 또는 하이퍼그래프 `G_task`로 구조화하여 스케줄링 및 실행 흐름을 제어한다:

```
G_task = <V_task, E_task, Φ_task, Σ_task>
```

- `V_task = {v_1, ..., v_m}`: 원자적 하위 과제(Subtask/Subgoal) 노드 집합. 각 노드 `v_k = <spec_k, input_k, output_k>`.
- `E_task ⊆ V_task × V_task × T_dep`: 타입화된 의존성 엣지 집합:
  - **데이터 흐름 의존성 (Dataflow)**: `u —(data)→ v  <=>  output(u) ⊆ input(v)`
  - **선행 제약 (Precedence)**: `u ≺ v  <=>  t_start(v) ≥ t_finish(u)`
  - **조건부 분기 (Conditional)**: `u —(cond(c))→ v`
  - **검증/승인 게이트 (Review Gate)**: `u —(verify)→ v`
- `Σ_task(v) ∈ {Pending, Ready, Running, Blocked, Committed, Failed}`: 런타임 노드 실행 상태.
- **Ready 상태 전이 조건**:
  ```
  Σ_task(v) ← Ready   <=>   ∀u ∈ Parents(v), Σ_task(u) = Committed
  ```
- **워크플로 최적화 (Workflow Optimization as Structural Search)**:
  ```
  G_task* = argmax_{G ∈ Ω(Goal)} E_{τ ~ G}[ R(τ) - λ_1 * Cost(τ) - λ_2 * Latency(τ) ]
  ```
- **임계 경로 메이크스팬 (Critical Path Makespan)**:
  ```
  T_makespan(G_task) = max_{p ∈ Paths(G_task)} Σ_{v ∈ p} Duration(v)
  ```

---

### 4.2 Agent Coordination (`G_cap, G_team, G_comm^t`)
에이전트 조율은 이종 에이전트의 역량 매핑, 팀 위계, 동적 통신 채널을 3대 그래프로 정식화한다:

#### (1) 에이전트 역량 이분 그래프 (Agent Capability Bipartite Graph `G_cap`)
```
G_cap = <V_agent ∪ V_res, E_cap, W_cap>
```

- `V_agent = {A_1, ..., A_n}`: 에이전트 집합.
- `V_res = K_skills ∪ T_tools ∪ D_data`: 자원(스킬, 도구, 데이터베이스) 노드 집합.
- `E_cap ⊆ V_agent × V_res`: 소유 및 접근 권한 엣지.
- `W_cap(A_i, r) = <proficiency_ir, permission_ir, reliability_ir>`: 역량 가중치 튜플.
- **에이전트-과제 최적 할당 함수 (`μ*: V_task → V_agent`)**:
  ```
  μ*(v) = argmax_{A_i ∈ V_agent} [ Match(W_cap(A_i), Req(v)) * Avail(A_i, t) ]
  ```

#### (2) 에이전트 팀 조직 그래프 (`G_team`)
```
G_team = <V_agent, E_team, Role>
```

- `E_team`: 위계 및 보고 관계(`A_lead —(supervise)→ A_sub`), 피어 협업(`A_i ←(peer)→ A_j`), 독립 검토 게이트(`A_worker —(submit)→ A_reviewer`).

#### (3) 동적 통신 그래프 (`G_comm^t`)
```
G_comm^t = <V_agent, E_comm^t, M^t>
```

- **동적 통신 가지치기 (Communication Sparsification)**: `O(N^2)` 메시지 범람을 차단하고 관련성 기반 활성 채널만 유지:
  ```
  E_comm^t = { (i, j) ∈ V_agent^2 | Score(m_{i->j}^t, Context_j^t) ≥ τ_comm  and  Perm(i->j) = 1 }
  ```
- `m_{i->j}^t`: 구조화된 교환 메시지 `<Sender, Receiver, Type, Payload, ArtifactID, t>`.

---

### 4.3 Runtime State Management (`G_state^t, G_exec^t`)
분산 런타임 환경에서 글로벌 상태의 일관성, 인과적 결함 격리 및 부분 롤백을 제어한다:

#### (1) 글로벌 상태 & 인과 실행 그래프 (`G_state^t, G_exec^t`)
```
G_state^t = <V_event^t ∪ V_art^t, E_causal^t, x_t>
```

- `V_event^t`: 시간 `t`까지의 실행 이벤트 `e_k = <A_i, action, args, result, t>`.
- `V_art^t`: 산출물 버전 `a_k` (코드 diff, 테스트 로그, 중간 상태 문서).
- `E_causal^t`: 인과 의존성 엣지 (`e_1 —(causes)→ e_2`, `e —(generates)→ a`, `a —(derives)→ a'`).

#### (2) 거버넌스 상태 업데이트 게이트 (Governed State Update Gate)
임의의 에이전트 `A_i`가 제안한 상태 변이(State Mutation) `Δx`가 공유 시스템 상태 `x_t`에 영속적으로 반영되기 위해서는 4대 무결성 검증 게이트 `Γ_gate`를 반드시 통과해야 한다:

```
x_{t+1} = x_t ⊕ Δx    (if Γ_gate(Δx; x_t) == True)
        = x_t          (otherwise: Reject / Conflict Flag)

Γ_gate(Δx; x_t) = SchemaCheck(Δx) ∧ PermCheck(Δx) ∧ InvariantCheck(Δx, x_t) ∧ NoConflict(Δx)
```

- **`SchemaCheck` (스키마 무결성 검증)**: `Δx`의 데이터 구조 및 타입 정의가 사전에 정의된 상태 스키마 규격을 충족하는지 검증.
- **`PermCheck` (접근 권한 검증)**: 수정을 시도한 에이전트 `A_i`가 대상 상태 객체에 대한 쓰기 권한(`permission_ir = 1`)을 보유하고 있는지 확인.
- **`InvariantCheck` (불변성 제약 검증)**: `Δx` 적용 후의 상태가 시스템 전역 불변성 규칙(예: 예산 한도, 데드락 방지, 비모순성 제약)을 만족하는지 검사.
- **`NoConflict` (동시성 충돌 검증)**: 병렬 실행 중인 타 에이전트의 상태 갱신과의 경합(Race Condition / Write-Write Conflict) 발생 여부 확인.

#### (3) 인과 결함 국소화 (Causal Fault Localization)
실행 장애 노드 `v_fail` 발생 시 인과 조상 집합 `Anc(v_fail)`로부터 근본 원인(Root Cause) 노드 `v_root`를 특정:

```
v_root = argmin_{u ∈ Anc(v_fail)} { Depth(u) | ¬Valid(u) ∧ (∀w ∈ Parents(u), Valid(w)) }
```

#### (4) 장애 복구 및 회복 경계 (Failure Recovery & Scoped Rollback)
복구 경계 `B_rec`를 산출하여 무효화된 서브그래프 `G_invalid`만 격리하고 부분 재실행:

```
B_rec = { u ∈ V | Valid(u) == True ∧ ∃w ∈ Children(u), ¬Valid(w) }
G_exec^recovered = ( G_exec \ G_invalid(v_root) ) ∪ RePlan( v_root, Context(B_rec) )
```

---

### 4.4 System Evolution (시스템 진화: 크로스 런 메타 최적화)
시스템 수준의 영속적 구조 그래프 튜플 `Θ_sys = <G_task^0, G_cap^0, G_team^0, G_comm^0>`에 대해, `k`번째 실행 세션 궤적 `T_k = <G_task^(k), G_coord^(k), G_state^(k), Y_k>`을 기반으로 크로스 런 갱신을 수행:

```
Θ_sys^(k+1) = Θ_sys^(k) ⊕ U_sys( Θ_sys^(k), T_k )
```

1. **과제 워크플로 진화**:
   ```
   G_task^0 ← TemplateInduction( G_task^0, { G_task^(k) | Y_k = Success } )
   ```
2. **역량 프로필 갱신**:
   ```
   W_cap(A_i, r) ← (1 - α) * W_cap(A_i, r) + α * Feedback_k(A_i, r)
   ```
3. **통신 위상 최적화**:
   ```
   E_comm^0 ← E_comm^0 \ { (i, j) | E_k[ Utility(i->j) ] < ε }
   ```

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
