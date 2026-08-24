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

## 4. Graph Engineering Architecture: The Core Triad & Evolution

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
- **Goal Decomposition**: Translates high-level unstructured intent into a directed acyclic graph (DAG) or hypergraph $\mathcal{G}_{\text{task}} = (\mathcal{V}_{\text{task}}, \mathcal{E}_{\text{task}})$, where vertices $\mathcal{V}$ denote atomic subtasks and edges $\mathcal{E}$ encode data dependencies, temporal ordering, and conditional execution branches (HuggingGPT, ReWOO, LLMCompiler, Plan-over-Graph, TDAG).
- **Workflow Optimization**: Formulates graph compilation and execution as structural optimization over candidate execution topologies (AFlow, GPTSwarm, MermaidFlow, DyFlow, EvoFlow, QualityFlow). Includes topological scheduling, parallel branch dispatching, and dynamic path pruning.

### 4.2 Agent Coordination ($\mathcal{G}_{\text{cap}}, \mathcal{G}_{\text{team}}, \mathcal{G}_{\text{comm}}$)
- **Agent Capability Modeling ($\mathcal{G}_{\text{cap}}$)**: Bipartite/attributed graph linking agent entities to verified skills, tool permissions, and contextual reliability metrics (DyLAN, SkillGraph, MasRouter).
- **Agent Team Organization ($\mathcal{G}_{\text{team}}$)**: Organizational structure mapping agents into hierarchical, decentralized mesh, or dynamic committee structures with explicit authority boundaries and review gates (MetaGPT, ChatDev, Magentic-One, SwarmAgentic, Mixture-of-Agents).
- **Multi-Agent Communication ($\mathcal{G}_{\text{comm}}$)**: Governs information flow channels, bandwidth throttling, semantic message filtering, and structured handoffs to prevent Quadratic Communication Overhead $O(N^2)$ and message flooding (AgentPrune, AgentDropout, DyTopo).

### 4.3 Runtime State Management ($\mathcal{G}_{\text{state}}, \mathcal{G}_{\text{exec}}$)
- **State Recording**: Maintains a global, auditable provenance DAG tracking events, artifact versions, tool execution side-effects, and role commitments across distributed execution threads (LangGraph, Burr, MemTX, SagaLLM, PatchBoard, AgentGit).
- **Fault Localization**: Analyzes causal dependencies on execution graphs to isolate root causes (e.g., distinguishing whether a failure was caused by bad upstream requirements, incorrect code synthesis, or flaky environment tests) (CausalFlow, ReflexGrad, TDAD, MAST).
- **Failure Recovery**: Executes transactional rollbacks, scoped sub-graph retries, and dynamic re-routing around failed nodes without discarding verified upstream progress (Atomix, SagaLLM, Cordon).

### 4.4 System Evolution
- Leverages post-execution evidence and trajectory audits to continuously mutate system graphs across runs. Updates task workflow templates ($\Delta \mathcal{G}_{\text{task}}$), refines agent capability scores ($\Delta \mathcal{G}_{\text{cap}}$), and prunes unproductive communication links ($\Delta \mathcal{G}_{\text{comm}}$) (EvoFlow, QueenBee Planner, CARD, Meta-Team, DyTopo).

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
