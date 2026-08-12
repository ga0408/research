# Self-Improvements in Modern Agentic Systems: A Survey — 핵심 발췌

> 출처: [분석 문서](../../report/[paper]_Self-Improvements_in_Modern_Agentic_Systems_A_Survey_2026_arxiv.md) / 원본: [arXiv:2607.13104](https://arxiv.org/abs/2607.13104)
>
> Zhe Ren, Yimeng Chen, Dandan Guo, et al. (Jilin University & KAUST; Jürgen Schmidhuber). arXiv:2607.13104v1 [cs.AI], 14 Jul 2026. 97 pages. Survey.

---

## 1. 핵심 정식화 (Section 3 Definitions)

### Agent configuration (Eq. 1, 2)

```
A_t = (θ_t , Σ_t)                                  # θ: FM 신경 파라미터, Σ: operational scaffold
Σ_t := ( p_t , m_t , T_t , g_t )                    # p=prompt, m=memory, T=tools, g=control logic
```

- FM(hypothesis/policy core)는 stateless 추론 엔진 → 자율성을 위해 영속·상호작용 scaffold Σ와 결합 필요.
- 실행 중 ephemeral state X_t(KV cache, 중간 계획, 단기 working memory)는 task 경계에서 reset → intrinsic config가 아님.
- 유도 정책: π_{θ_t,Σ_t}( A_t | X_t ) (Eq. 3). 행동은 θ와 Σ가 **공동** 결정.

### Self-Improvement as self-induced operator (Eq. 4)

```
A_{t+1} = U( A_{1:t}, E(π_{θ_t,Σ_t} ; Σ_t, C_t) )
```

- E: agent 실행 절차 → 학습 신호(trajectory, reflection, critique, 제안 edit) 생성. Σ_t가 명시 인자로 들어가 **직접 self-inspection** 허용(예: prompt 템플릿 비판, tool config audit).
- U: 시스템 수준 update rule → intrinsic component(θ 또는 Σ)에 **durable** 변경 commit. (단순 X_t 진화와 구분)
- 두 가지 self-reference mode:
  1. **분포 수준**: policy 실행 → 경험/보조 artifact 생성 → 외부 optimizer가 θ update (FM improvement)
  2. **실행 수준 직접 수정**: policy 실행 → prompt/memory/tool/control logic 직접 edit → Σ update (scaffolding improvement)

### FM improvement (Eq. 5) vs Scaffolding improvement (Eq. 6)

```
FM:           θ_{t+1} = U_θ( θ_{1:t}, E(π;Σ,C) ),  Σ_{t+1}=Σ_t     # parametric, slow, stable, global
Scaffolding:  Σ_{t+1} = U_Σ( Σ_{1:t}, E(π;Σ,C) ),  θ_{t+1}=θ_t     # non-parametric, fast, reversible, context-dependent
```

### Skill = reusable update operator

- skill은 U의 named reusable 인스턴스. substrate(T, p, m, θ, g) 중 하나로 직렬화. **substrate 축과 직교**(동일 store-retrieve 구조가 tool routing·memory 모두에 재등장).
- 범위: **object-level**(세계/태스크 상태에 작용, HRL option의 analog; 획득 시 U 실행=`collect-wood` 루틴) vs **meta-level**(자기 config A_t에 작용 → Σ edit 또는 θ update). 자기개선의 핵심은 **meta-level**.
- meta-level skill은 A_t에 작용하면서 자신도 A_t에 직렬화 → improver와 피개선계가 함께 진화하는 **self-referential loop** 회복 (Schmidhuber 1987/1993/2003).

### RL과의 관계 (Section 3.3)
- θ update = 고정 decision process 하의 표준 policy optimization (RLHF/PPO/DPO).
- Σ update = 표준 RL에 대응 없는 structural meta-learning: action space·state 표현·관측 처리 로직을 동적 변경 → **MDP 자체를 재형성**. 단일 trial 내 fast weight ~ in-context adaptation(non-parametric); RLHF/Reflexion/스킬 라이브러리는 slow consolidation 루프.

---

## 2. 통합 분류 체계 (Section 4 Taxonomy)

```
Self-Improvement
├── (1) Foundation Model Improvement  θ_t → θ_{t+1}    [parametric / slow loop]
│   ├── 5.1 Intrinsic Generative Demonstrations   (S_t ≈ D_t: 예시·합성데이터)
│   ├── 5.2 Intrinsic Evaluative Feedback          (S_t ≈ e_t: rubric/consistency/corrective)
│   └── 5.3 Extrinsic Exploratory Experience       (S_t ≈ τ_t: grounded/simulated env)
│        ├── 5.3.1 Grounded task env interaction
│        └── 5.3.2 Simulated proxy env interaction
└── (2) Scaffolding Improvement  Σ_t → Σ_{t+1}    [non-parametric / fast loop]
    ├── 6.1 Prompt    p_{t+1}=IMPROVE_p(p;S_t)
    │   ├── 6.1.1 Scalar-Feedback Optimization
    │   ├── 6.1.2 Qualitative-Feedback Refinement
    │   ├── 6.1.3 Population-Based Evolution
    │   └── 6.1.4 Textual Gradient Optimization
    ├── 6.2 Memory   m_{t+1}=IMPROVE_m(m;S_t)      Eq.20
    │   ├── 6.2.1 Memory Object   (explicit / implicit)
    │   ├── 6.2.2 Memory Structure (flat / hier / graph / vector)
    │   └── 6.2.3 Memory Processing (CRUD: Create/Read/Update/Delete)
    ├── 6.3 Tool     T_{t+1}=IMPROVE_T(T;S_t)      Eq.21  [Tool Governance Metacognition]
    │   ├── 6.3.1 Dynamic Tool Routing
    │   ├── 6.3.2 Iterative Tool Refinement
    │   └── 6.3.3 Autonomous Tool Creation
    └── 6.4 Full Scaffolding  Σ_{t+1}=IMPROVE_Σ(Σ;S_t)  Eq.22/23  [self-referential]
```

Fig.1: 두 pathway를 "what is modified"로 구분. 신호가 어디서 오는지(intrinsic demo / intrinsic feedback / extrinsic experience)로 세분화. 개별 scaffold component마다 generic update signal S_t가 prompt/memory/tool/full-Σ 구동.

---

## 3. Foundation Model Improvement (Section 5)

학습 신호 S_t 성격에 따라 3분류. θ만 update, Σ 고정.
- **5.1 Intrinsic generative demonstrations** (S_t≈D_t): agent가 스스로 예시/합성 데이터 생성(self-instruct, consistency filtering). 정답 추출 어려운 도메인에서 자가 학습 데이터 구축.
- **5.2 Intrinsic evaluative feedback** (S_t≈e_t): 환경 추가 상호작용 없이 rubric/consistency/corrective 신호 생성.
  - *Rubric feedback*: 점수·랭킹·선호(constitutional AI, AI feedback → preference model).
  - *Consistency feedback*: 확률적 동작 활용, 다수결/자기합치로 보상·선호 구성(TTRL). "correct→self-agreeing" 가정 → 취약.
  - *Corrective feedback*: 자연어 critique/modification 자체가 산출(점수 아님). 보상 hacking·ego bias 위험.
- **5.3 Extrinsic exploratory experience** (S_t≈τ_t): 실제 환경/시뮬레이터와 상호작용해 trajectory 수집(RL/자가플레이). grounded env ↔ simulated proxy env.

Algorithm 1: 각 subcategory 신호를 합집합→filter/weight→U_θ update→checkpoint rollback 가능.

---

## 4. Scaffolding Improvement — Memory (Section 6.2) ★

메모리를 수동 저장이 아닌 **능동 진화 scaffold**로 재정의. frozen FM 가정 하의 *비모수 외부 메모리* 만 다룸(가중치 내 parametric memory는 제외).

```
m_t := ( object_t , structure_t )
m_{t+1} = IMPROVE_m( m_t ; S_t )        # Eq.20, signal-driven CRUD family
```

### 6.2.1 Memory Object (object_t) — 무엇을 저장?
- **Explicit**(가독성·감사 용이, scalability 비용):
  - (i) *Processed interaction trails*: raw 궤적 → 재사용 가능 의미 단위로 압축(routine/heuristic/reflection). generalization ↑.
  - (ii) *Curated raw content*: 요약손실 큰 정확 surface 보존(코드/수식/스크린샷). trial-error 검증된 고가치 artifact만 write-back. precision ↑.
  - (iii) *Integrated external knowledge*: 외부 저장소 사실 통합·동적 갱신/주석/가지치기. verifiability ↑.
- **Implicit**(compact·고속 associative, debug/감사 어려움, representation drift): latent token, hidden state, KV cache augmentation(generative latent memory, latent state reconstruction, offline coprocessor injection, updatable latent pool).
- Trade-off 표(Table 3): processed trails→summary bias/stale heuristics; raw content→retrieval noise/privacy; external→staleness/tool brittleness; latent→silent corruption.

### 6.2.2 Memory Structure (structure_t) — 어떻게 조직?
- **Flat**: 시간순 append-only. write cheap, trajectory replay에 유리. recency bias, 확장 시 truncation 의존 → 저수준 중복 누적.
- **Hierarchical**: 다중 추상화 단계. 압축·검색 균형. coarse-to-fine traversal.
- **Graph-based**: entity/concept/causal/relationship. 관계 추론·context 분해 해결.
- **Vector retrieval**: dense vector + cosine similarity. compact·fast, 의미 검색.
- Table 4: 시스템별 object×structure×CRUD×governance 매트릭스(AWM=hier explicit CRUD ✓, Mem0=graph/vector explicit, A-MEM=graph+vector, M+=flat implicit, ExpeL=vector explicit ...).

### 6.2.3 Memory Processing (IMPROVE_m = adaptive CRUD)
신호 S_t가 각 연산을 동적 조정:
- **C(reate)**: (1) semantic compression(메타데이터/요약/스키마) (2) context-aware discrete decision(add/update/delete/no-op; 인접 항 기반 중복·충돌 방지) (3) controlled boundary insertion(사용 시점 write 정책). 과잉 write=검색 noise, 과소 write=장기능력 손실.
- **R(read)**: (1) hybrid heuristics(의미+recency+importance) (2) structure-aware retrieval(coarse-to-fine 그래프/계층) (3) retrieval gating(질의 여부·context 양 동적 결정, token 절약) (4) retrieval-driven adaptation(과거 궤적을 case로 fetch → 모델 update 없이 행동 유도).
- **U(pdate)**: (1) scheduled review & attenuation(고효율 강화·구식 감쇠) (2) local refresh(삽입 시 위상 인접 갱신) (3) iterative distillation(반복 성공 → 압축 추상, 선택/치환) (4) offline aggregation(온라인 루프 외부 고비용 압축).
- **D(elete)**: (1) multi-stage pruning(write-time + 접근빈도/관련성 주기 정리) (2) consensus-based eviction(분산 투표로 공유 지식 보호) (3) tiered eviction(OS 영감 레이어별 규칙, size bound + 장기 일관성). 과잉 가지치기=핵심 지식 손실, 과소=노동 홍수.

### Signal-driven memory loop (Fig.8)
(i) Observe & Detect saliency → (ii) Create compact object → (iii) Organize into structure → (iv) Read on demand → (v) Plan & Act → (vi) Evaluate → S_t → (vii) Update/Delete. 정적 cache에서 **self-governing engine**으로.

---

## 5. Scaffolding — Tool & Full Scaffolding (Section 6.3, 6.4)

### 6.3 Tool = Tool Governance Metacognition (Eq.21)
```
T_{t+1} = IMPROVE_T( T_t ; S_t )
```
- **6.3.1 Dynamic Tool Routing**(선택·순서·조정): retrieval/graph 기반(MemTool pruning, TAR atomic↔agent 확장, VOYAGER/MetaAgent 궤적 인덱스, ToolNet/OrchDAG 의존 그래프) / policy-learning(AUTOACT, Tool-Star, ToolGen 토큰 통합 생성) / proactive·interactive(MCP-Zero, ASKTOACT 능동 발견/질의, Tool-Planner 로컬 repair).
- **6.3.2 Iterative Tool Refinement**(취약 program → 신뢰 skill; debugging+gatekeeping): VOYAGER generate-execute-revise 루프. (1) critique 특화(STELLA 전용 critic) (2) API 추상화(SkillWeaver, PyVision) (3) interface alignment(DRAFT: 코드 대신 문서 정제).
- **6.3.3 Autonomous Tool Creation**: 신규 실행 함수 합성. 재사용 procedural 지식 전환. 검증·문서화·통합 없으면 brittleness↑.

### 6.4 Full Scaffolding (Eq.22/23) — self-referential 최심부
```
Σ_{t+1} = IMPROVE_Σ(Σ_t ; S_t)        # Eq.22
Σ_{t+1} = I_{Σ_t}(Σ_t ; S_t)          # Eq.23  improver 자체가 현재 Σ 안에 구현 → self-referential
⟨Σ̃_{t+1}⟩ = exec(⟨Σ_t⟩; S_t)         # Eq.24  candidate program = patch Δ_t,  Σ̃=Σ⊕Δ
Σ_{t+1} = Σ̃  if V(Σ̃)=1  else Σ_t     # Eq.25  verifier gate(unit test/regression/safety)
```
Turing completeness → 이론상 연산 한계까지만 제한. AlphaEvolve(진화 coding agent, 과학 발견), ShinkaEvolve(소샘플 program 진화), ADAS(시스템 설계 공간 탐색), Self-Taught Optimizer(재귀 improver), Agent Symbolic Learning(기호 역전파로 prompt/tool/pipeline 동시 update), Gödel-machine 영감 RQGM.

---

## 6. Evaluation (Section 8)

- **평가 목표식 (Eq.26)**: m_t = E_{x∼D_eval, τ∼A_t(x)}[ Φ(x,τ) ]. 단일 종점 점수가 아닌 **반복 t에 따른 성능 trajectory** 추적(예산 b_t≤B_max).

| | Φ_metric (결정론 실행 evaluator) | Φ_judge (모델/rubric 기반) |
|---|---|---|
| | 단위테스트 등 형식 성공기준 | LLM/Agent-as-Judge로 장시간·open-ended 평가 |
| 한계 | 형식 기준 있는 task만 | judge latent bias로의 과최적화 위험 |

- **Metric-based 권고**: (1) 고정 예산 하 trajectory·분산·다중 seed 보고(plateau/regression) (2) improvement 신호와 안 겹치는 held-out D_eval(숨김/temporally-shifted)로 진 transfer 확인 (3) 자원 효율·human-in-the-loop 정량화(그 자체가 "self" 훼손) (4) regression률·tail-risk·safety 위반 추적.
- **Judge-based 안전장치**: judge 신원·rubric κ·예산 분리 공개; update 구동 judge ≠ 최종 보고 judge 독립(θ'_judge 또는 κ'); Φ_metric/인간으로 calibration.
- **Benchmark**: (8.2.1) mechanism benchmark(update channel·평가 interface 분리해 메커니즘 격리) / (8.2.2) domain benchmark(실제 환경 제약 하 instantiate). 권고 항목: baseline, 고정 예산 후 성능, learning curve, held-out transfer, regression률, cost 요약.

---

## 7. Discussion & Future Directions (Section 9)

### 9.1 시스템 설계 시사점
- **Fast exploration → slow consolidation 비대칭**: Σ(빠름, 가역) vs θ(느림, 크레딧 불투명). 노이즈 환경에서는 Σ 내 update로 제한·검증 후, 안정화되면 parametric consolidation(distillation/파인튜닝) 지연. 단 consolidation은 손실 압축 → 희귀 error-recovery 전략 폐기, θ update는 기존 safety bound 무효화(재대항 테스트 필요).
- **Critic as governed infrastructure**: 폐루프 critic은 공격 표면. agent가 critic 최적화 → shortcut 발견 유인. capability 상한 = critic exploit-resistance. 제안-수락 역할 분리(critic/generator 분리); 진화하는 critic은 단조 변경(가산적 테스트 생성) + 인간 감사 gate로 제한.
- **Safety through layered gating**: 동적 정렬 대상. full-scaffolding에서 prompt injection이 영속 아키텍처 취약점으로 진화(poisoned memory/hijacked tool → 안정 update commit). "자기개선 agent = 보호 runtime 내 신뢰할 수 없는 코드". 모든 구조 update는 verifier-gated(기능 정확성·tool 권한 경계·랜덤 섭동 robustness) 통과 후 commit.

### 9.2 Future Directions (두 bottleneck, 6 방향)
**Theme A: 종신 적응 알고리즘**
1. Test-Time continual adaptation: 배포 중 retrieval/routing/memory policy 동적 patch(국소 update가 장기 전역 성능 침식 방지).
2. Active exploration & curiosity: 자율적 가치 경험 탐색(예측오차/검증불일치에 내재 가치). 자기기만 퇴행 회피.
3. Parametric distillation & joint optimization: System2 구조 → 작은 모델 System1 가중치 증류; θ·Σ 동시 update 시 엄중 credit assignment(실패 시 prompt/tool/gradient 중 어느 수정판인지 자율 결정).

**Theme B: 복잡성·제약·open-world robustness**
4. Resource-constrained improvement dynamics: 비생산적 탐색 방지. 정적 peak → 효율 최적화 문제로 재구성(context 동적 할당, expensive neural 평가 앞 lightweight invariant gate, 낭비 반복 페널티).
5. Multi-agent cooperative co-evolution: 특화 agent 비모수 artifact(회귀 test/패치/tool wrapper) 공유·공동진화. 보안 버전관리 프로토콜(artifact repository).
6. Surviving open-world distribution drift: 정적 리더보드 → 비정상 시뮬레이터(API/UI drift). catastrophic forgetting·환경 비정상성 저항. 신경-computer: 고정 실행 interface → 학습 runtime 상태로 대체(연산·메모리·I/O 통합 adaptive neural runtime).

---

## 8. 역사적 맥락 (Section 2)
- 1790s-60s: 최소제곱법(Gauss) 오류구동 적응의 수학적 뿌리.
- 60s-80s: 기호·휴리스틱 자기수정.
- 80s-2000s: 연결주의·meta-learning 출현(Schmidhuber self-referential learning 1987, fast weights/linear Transformer, Gödel Machine 2003: 자기 코드를 기대효용 개선 *수학적 증명* 가능 시에만 rewrite, success-story algorithm incremental self-improvement 1994).
- 2000s-2020s: 형식·아키텍처 수준 자기개선.
- 2020s-현재: 확장 가능한 FM + agentic system. 자연어가 자기수정 검색공간 축소 → Schmidhuber "learning to think" controller가 세계모델을 prompt 하는 것이 현대 CoT 예시.
