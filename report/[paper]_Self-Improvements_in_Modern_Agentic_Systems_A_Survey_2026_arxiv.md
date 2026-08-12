> [paper] https://arxiv.org/abs/2607.13104

# Self-Improvements in Modern Agentic Systems: A Survey

## Summary & Outline

**한 줄 요약:** FM(LLM/VLM) 기반 agent가 스스로를 개선하는 모든 경로를 **"무엇을 수정하느냐(θ vs Σ)"** 와 **"개선 신호가 어디서 오느냐"** 두 축으로 정식화한 통합 survey. agent를 `A_t = (θ_t, Σ_t)`라는 설정(파라미터 + prompt·memory·tool·control logic 으로 이루어진 scaffold)으로 보고, 자기개선을 이 설정 위에서 동작하는 **self-induced update operator `U`** 로 수학적으로 정의한 뒤, (1) 파라미터를更新하는 *Foundation Model Improvement*(느리고 안정적, intrinsic demo/intrinsic feedback/extrinsic experience 3 신호)와 (2) scaffold를 바꾸는 *Scaffolding Improvement*(빠르고 가역적, prompt/memory/tool/full-scaffolding 4 부분)로 전체 문헌을 재배치한다. Schmidhuber의 self-referential learning·Gödel Machine 계보에서 현대 자연어 기반 agent까지 역사를 추적하며, 평가 프로토콜(metric/judge)과 안전(layered gating, critic as governed infrastructure)까지 망라한다. 본 분석 대상 repo의 agent memory / 자율 agent 문서들(Memora, AWM, Hermes, ReflectWorld-MM, ABot 등)을 하나의 프레임으로 정렬해주는 **메타 지도(map)** 역할.

**이 survey의 3가지 핵심 기여:**
1. **통합 정식화** — 흩어진 용어(self-correction, meta-prompting, self-play를 한 이름으로 묶기)를 하나의 수학 언어(`A_t=(θ_t,Σ_t)`, operator `U`, signal `S_t`)로 정리. 특히 FM fine-tuning과 scaffold工程을 **분리 주제가 아닌 동일 프레임의 두 pathway**로 취급(기존 survey들의 빈 곳).
2. **Skill = substrate에 직교하는 reusable update** — skill을 "한 번 쓰고 끝내는 action"이 아니라 **직렬화된 update operator** 로 모델화. tool/instruction/memory/가중치 어느 substrate에 저장되든 같은 store-retrieve 구조. meta-level skill(자기 config에 작용)이 곧 자기참조 루프의 재현.
3. **평가·안전 전면 통합** — 정적 zero-shot 점수가 아닌 **반복별 trajectory + 예산 + held-out transfer + regression률** 보고 요구, critic을 "수동 baseline이 아닌 공격 표면"으로 취급해 단조 진화·역할 분리 제안.

**논문 구조 outline:**
1. Introduction — 자기개선의 자기참조 본질, FM이 자연어로 검색공간을 줄인 의미, core+scaffold 아키텍처, 두 pathway
2. Historical Context & Theoretical Foundations — 1790s 최소제곱법 → 기호/연결주의/meta-learning → Gödel Machine·success-story algorithm → 현대 FM
3. Definitions — agent 정식화(Eq.1-3), 자기개선 operator(Eq.4-6), skill, RL/meta-learning/사고체제와의 관계
4. Taxonomy — 두 pathway 개요
5. Foundation Model Improvement — §5.1 intrinsic demo / §5.2 intrinsic feedback(rubric·consistency·corrective) / §5.3 extrinsic experience(grounded·simulated)
6. Scaffolding Improvement — §6.1 prompt / §6.2 memory(object·structure·CRUD) / §6.3 tool(routing·refinement·creation) / §6.4 full-scaffolding(자기참조)
7. Applications — SWE / web / game / scientific discovery / embodied / general computer control
8. Evaluation — §8.1 metric·judge 측정(Eq.26) / §8.2 mechanism·domain benchmark
9. Discussion — fast/slow 비대칭·critic 인프라·layered gating, future 6 방향
10. Conclusion

상세 발췌 → [excerpt](../source/paper/Self-Improvements_in_Modern_Agentic_Systems_A_Survey_2026_arxiv.md)

## Problem & Motivation

- **연구 배경**: FM 기반 agent가 실환경에 배포되면서 "경험으로부터 최소 인류 개입으로 스스로 진화(controllable evolution)"가 연구 프로토타입을 넘어 실제 과제가 됨. 자기개선은 본질적으로 **self-referential**(자기 최적화 메커니즘·운영 논리를 자율 검사·평가·수정) — Good의 "지능 폭발", Schmidhuber의 self-referential learning(1987)·Gödel Machine(2003, 기대효용 개선을 *수학적 증명*할 때만 자기 코드 rewrite) 계보.
- **풀고자 하는 문제**: 현대 self-improving agent 문헌은 용어·범위가 극도로 파편화(self-correction / meta-prompting / self-play 등 유사 아이디어가 서로 다른 이름). 기존 survey는 FM fine-tuning과 agent scaffolding을 **분리 주제**로 다루거나 신호·기질·평가·역사 중 일부만 다뤄 통일 시각 부재.
- **기존 접근의 한계**:
  - 과거 자기개선(Gödel Machine 등)은 어셈블리 코드·raw 가중치라는 광대 저수준 공간 탐색에 갇혀 scaling 불가.
  - 현대 survey들(Gao 2026, Fang 2025, Tao 2024)은 "무엇/언제/어떻게 진화" 또는 "정적 LLM 자율 학습"에 국한, FM 개선과 scaffolding을 통일 정식화로 묶지 않고, 역사적 뿌리 추적도 부족(Table 1 대비).

## Contributions

- **방법론/이론 기여**: `A_t=(θ_t,Σ_t)`·operator `U`·signal `S_t` 통합 정식화; skill=직렬화 update operator(→ substrate 직교); RL·meta-learning·사고체제와의 정합 매핑(θ update=표준 policy optimization, Σ update=MDP 자체 재형성의 structural meta-learning).
- **분류 체계 기여**: 두 pathway × 신호源(intrinsic demo/feedback/extrinsic) × scaffold 부분(prompt/memory/tool/full)의 체계적 taxonomy; memory를 object·structure·CRUD 3축으로, tool을 routing·refinement·creation 3축으로 분해.
- **실증/평가 기여**: 응용 6 도메인 배치; metric/judge 평가 프로토콜(Eq.26 trajectory 보고·held-out transfer·regression 추적·인간 개입 정량화); 안전 설계 원칙(critic governed infrastructure·layered gating·verifier gate Eq.25).
- **역사 기여**: 1790s 최소제곱법 → Gödel Machine → success-story algorithm → "learning to think"(2015, 세계모델을 prompt하는 controller=현대 CoT 선조)까지 지적 계보 정리.

## Method (Survey Framework)

### 기호 정의표 (Section 3 원문 기반)

| 기호 | 명칭 | 정의 (논문 원문) | 성질 |
|------|------|------------------|------|
| `t` | 시간 단계 | agent update의 반복 iteration 인덱스 | — |
| `A_t` | **agent configuration** (intrinsic) | 시점 t에서 agent의 고유 설정. `A_t=(θ_t,Σ_t)` (Eq.1). 단, Eq.3에서 `A_t`는 action을 가리키는 데 **동일 기호 재사용**에 주의(문맥으로 구분) | 영속 |
| `θ_t` (theta) | **FM 파라미터** | foundation model(LLM/VLM)의 신경 파라미터. stateless 추론 코어·general generative distribution 구현 | parametric, 느림·고비용·안정·전역 |
| `Σ_t` (Sigma) | **operational scaffold** | FM이 어떻게 conditioning·grounding·외부 연결되는지 지정하는 동적 scaffold. FM을 둘러싼 구조물 | non-parametric, 빠름·가역·context-dep |
| `p_t` | **prompt** | structured prompts / system instructions (Σ의 성분) | Σ 성분 |
| `m_t` | **memory** | memory mechanisms + retrieval/update policies (Σ의 성분) | Σ 성분 |
| `T_t` | **tools** | external tools 집합 + invocation interfaces (Σ의 성분) | Σ 성분 |
| `g_t` | **control logic** | routing·scheduling·safety constraints 등 추가 제어 논리 (Σ의 성분) | Σ 성분 |
| `X_t` | **ephemeral execution state** | KV cache·중간 계획·단기 working memory. interaction stream 처리 중 진화하지만 task 경계에서 reset → **intrinsic config가 아님** | 일시적(ephemeral) |
| `π_{θ_t,Σ_t}` | **induced policy** | 유도 정책. FM이 general 분포이지만 실현 행동은 θ와 Σ가 **공동** 결정 (Eq.3) | — |
| `A_t` (Eq.3) | **action** | 시점 t에서 시스템이 산출하는 행동. `X_t` conditioning이 진행 중 interaction의 일시적 맥락 포착 | 행동 |
| `U` | **self-induced update operator** | agent 실행(execution phase, `E`)과 갱신(update phase)으로 인자분해. intrinsic component(θ 또는 Σ)에 **durable** 변경 commit. 롤백 가능하나 장기 정책 안정화 | 영속 commit |
| `E` | **execution procedure** | agent가 유도 정책을 task context `C_t`에 대해 실행해 **학습 신호**(trajectory·reflection·critique·proposed edit)를 생성하는 절차. **`Σ_t`를 명시 인자로 받아 직접 self-inspection**(prompt 비판·tool audit) 허용 | 신호 생성 |
| `C_t` | **task context** | task 분포 / user interaction stream / self-play environment 등. `E`가 실행되는 맥락 | 환경 |
| `S_t` (§4) | **update signal** | operator `E`가 산출하는 갱신 신호. interaction trajectories·critiques·preferences·기타 자가생성 artifact. `IMPROVE_target(·;S_t)`의 구동 입력 | 신호 |
| `D_t` / `e_t` / `τ_t` | **신호源 3종** | 각각 intrinsic generative demonstrations / intrinsic evaluative feedback / extrinsic exploratory experience — `S_t`의 구체 형태 | FM pathway 신호 |
| `U_θ` | **parameter-learning procedure** | FM 파라미터 갱신 절차(policy gradient·on/offline RL·preference optimization·distillation 등). 학습 신호의 **데이터 분포는 agent 자기 정책**이 유도 | θ만 update |
| `U_Σ` | **scaffolding update mechanism** | 구조 변경 artifact(prompt edit·memory 재조직·tool interface 변경·신규 control routine)를 system-level에서 Σ에 commit | Σ만 update |
| `θ_{1:t}` / `Σ_{1:t}` | **history** | 파라미터/scaffold의 시계열 이력. validation·rollback(성능 저하 시 prior checkpoint 회귀) 허용 | 이력 |
| skill | **reusable update operator** | `U`의 named·직렬화된 재사용 인스턴스. substrate(`T`/`p`/`m`/`θ`/`g`) 중 하나에 저장 → **substrate 축과 직교**. object-level(task/world 작용, HRL option analog) vs meta-level(자기 config `A_t` 작용 → 자기참조 루프) | 직렬화 operator |
| `IMPROVE_target` | **abstract SI procedure** (§4 표기) | `IMPROVE_target(·;S_t)`. target ∈ {θ, Σ, p, m, T, Σ(full)}. `U`의 섹션별 별칭 | 추상 절차 |

### 통합 정식화 — agent = (파라미터, scaffold)

> FM은 본질적으로 stateless 추론 엔진이므로 자율성을 위해 영속·상호작용 scaffold Σ와 결합해야 한다. 아래 수식들은 이 "core+scaffold" 결합과 그 위에서의 자기개선을 정의한다.

**Eq.1 — agent configuration.** 시점 t의 agent 고유 설정은 FM 파라미터 `θ_t`와 operational scaffold `Σ_t`의 짝.
```
A_t = ( θ_t , Σ_t )        # θ=신경 파라미터(stateless 코어), Σ=scaffold. 둘이 함께 reason/plan/act 결정
```

**Eq.2 — scaffold 분해.** `Σ_t`는 FM을 어떻게 conditioning/grounding/외부 연결할지 지정하는 네 동적 성분.
```
Σ_t := ( p_t , m_t , T_t , g_t )   # p=prompt, m=memory(+검색/갱신 정책), T=tools(+호출 interface), g=control logic
```

**Eq.3 — induced policy.** FM 파라미터 `θ_t`는 general 생성 분포를 구현하지만, agent의 실현 행동은 `θ_t`와 `Σ_t`가 공동 결정. `X_t` conditioning은 진행 중 interaction의 일시적 맥락 포착(단, intrinsic이 아님).
```
π_{θ_t,Σ_t}( A_t | X_t )   # A_t=action(이 문맥). X_t=ephemeral state → task 경계 reset
```

**Eq.4 — self-improvement as self-induced operator.** 자기개선을 execution phase(`E`)와 update phase(`U`)로 인자분해.
```
A_{t+1} = U( A_{1:t},  E(π_{θ_t,Σ_t} ; Σ_t, C_t) )
#         ↑update rule      ↑execution: 정책을 C_t에 실행해 학습 신호 생성
#         A_{1:t}=이력(rollback용).  Σ_t 명시 인자 → 직접 self-inspection(prompt 비평/tool 감사) 허용
#         U는 intrinsic(θ/Σ)에 durable commit — 단순 X_t 진화(대화 누적)와의 분리선
```

핵심 통찰: `E`가 학습 신호를 만들 때 **`Σ_t`를 명시 인자로 받아 직접 self-inspection**(prompt 비판·tool audit)을 허용하는 점이 단순 `X_t` 진화(대화 누적)와의 분리선. `U`는 **durable** commit(롤백 가능하나 장기 정책 안정화). 두 가지 self-reference mode — (i) 분포 수준: 정책 실행→경험/보조 artifact→외부 optimizer가 θ update (FM improvement); (ii) 실행 수준 직접 수정: 정책 실행→prompt/memory/tool/control 직접 edit→Σ update (scaffolding improvement).

**Eq.5 / Eq.6 — 두 pathway.** 어느 intrinsic component를 수정하느냐로 분리.
```
FM (Eq.5):          θ_{t+1} = U_θ( θ_{1:t},  E(π;Σ_t,C_t) ),   Σ_{t+1}=Σ_t     # parametric: 느림·안정·전역·고비용
Scaffolding (Eq.6): Σ_{t+1} = U_Σ( Σ_{1:t},  E(π;Σ_t,C_t) ),   θ_{t+1}=θ_t     # non-parametric: 빠름·가역·국소
```
- **FM improvement**: `U_θ`는 policy-gradient/RL/preference optimization 등으로 θ만 갱신. 학습 신호의 데이터 분포는 agent 자기 정책이 유도. 느린 시간 규모, 막대한 계산 비용, 표현·일반화·능력의 안정적 전역 변화. tilt: 크레딧 귀속 불투명(regression 추적 어려움).
- **Scaffolding improvement**: `U_Σ`는 prompt edit/memory 재조직/tool interface 변경/신규 control routine을 Σ에 commit. θ 고정. 정책을 (i) conditioning context, (ii) 유효 action space(admissibility·tool schema), (iii) 실행 의미론(token→환경 action 구문 분석) 측면에서 재형성. 빠르고 가역적이지만 context-dependency 강함.

### 두 pathway와 신호源 (전체 분류도)

```
자기개선
├─ (1) FM Improvement  θ→θ'  [느린 루프]
│   ├ 5.1 Intrinsic Generative Demos   S≈D  (예시/합성데이터, self-instruct+consistency filter)
│   ├ 5.2 Intrinsic Evaluative Feedback S≈e  (rubric / consistency / corrective — env 추가상호작용 無)
│   └ 5.3 Extrinsic Exploratory Experience S≈τ (grounded env / simulated proxy env)
└─ (2) Scaffolding Improvement  Σ→Σ'  [빠른 루프]
    ├ 6.1 Prompt  p'=IMPROVE_p(p;S)   (scalar / qualitative / population / textual-gradient)
    ├ 6.2 Memory  m'=IMPROVE_m(m;S)   Eq.20  object × structure × CRUD
    ├ 6.3 Tool    T'=IMPROVE_T(T;S)   Eq.21  Tool Governance Metacognition
    └ 6.4 Full    Σ'=IMPROVE_Σ(Σ;S) = I_{Σ_t}(Σ;S)  Eq.22/23  self-referential
```

### (1) Foundation Model Improvement — θ_t → θ_{t+1}

θ만 갱신, Σ 고정(Eq.7, `IMPROVE_θ(θ_{1:t};S_t)`). agent가 자기 실행으로 학습 신호를 만들고 `U_θ`(policy gradient / on·offline RL / preference optimization / distillation)로 θ에 commit. 느리고 고비용이지만 전역·안정적. 신호源 3종으로 세분:

**§5.1 Intrinsic Generative Demonstrations (`S_t ≈ D_t`) — 스스로 학습 데이터를 만든다**
환경과 추가 상호작용 없이 agent가 *explicit training instance*(예시·합성·증강 데이터)를 직접 합성해 supervised-style 학습에 사용. 정답 추출이 어려운 도메인에서 agent가 "자기 학습 교재"를 자가 구축하는 경로.

| 시스템 | 생성 전략 | 핵심 특징 |
|--------|-----------|-----------|
| Self-Instruct (Wang 2022) | seed 확장 | 소수 예시에서 instruction-response 쌍 부트스트랩 |
| Evol-Instruct (Xu 2024a) | 복잡도 진화 | LLM이 instruction을 재작성해 난이도 점진 상승 |
| Self-Consistency filter (Huang 2023) | 합치 필터 | 다수 추론 경로 중 자기 합치하는 고신뢰 경로만 채택 |
| Verifier filtering (Singh 2024) | external verifier | 단위테스트 등 외부 검증기로 정답만 선별 |
| Curriculum (Simonds 2025) | 재귀 분해 | 복잡 문제를 단순 sub-problem으로 분해하는 학습 커리큘럼 |
| Diversity pooling (Qin 2025) | 다양성 유지 | 반복 시 출력 다양성 감소 대비 샘플 풀 확장 |
| TT-SI (Acikgoz 2025) | test-time 적응 | 추론 시 불확실성으로 약점 탐지 → LoRA 미세튜닝 |

보존: 정답 검증 어려운 도메인에서 자가 학습 데이터 구축; 한계: 가정 취약("correct→self-agreeing"), model collapse/pattern collapse → 정리 증명기 등 verification 형식체계 보강 필요.

**§5.2 Intrinsic Evaluative Feedback (`S_t ≈ e_t`) — 스스로 평가·판정하여 보상을 만든다**
환경 추가 상호작용 없이 rubric·consistency·corrective 신호를 자가 생성. `e_t`는 스칼라 보상 `r_t`·선호 쌍·자연어 critique 등으로 인스턴스화되어 reward model 학습·preference optimization·critique-conditioned fine-tuning 구동. 3가지 패밀리:
- *Rubric feedback* — 채점 기준·safety 원칙·constitutional 규칙·도메인 선호에 따라 산출을 비판·랭킹(constitutional AI, AI feedback→preference model). 산출은 기준하의 점수·랭킹·선호.
- *Consistency feedback* — FM의 확률적 동작을 이용, 다수결/자기합치로 보상·선호 구성(TTRL: 다수결 투표). "정답은 자기합치한다"는 *가정*이지 보장이 아님 → 취약; ego bias(자기 선호에 부합하는 산출에 과보상) 위험.
- *Corrective feedback* — 자연어 critique/수정 자체가 핵심 산출(점수 아님). error-correcting: 수정 제안 `y*_t`와 설명 `c_t`로 preference 신호 구성. 비교가 아닌 교정이지만, 여전히 자기 맹점·보상 hacking에 노출.

| 패밀리 | 시스템 | 신호 형태 | 핵심 특징 |
|--------|--------|-----------|-----------|
| Rubric | Constitutional AI (Bai 2022) | 선호(원칙 기반) | written principle로 비판·랭킹 → AI feedback → preference model |
| Rubric | Meta-Rewarding (Wu 2025b) | 선호+meta-판단 | 판단+meta-판단 모두 선호 쌍으로 변환해 반복 정렬 |
| Rubric | Self-Evolved Reward (Huang 2025a) | 학습된 보상모델 | 보상모델이 자가 라벨링하며 자기 개선 |
| Rubric | LLM-as-judge (Simonds 2025) | 점수 | 참조 해답 없이 LLM judge로 보상 생성 |
| Consistency | TTRL (Zuo 2025) | 다수결 보상 | test-time 다수결 투표 → RL 보상 |
| Consistency | SRT (Shafayat 2025) | 자기합치 | 정답 없이 self-consistency로 자기개선 |
| Consistency | EMPO (Zhang 2025h) | 엔트로피 신호 | 엔트로피 기반으로 추론 행동 장려 |
| Consistency | INTUITOR (Zhao 2025c) | 자기확신 | self-certainty를 내재 보상으로 사용 |
| Corrective | ReST-meets-ReAct (Aksitov 2024) | critique+개정 | agentic reasoning + self-training 결합 |
| Corrective | SELF (Lu 2024b) | 언어 feedback | critique/개정을 자기개선 신호로 변환 |
| Corrective | RISE (Qu 2024) | 재귀 self-reflection | 상호작용 기반 재귀 반성 → fine-tune |
| Corrective | Reflect-Retry-Reward (Bensal 2025) | 반성+재시도 | 실패 반성→재시도→보상으로 RL |
| Corrective | AlphaAllM (Tian 2024) | 탐색+critique | 탐색 중 모델 평가로 강한 훈련 신호 구축 |

> 안전장치: generator/evaluator 체크포인트 분리, 외부 anchor 유지, evaluator 불일치를 불확실성 신호로 활용.

**§5.3 Extrinsic Exploratory Experience (`S_t ≈ τ_t`) — 환경과 상호작용해 경험을 모은다**
agent가 실제 환경(또는 시뮬레이터)과 상호작용하며 trajectory를 수집, RL/자가플레이로 θ 갱신. §5.1/5.2(자기 생성 demo/평가)와 달리 *외부 실행 경험*이 신호.
- *5.3.1 Grounded task env interaction* — 실제 과제 환경(코드 실행기, 브라우저, 로봇)에서 grounded trajectory 수집. 성공/실패가 환경 객관 신호.
- *5.3.2 Simulated proxy env interaction* — 실환경 비용·위험 시 시뮬레이터/세계모델 proxy에서 경험. 자가플레이·모델링 기반.

| 모드 | 시스템 | 신호원 | 핵심 특징 |
|------|--------|--------|-----------|
| Grounded | Agent-RLVR (Da 2025) | 단위테스트 | pass/fail로 policy optimization, 성공/실패 program 대비 |
| Grounded | WebRL (Qi 2025) | 학습 보상모델 | web-navigation 궤적에 outcome-supervised 보상모델 자동 라벨 |
| Grounded | UI-Genie (Xiao 2025) | 보상모델 정제 | policy 학습 + reward-model refinement 동시, 성공/실패 step label |
| Grounded | MobileGUI-RL (Shi 2025b) | GRPO | 모바일 GUI GRPO, task 성공+실행효율 결합 보상 |
| Grounded | Absolute Zero (Zhao 2025a) | 자가 task + 실행 검증 | self-play로 task 생성, 실행 검증이 채택 결정 |
| Grounded | ETO (Song 2024c) | 성공/실패 대비 | 고정 환경에서 궤적 대비로 보수적 학습 |
| Simulated | WebEvolver (Fang 2025b) | 공진화 세계모델 | next web 관측 예측 →模拟 rollout로 policy 정제 |
| Simulated | WebSynthesis (Gao 2025b) | 학습 web WM | 가역적 search 기반 궤적 합성 |
| Simulated | WebDreamer (Gu 2025) | web 전이모델 | model-based planning으로 action 선택 안내 |
| Simulated | SPA (Chen 2025a) | 상태추정+전이모델 | self-play SFT로 모델 초기화·안정화 |
| Simulated | WMPO (Zhu 2025b) | pixel-space WM | 상상 rollout로 policy 최적화, 물리 시행착오 회피 |
| Simulated | GLoW (Kim 2026) | 구조화 기억 | dual-scale 텍스트 세계 기억, 100–800× 적은 환경 상호작용 |

> Algorithm 1: 신호 합집합 → filter/weight → `U_θ` 갱신 → checkpoint rollback. 위험: reward hacking(언어 verifier exploit), capability regression(좁은 보상 RL이 사전학습 역량 침식), hallucinated dynamics.

### (2) Scaffolding Improvement — Σ_t → Σ_{t+1}

Σ만 갱신, θ 고정(Eq.8, `IMPROVE_Σ(Σ_{1:t};S_t)`). frozen FM을 둘러싼 구조(prompt·memory·tool·control)를 바꿔 정책의 관측·행동 의미론을 재형성. 빠르고 가역적. scaffold 성분별 4분야:

**§6.1 Prompt — `p_{t+1}=IMPROVE_p(p_{1:t};S_t)` (prompt optimization loop)**
prompt 최적화를 애드훅 관행에서 신호 기반 개선 루프로 승격. 신호가 풍부해질수록(prompt loop Fig.1: init p_0 → execute τ → evaluate S → update Δ → apply p_{t+1}) 덜 휴리스틱·더 표적적. 4 패러다임:
- *6.1.1 Scalar-Feedback Optimization* — 스칼라 점수 신호로 prompt 탐색(OPRO 등).
- *6.1.2 Qualitative-Feedback Refinement* — 정성 critique로 prompt 정제(Self-Refine, Reflexion).
- *6.1.3 Population-Based Evolution* — prompt 집단을 진화/선택(PromptBreeder 등).
- *6.1.4 Textual Gradient Optimization* — "텍스트 gradient"(자연어 방향 지시)로 gradient ascent 흉내(TextGrad 등).

| 패러다임 | 시스템 | 신호 `S_t` | 장점 | 한계 |
|----------|--------|-----------|------|------|
| Scalar | RLPrompt, BBT, APE, OPRO, DSPy | 스칼라 점수 | 모델 무관·배포 간단 | 해석성 낮음·샘플 비효율 |
| Scalar | InstructZero, BPO | 스칼라(선호) | 내부 접근 불필요 | 검색 민감 |
| Qualitative | Self-Refine, Reflexion, Critic | 텍스트 critique | 해석 가능 edit·표적 교정 | critique 노이즈·drift |
| Qualitative | ACE | 텍스트 critique | 재사용 가능 feedback | validator 의존 |
| Population | Promptbreeder, STOP, GPTSwarm | 선택 신호 | 강 탐색·다양성 유지 | compute 집중·population drift |
| Population | AutoDAN, Evol-Instruct, GEPA | 선택 신호 | local optimum 탈출 | fitness 도메인 특화 |
| Textual gradient | APO, TextGrad, metaTextGrad, SkillOpt | 텍스트 gradient | 방향성 update·자동화 | gradient 취약·LLM 품질 의존 |

> Takeaway: 신호가 스칼라→정성→집단→구조적 방향 지시로 진화하며 자동·샘플효율적.

**§6.2 Memory — `m_{t+1}=IMPROVE_m(m_t;S_t)` (Eq.20) ─ 본 repo 주토픽과 직결 ★**
메모리를 수동 저장이 아닌 **능동 진화 scaffold**(self-governing engine)로 재정의. frozen FM 가정 하 비모수 외부 메모리만 다룸(가중치 내 parametric memory 제외). 3축으로 분해:

```
m_t := (object_t, structure_t);  m_{t+1}=IMPROVE_m(m_t;S_t)   # signal-driven CRUD
```

- *6.2.1 Memory Object (object_t) — 무엇을 저장?*
  - **Explicit**(가독성·감사 우세, scalability 비용): (i) *processed interaction trails*(raw 궤적→재사용 의미 단위 압축: routine/heuristic/reflection, generalization ↑); (ii) *curated raw content*(요약손실 큰 정확 surface 보존: 코드/수식/스크린샷, precision ↑); (iii) *integrated external knowledge*(외부 저장소 사실 통합·동적 갱신, verifiability ↑).
  - **Implicit**(compact·고속 associative, debug/감사 어려움, representation drift): latent token·hidden state·KV cache 증강.
  - Table 3 trade-off: processed→summary bias/stale; raw→retrieval noise/privacy; external→staleness/tool brittleness; latent→silent corruption.

- *6.2.2 Memory Structure (structure_t) — 어떻게 조직?*
  - **Flat**: 시간순 append. write cheap·trajectory replay 유리. recency bias·확장 시 truncation 의존.
  - **Hierarchical**: 다중 추상화 단계. coarse-to-fine traversal로 압축·검색 균형.
  - **Graph**: entity/concept/causal/relationship. 관계 추론·context 분해.
  - **Vector**: dense vector + cosine. compact·fast 의미 검색.
  - Table 4: 시스템별 object×structure×CRUD 매트릭스(AWM=hier explicit, Mem0=graph+vector, A-MEM=graph+vector, ExpeL=vector ...).

- *6.2.3 Memory Processing (IMPROVE_m = 신호 기반 adaptive CRUD)* — 신호 `S_t`가 각 연산을 동적 조정:
  - **C(reate)**: (1) semantic compression(메타데이터/요약/스키마) (2) context-aware discrete decision(add/update/delete/no-op; 인접 항 기반 중복·충돌 방지) (3) controlled boundary insertion(사용 시점 write 정책). 과잉 write=검색 noise, 과소 write=장기능력 손실.
  - **R(ead)**: (1) hybrid heuristics(의미+recency+importance) (2) structure-aware retrieval(coarse-to-fine 그래프/계층) (3) retrieval gating(질의 여부·context 양 동적 결정, token 절약) (4) retrieval-driven adaptation(과거 궤적을 case로 fetch → 모델 update 없이 행동 유도).
  - **U(pdate)**: (1) scheduled review & attenuation(고효율 강화·구식 감쇠) (2) local refresh(삽입 시 위상 인접 갱신) (3) iterative distillation(반복 성공→압축 추상) (4) offline aggregation(온라인 외부 고비용 압축).
  - **D(elete)**: (1) multi-stage pruning(write-time+접근빈도 주기 정리) (2) consensus-based eviction(분산 투표 공유 지식 보호) (3) tiered eviction(OS 영감 레이어별 규칙). 과잉=핵심 손실, 과소=노동 홍수.

- *Signal-driven memory loop (Fig.8)*: observe→create→organize→read→plan/act→evaluate(`S_t`)→update/delete. 정적 cache에서 **self-governing engine**으로.

| 시스템 | Object | Structure | 핵심 특징 |
|--------|--------|-----------|-----------|
| Self-Notes (2023) | explicit | flat | 추론 중 inline으로 transient insight 직접 기록 |
| Generative Agents (2023) | explicit | flat+vector | 의미+recency+importance hybrid 검색 |
| AWM (2024) | explicit | hier | 과거 경험에서 workflow 추출·재사용(snowball) |
| Reflexion (2023) | explicit | flat | 실행 오류→자연어 반성→다음 시도 조건 |
| ExpeL (2024) | explicit | vector | 성공 궤적에서 insight 추출·재사용 |
| ReadAgent (2024) | explicit | flat | 긴 문서 bookmark+摘要로 압축 |
| SAGE (2025) | explicit | hier | 계층적 memory로 long-horizon |
| Mem0 (2025) | explicit | graph+vector | add/update/delete/no-op discrete 결정 |
| A-MEM (2025) | explicit | graph+vector | 자율 메모리 구성·인접 갱신 |
| MemInsight (2025) | explicit | hier+vector | 경험 자동 분류·구조화 |
| ACE (2025) | explicit | vector | retrieval-driven adaptation |
| Reasoning Bank (2025) | explicit | vector | 추론 case 저장·활용 |
| MemGen (2025) | implicit | flat | latent 생성 메모리 |
| M+ (2025) | implicit | flat | updatable latent pool, 영속 상태 추적 |
| MemoryLLM (2024) | implicit | flat | latent state reconstruction |
| M3-Agent (2025) | explicit | hier | 멀티모달 3-level memory |
| H-MEM (2025) | explicit | hier | 계층적 진화 메모리 |
| SALM (2025) | explicit | hier | 구조화 평생 메모리 |
| PRIME (2025) | explicit | vector | RL 통합 case 기반 |
| CodeAgent (2024) | explicit | vector | 코드 도메인 메모리 |
| HGM (2025–26) | explicit | graph | 계층 그래프 메모리 |
| MovieChat (2024) | implicit | hier+vector | 비디오 스트림 메모리 |

**§6.3 Tool — `T_{t+1}=IMPROVE_T(T_t;S_t)` (Eq.21) — Tool Governance Metacognition**
정적 사전 큐레이션 toolkit에서 자율 적응·발견·통합으로 전환. agent가 tool의 필요성·효용·신뢰성을 메타인지적으로 추론. 3 차원:
- *6.3.1 Dynamic Tool Routing*(선택·순서·조정): retrieval/graph 기반(MemTool pruning, TAR atomic↔agent 확장, VOYAGER 궤적 인덱스, ToolNet 의존 그래프) / policy-learning(AUTOACT·ToolStar·ToolGen 토큰 통합) / proactive·interactive(MCP-Zero 능동 발견·Tool-Planner 로컬 repair).
- *6.3.2 Iterative Tool Refinement*(취약 program→신뢰 skill; debugging+gatekeeping): VOYAGER generate-execute-revise. (1) critique 특화(STELLA 전용 critic) (2) API 추상화(SkillWeaver·PyVision) (3) interface alignment(DRAFT: 코드 대신 문서 정제). 불안정 tool이 재사용되면 future 행동 부패 → gatekeeping 핵심.
- *6.3.3 Autonomous Tool Creation*: 신규 실행 함수 합성. 재사용 procedural 지식 전환. 검증·문서화·통합 없으면 brittleness↑.

| 차원 | 시스템 | 메커니즘 | 핵심 특징 |
|------|--------|----------|-----------|
| Routing | MemTool (Lumer 2025a) | 풀 가지치기 | lightweight operational memory로 routing degrade 방지 |
| Routing | TAR (Lumer 2025b) | 검색 단위 확장 | atomic API ↔ competent agent 동적 전환 |
| Routing | VOYAGER (Wang 2023) | 궤적 인덱스 | tool-use 궤적을 procedural memory로 인덱싱 |
| Routing | MetaAgent (Qian 2025) | 궤적 인덱스 | 구조적 지식으로 routing 정제 |
| Routing | ToolNet (Liu 2024c) | 의존 그래프 | tool 전이를 방향 그래프로 모델링 |
| Routing | OrchDAG (Lu 2025) | 의존 그래프 | 다단계 feasibility·전제조건 고려 |
| Routing | MassTool (Lin 2025) | 의미 매칭+그래프 | 거대 토폴로지에서 고정밀 탐색 |
| Routing | AUTOACT (Qiao 2024b) | SFT 정책 | 합성 궤적으로 정책 부트스트랩 |
| Routing | Tool-Star (Wang 2025g) | SFT+RL | 도구 선택 정책 학습 |
| Routing | DeepEyesV2 (Hong 2026) | SFT 정책 | 멀티모달 도구 라우팅 |
| Routing | MCP-Flow (Dong 2025) | SFT 정책 | MCP 기반 도구 흐름 |
| Routing | ToolGen (Wang 2025f) | 토큰 통합 | 검색+선택+호출을 단일 생성으로 통합 |
| Routing | AGENTFLOW (Li 2026b) | 보상 shaping | sparse reward로 계획·탐색 |
| Routing | SPORT (Li 2025a) | 선호 신호 | 도구 선택 multi-objective 정렬 |
| Routing | AutoTIR (Wei 2025c) | multi-objective | action-level attribution |
| Routing | MCP-Zero (Fei 2025) | 능동 발견 | 모호 시 도구 발견/질의 능동 개시 |
| Routing | ASKTOACT (Zhang 2025l) | 능동 질의 | ambiguity → 학습 신호로 자기교정 |
| Routing | Tool-Planner (Liu 2025e) | 로컬 repair | 도구를 상호 교체 가능 kit로 클러스터링 |
| Routing | ToolACE-R (Zeng 2025) | 난이도 보정 | 과제 난이도별 revision effort 동적 조정 |
| Refinement | VOYAGER (Wang 2023) | generate-execute-revise | 오류 trace+환경 feedback→수정 루프 |
| Refinement | STELLA (Jin 2025) | 전용 critic | 도메인 특화 critic agent로 진단 강화 |
| Refinement | SkillWeaver (Zheng 2025a) | API 추상화 | 상호작용 → 재사용 web tool 증류 |
| Refinement | PyVision (Zhao 2025b) | API 추상화 | Python program 정제로 multimodal 안정화 |
| Refinement | DRAFT (Qu 2025) | interface alignment | 코드가 아닌 tool *문서* 정제로 의미 갭 해소 |
| Creation | RQGM / Gödel-agent (2025–26) | 자기참조 생성 | 신규 도구 합성 + 코드 자기수정 |

**§6.4 Full Scaffolding — `Σ_{t+1}=IMPROVE_Σ(Σ_t;S_t)` = 자기참조 최심부**
개별 성분이 아닌 전체 operational logic/codebase를 mutable 기질로. improver 자체가 현 Σ 안에 구현되어 self-referential(Eq.23).
```
⟨Σ̃_{t+1}⟩ = exec(⟨Σ_t⟩; S_t)        # candidate = patch Δ, Σ̃=Σ⊕Δ          Eq.24
Σ_{t+1} = Σ̃ if V(Σ̃)=1 else Σ_t      # verifier gate(unit test/regression/safety) Eq.25
Σ_{t+1} = I_{Σ_t}(Σ;S)               # improver 자체가 현 Σ 안 구현 → 자기참조      Eq.23
```
Turing completeness 한계 내 가장 넓은 설계 공간 탐색.

| 시스템 | 메커니즘 | 핵심 특징 |
|--------|----------|-----------|
| AlphaEvolve (Novikov 2025) | 진화 coding agent | 평가자 feedback로 알고리즘 반복 개선, 과학 발견 가속 |
| ShinkaEvolve (Lange 2026) | LLM program 진화 | 소샘플 탐험-이익 균형 + bandit LLM 앙상블 |
| ADAS (Hu 2025) | 설계 공간 탐색 | 평가 함수 최대화 agent system 설계 탐색 |
| EvoFlow | 진화 workflow | 이질적 workflow Pareto set, 비용-성능 균형 |
| Self-Taught Optimizer (Zelikman 2024) | 재귀 improver | 자기 program+효용함수로 복수 후보 생성→최고 점수 선택 |
| Agent Symbolic Learning (Ou 2025) | 기호 역전파 | prompt/tool/pipeline을 "가중치"로 보고 기호 gradient descent |
| Gödel Agent (2025) | Gödel machine 영감 | 자기 코드를 runtime에서 검증·개선하는 자기참조 루프 |
| DGM (2025) | Darwinian 자기수정 | 코드 패치 생성·평가·채택의 진화적 자기개선 |
| Alita (2025) | 계층 자기구성 | 역할 분해·도구 자기 생성 |
| RQGM (2025–26) | 자기참조 도구 | 도구·코드 자기 합성 루프 |
| GenericAgent (2026) | 범용 진화 | 전체 scaffold를 mutable 기질로 |
| ContinualHarness (2026) | 지속 harness | harness 자체를 횡단 반복 개선 |
| AgentDevel (2026) | 개발 자동화 | agent 설계·개발 코드 자동 개선 |
| CoEvoSkill (2026) | 공동진화 도구 | 도구-에이전트 공동 진화 |
| DecentMem (2026) | 분산 메모리 | 다중 agent 메모리 공유·합의 |
| Evo-Memory (2026) | 진화 메모리 | 메모리 구조 자체를 진화 대상으로 |
| SePO (2026) | scaffold 최적화 | 전체 scaffold search+평가 |
| MemRL (2026) | RL 기반 메모리 | RL로 메모리 정책 학습 |

> 위험: prompt injection이 영속 아키텍처 취약점으로 진화(poisoned memory → 안정 commit). 모든 patch는 verifier gate(Eq.25) 통과 필요.

## Survey Scope, Evaluation & Coverage

> 본 문서는 단일 메서드 논문이 아닌 survey이므로, "Experiments"를 문헌 커버리지·평가 프로토콜 제안·repo 문서 정렬로 대체한다.

### 커버리지
- 97페이지, Fig.2 timeline에 2023–2026 대표작 80+ 배치(FM lane: Self-Instruct·SEAL·TTRL·Q-Evolve·SAGE... / Scaffold lane: Voyager·Reflexion·AWM·Mem0·A-MEM·STELLA·ADAS·AlphaEvolve...).
- Table 4(memory 시스템 object×structure×CRUD 매트릭스), Table 1(선행 survey 대비 강조축) 등 정성 scorecard 제공.

### 평가 프로토콜 제안 (§8)
| 측정 | Φ_metric(결정론: 단위테스트 등) | Φ_judge(LLM/Agent-as-Judge + rubric κ) |
|---|---|---|
| 핵심 요구 | 고정 예산 `b_t≤B_max` 하 **trajectory+다중 seed 분산**, held-out `D_eval`(숨김/시간이동) transfer, regression률·safety 위반 추적, cost+인간개입 정량화 | judge 신원·rubric·예산 분리 공개; update 구동 judge ≠ 최종 보고 judge(독립 `θ'_judge`/`κ'`); Φ_metric/인간으로 calibration |
| 위험 | 형식 기준 있는 task만 | judge latent bias로 과최적화 |

`m_t = E[Φ(x,τ)]` (Eq.26) — 단일 종점이 아닌 반복 trajectory로 평가.

### 본 repo 기존 분석과의 정렬 (Topics 매칭)
survey 프레임으로 repo 문서들을 동일 축에 배치 가능:
- **FM improvement (θ)**: Draft-OPD·Speculative KD는 파라미터 증류/학습轨道交通 — §5.3/Theme A-3에 해당.
- **Scaffolding-Memory (Σ·m)**: Memora(harmonic rep, cue index, GRPO 검색)·ReflectWorld-MM(episodic/semantic/procedural, evolving entity)·ABot(graph memory, lifelong self-evolution)·AWM(workflow 추출 snowball) — 모두 §6.2 object/structure/CRUD로 정렬. 특히 ABot의 failure-driven promotion gate, ReflectWorld의 evolving semantic(A/U/D)은 survey가 강조하는 **CRUD Update/Delete** 의 구체 인스턴스.
- **Scaffolding-Full (Σ)**: Hermes·OpenClaw의 background review(dreaming)+curator 스킬 통합은 §6.4 self-referential + meta-level skill의 실례.
- **평가/long-horizon**: AgenticSTS(memory contract, typed retrieval)·Remember-When(proactive memory intervention)·Long-Horizon-Terminal-Bench는 §8 trajectory 평가·§9.2-1 test-time adaptation과 직결.

## Analysis

### Strengths & Significance
- **정합성**: 분산 용어를 `U`/`S`/substrate 한 언어로 통일. θ vs Σ 이분법이 단순 prompt 튜닝과 파라미터 RLHF를 동일 스펙트럼에 올려 비교 가능하게 함.
- **skill 직교성 모델**: substrate(T/p/m/θ/g)와 무관한 "직렬화 update operator" 정의가 tool routing과 memory 조직에서 같은 store-retrieve 패턴이 재등장하는 이유를 설명 — 개념적 인사이트.
- **안전 전면성**: 자기개선을 "보호 runtime 내 신뢰할 수 없는 코드"로 프레이밍, critic을 공격 표면으로, full-Σ의 영속 취약점(poisoned memory/hijacked tool) 경고는 단순 성능 survey와 차별.
- **역사적 깊이**: Gödel Machine·success-story algorithm·"learning to think"까지 계보 추적이 현대 자기참조 루프의 정당성·한계를 조명.

### Limitations
- **정성 평가 중심**: Table 3/4가 "literature-grounded 추세" scorecard로 단일 표준 벤치마크 기반 아님 — 메모리 object/structure 간 head-to-head 수치 비교 부재.
- **구현 디테일 얕음**: survey라 본질이지만, CRUD 패턴(예: hybrid retrieval 가중치·gating 임계·eviction 규칙)을 코드 수준이 아닌 패턴 카테고리로만 서술 → 본 repo의 스니펫 수준 분석(Memora RRF, Hermes dreaming 점수)과 보완 필요.
- **FM↔Scaffold joint optimization 미해결**: §9.2-3가 "어느 수정판(prompt/tool/gradient)인지 자율 결정"을 미래 과제로만 — θ·Σ 동시 루프의 credit assignment는 공식으로 제시되지 않음.
- **안전 게이트 형식화 부족**: layered gating·verifier gate(Eq.25)를 제안하나, "단조 critic 진화·인간 감사 trail"이 원칙적 권고에 그침.

### Future Work / Improvements
- Theme A(종신 적응): test-time continual adaptation·active exploration/curiosity·parametric distillation & joint θ-Σ optimization.
- Theme B(robustness): 자원 제약 improvement dynamics·multi-agent cooperative co-evolution(artifact repository)·open-world distribution drift(비정상 시뮬레이터, adaptive neural runtime으로 실행 interface 자체를 학습).
- 확장: 본 repo 문서들을 이 survey의 taxonomy로 재태깅해 `topics/self_improving_agents.md` 통합 비교 문서 구성 가능(FM lane vs scaffold lane·memory CRUD 차원 비교표).

## References
- 원본: [arXiv:2607.13104](https://arxiv.org/abs/2607.13104) · [Project page § Self-Improving-Agents](https://github.com/Self-Improving-Agents)
- 역사 계보: Schmidhuber self-referential learning (1987), Gödel Machine (2003), success-story algorithm (1994), "learning to think" (2015)
- 본 repo 관련(동일 taxonomy로 정렬): [Memora](../report/[paper][git]_Memora_A_Harmonic_Memory_Representation_Balancing_Abstraction_and_Specificity_2026_ICML.md) · [ReflectWorld-MM](../report/[paper]_ReflectWorld-MM_An_Entity-Oriented_Multimodal_Memory_System_for_Open-Ended_Video_Streams_2026_Rightly_Robotics.md) · [ABot-AgentOS](../report/[paper]_ABot-AgentOS_A_General_Robotic_Agent_OS_with_Lifelong_Multi-modal_Memory_2026_Alibaba.md) · [AWM](../report/[paper][git]_Agent_Workflow_Memory_2024_arxiv.md) · [Hermes](../report/[git]_hermes-agent_NousResearch.md) · [OpenClaw](../report/[git]_openclaw_openclaw.md) · [AgenticSTS](../report/[paper]_AgenticSTS_A_Bounded-Memory_Testbed_for_Long-Horizon_LLM_Agents_2026_arxiv.md) · [Remember-When](../report/[paper]_Remember_When_It_Matters_Proactive_Memory_Agent_for_Long-Horizon_Agents_2026_Meta_AI.md)
