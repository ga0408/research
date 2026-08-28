> [paper][git] https://github.com/Gen-Verse/Recuris.git · https://arxiv.org/abs/2608.24876

# Recursive Experiential–Working Memory Evolution for Long-Horizon Agent Harnesses

## Summary & Outline
- **한 줄 요약**: 장기 과제(long-horizon tasks)에서 상호작용 히스토리가 길어짐에 따라 발생하는 상태 유실 및 스킬 오호출 문제를 해결하기 위해, 작업 상태를 추적·검증하는 작업 기억(Working Memory)과 재사용 가능한 스킬을 제공하는 경험 기억(Experiential Memory)을 결합하고, 구조화된 실행 추적(Structured Trace) 기반 고장 국소화(Failure Localization)와 결정론적 검증 게이트를 통해 스킬 메모리 메커니즘을 경계 내에서 자기 진화시키는 자율형 에이전트 하네스 프레임워크 **Recuris**.
- **핵심 아키텍처**:
  - `M_k = (E_k, W_k, ρ_k, C_k)` 4중 튜플로 정형화된 스킬 메모리 (경험 기억 `E`, 작업 기억 명세 `W`, 호출 정책 `ρ`, 검증 체커 `C`).
  - 과제 내(Within-Task) 루프: `w_t → ρ_k(w_t, e_t) → a_t → o_t → c_t → w_{t+1}`의 상태 기반 스킬 호출 및 증거 기반 상태 갱신.
  - 과제 간(Cross-Task) 루프: 구조화된 궤적 `Γ_k`를 통한 결함 부품 특정(`Z_k`), 국소 컴포넌트 패치(`⊕_{Z_k}`), 보류 개발셋 기반 결정론적 페어드 부트스트랩 게이트(`G_fixed`).
- **주요 성과**: 4개 장기 벤치마크 및 10개 언어 모델(3B 경량 오픈소스부터 GPT-5.6 Sol, Claude Opus 5 등 프론티어 모델) 대상 37개 완료 실험 중 35개에서 성능 향상 달성. τ²-Bench에서 GPT-5.6 Sol +17.8점(76.1%), Claude Opus 5 +15.6점(87.9% SOTA) 도달. 상호작용 지평선이 길어질수록 베이스라인과의 격차가 최대 +32.2점까지 확대.

---

## Problem & Motivation
- **연구 배경**:
  LLM 기반 자율 에이전트는 복잡한 환경에서 추론, 도구 활용 및 자율 실행을 수행하며 급격히 발전하고 있다. 이러한 에이전트 시스템은 모델 본체뿐만 아니라 메모리, 상태 추적, 스킬 호출, 도구 상호작용, 검증 계층을 포괄하는 '에이전트 하네스(agent harness)'에 의해 통제된다. 그러나 상호작용 단계가 수십~수백 턴에 달하는 장기 실행(long-horizon execution) 환경에서 에이전트가 스스로 경험을 축적하여 점진적으로 성능을 개선하는 재귀적 자기개선(Recursive Self-Improvement, RSI)을 달성하는 것은 여전히 난제로 남아있다.
- **풀고자 하는 문제 (Task)**:
  - **Long-Horizon Agent Execution & Self-Improvement**: 긴 턴 수와 복잡한 비즈니스 로직, 상태 변경 도구 호출을 수반하는 복합 과제 환경에서, 에이전트가 목표를 유실하지 않고 올바른 절차를 실행하며, 실패 경험으로부터 통제 가능한 방식으로 하네스를 자기 갱신하는 문제.
- **기존 접근의 한계**:
  1. **초기 지시문 또는 전체 히스토리 기반 검색의 한계**: 기존 경험 기억(Experiential Memory) 방식은 초기 지시문(prompt)이나 대화 전체 히스토리로부터 스킬을 검색한다. 그러나 장기 실행 중에는 초기 지시문이 현재 해결해야 할 서브태스크와 일치하지 않으며, 거대해진 히스토리는 완료된 단계, 폐기된 정보, 실행 노이즈가 뒤섞여 있어 정확한 스킬 인출을 방해한다.
  2. **조잡한 실패 신호와 고장 원인 특정 부재**: 기존 자기개선 시스템은 최종 과제 성공/실패 여부(바이너리 보상)만으로 프롬프트나 메모리를 통째로 재작성한다. 이는 실패가 누락된 스킬 때문인지, 잘못된 상태 추적 때문인지, 늦은 스킬 호출 때문인지, 잘못된 완료 판정 때문인지 구분하지 못해 메모리 전체를 훼손하는 부작용(catastrophic regression)을 낳는다.
  3. **암묵적 상태 추적과 자의적 완료 판단**: 에이전트 모델이 스스로 "작업을 완료했다"고 주장하는 경우 환경에 실제 쓰기(write) 도구를 호출하지 않고도 과제를 임의 종료해 버리는 '자기 선언적 완료 환각'이 빈번하게 발생한다.

---

## Contributions
- **상태 기반 메모리 활용(State-Grounded Memory Use) 원칙 정립**: 장기 에이전트 하네스에서 진화하는 작업 상태와 저장된 경험 지식을 정밀하게 일치시키는 동적 작업 기억(Working Memory) 인터페이스의 필수성을 규명.
- **Recuris 아키텍처 제안**: 영속적 경험 지식(`E`)과 검증된 작업 상태(`W`), 이벤트 기반 호출 정책(`ρ`), 결정론적 체커(`C`)를 결합한 모듈형 스킬 메모리 `M = (E, W, ρ, C)` 및 실행 하네스 설계.
- **구조화된 실행 추적(Structured Trace) 기반 고장 국소화 및 Bounded RSI 구현**: 실행 과정을 `Γ = (w_t, E_t, a_t, o_t, w̃_{t+1}, c_t, w_{t+1})`의 구조화된 증거로 기록하여, 실패 원인을 특정 메모리 부품으로 국소화(`A_fixed`)하고 외부 코드 게이트(`G_fixed`)를 통해 안전하게 진화시키는 제한적 재귀 루프 확립.
- **광범위한 벤치마크 및 크로스 모델/과제 전이성 입증**: 4개 벤치마크(τ²-Retail, τ²-Airline, SkillFlow, Terminal-Bench 2.1)와 10개 모델(Open-weight 3B~35B, GPT-5.6 Sol, Claude Opus 5 등)에서 전반적인 성능 개선 및 단일 모델(doubao-seed-2-0-pro)에서 진화된 메모리의 타 모델 무수정 전이 유효성 실증.

---

## Architecture & System Workflow

Recuris는 과제 내부의 **결정론적 실행 루프(Invariant Turn Loop)**와 과제 간의 **제한적 진화 루프(Bounded Evolution Loop)**의 2계층으로 구성된다.

```
========================================================================================
                          RECURIS DUAL-LOOP ARCHITECTURE
========================================================================================

 [ WITHIN-TASK EXECUTION LOOP ]
  
   User / Env Msg ──> [ 1. Grounding ] ── (Tool Receipts) ──> Update Ledger (DONE)
                            │
                            ▼
                     [ 2. TURN_START ] ── (ρ: boundary) ────> Deliver E_t
                            │
                            ▼
                     [ 3. WM Update ] ── (from Input) ─────> Drafted w̃_{t+1}
                            │
                            ▼
                     [ 4. INTENT_REC ] ── (ρ: intent) ──────> Deliver E_t
                            │
                            ▼
                     [ 5. Render WM ] ── (Status Board) ───> State Context w_t
                            │
                            ▼
                     [ 6. LLM Draft ] ── (Base Model π_θ) ──> Draft Action a_t
                            │
                            ▼
                     [ 7. DRAFT_READY] ── (C: Checkers) ────> Bounce / Redraft
                            │
                            ▼
                     [ 8. Commit ] ───── (Tool Dispatch) ───> Issue a_t
                            │
                            ▼
                     [ 9. PRE_WRITE ] ── (ρ: call-time) ────> Deliver Tool Skill
                            │
                            ▼
                     Structured Trace Γ_k = {(w_t, E_t, a_t, o_t, w̃_{t+1}, c_t, w_{t+1})}
                                                                   │
                                                                   │ (Failed Runs)
===================================================================│====================
 [ CROSS-TASK RECURSIVE EVOLUTION LOOP ]                           ▼
                                                         ┌──────────────────┐
                                                         │ Trace Diagnosis  │
                                                         │   A_fixed(Γ_k)   │
                                                         └─────────┬────────┘
                                                                   │ Failures f_j &
                                                                   │ Target Components Z_k
                                                                   ▼
                                                         ┌──────────────────┐
                                                         │ Component Patch  │
                                                         │  M_k ⊕_{Z_k} P   │
                                                         └─────────┬────────┘
                                                                   │ Candidate M_k^+
                                                                   ▼
                                                         ┌──────────────────┐
                                                         │ Validation Gate  │
                                                         │   G_fixed(D_dev) │
                                                         └─────────┬────────┘
                                                                   │ Paired Bootstrap CI > 0
                                                                   │ & n_regressed ≤ reg_cap
                                                                   ▼
                                                         ┌──────────────────┐
                                                         │ Evolved Memory   │
                                                         │      M_{k+1}     │
                                                         └──────────────────┘
========================================================================================
```

---

## Method

### 1. Skill Memory 4중 튜플 정형화

Recuris의 스킬 메모리는 진화 라운드 `k`에서 아래 4가지 요소의 결합으로 정의된다:

```
M_k = (E_k, W_k, ρ_k, C_k)
```

상세 스펙 정의 및 매니페스트 구조 → [스니펫: skillmemory_spec.md](../source/git/snippets/Recursive_Experiential-Working_Memory_Evolution_for_Long-Horizon_Agent_Harnesses_2026_arxiv__skillmemory_spec.md)

| 구성 요소 | 기호 | 정의 및 역할 | 저장 위치 및 포맷 |
|---|---|---|---|
| **Experiential Memory** | `E_k` | 과거 실패 과제에서 추출된 도구 사용 규칙, 오류 방지 지침, 절차적 행동 가이드 | `em/**/*.md` (마크다운 카드) |
| **Working-Memory Spec** | `W_k` | 작업 목표의 상태 스키마(`pending`, `done`, `blocked`), 상태 보드 렌더러, 목표 갱신 규칙 | `manifest.yaml` 내 `wm:` 블록 |
| **Invocation Policy** | `ρ_k` | 특정 실행 이벤트(`e_t`)와 작업 상태(`w_t`)를 감지하여 적절한 스킬 카드를 주입하는 딜리버러 | `manifest.yaml` 내 `delivery:` 블록 |
| **Checkers** | `C_k` | 상태 변경 도구 호출 작성 시 인자 유효성 및 완료 조건을 평가하여 위반 시 재작성(bounce)을 유도하는 검증기 | `manifest.yaml` 내 `checkers:` 및 `plugin.py` |

### 2. 과제 내부(Within-Task) 검증된 EM–WM 결합

- **구조화된 작업 상태 (Structured Working State)**:
  - 각 목표 항목 `g`는 내용(content), 상태(status: `pending`, `done`, `blocked`), 뒷받침 증거(supporting evidence), 선택적 차단 사유(blocker)를 명시적으로 추적한다.
  - 에이전트는 대화창 상단에 렌더링된 마크다운 상태 보드(Status Board)를 통해 미해결 목표(`pending`)를 항상 인지한다.
- **상태 기반 스킬 호출 (State-Grounded Skill Invocation)**:
  - 스킬은 전체 히스토리가 아닌 현재 상태 `w_t`와 실행 이벤트 `e_t`의 조건부 함수로 인출된다:
    ```
    E_t = ρ_k(w_t, e_t; E_k)
    ```
  - **Call-Time Invocation**: 모델이 상태 변경 도구 호출을 작성하는 순간(`PRE_WRITE`), 해당 도구의 이름과 연계된 스킬 카드를 즉시 컨텍스트에 주입하여 올바른 인자 규격을 준수하도록 강제한다.
  - **Boundary Invocation**: 턴 시작(`TURN_START`) 또는 의도 기록(`INTENT_RECORDED`) 시점에 미완료 목표에 부합하는 가이드를 주입한다.
- **증거 기반 상태 갱신 (Evidence-Grounded State Update)**:
  - 모델의 상태 갱신 제안 `w̃_{t+1}`은 결정론적 체커 `C_k`에 의해 평가되며 고정 커널 `K`를 통해서만 커밋된다:
    ```
    w̃_{t+1} = U_{W_k}(w_t, a_t, o_t)
    w_{t+1} = K(w_t, w̃_{t+1}, c_t)
    ```
  - **핵심 불변성(Invariant)**: 에이전트 모델 자신이 생성한 "완료했습니다"라는 텍스트 주장은 완료 증거로 인정되지 않으며, 오직 환경에서 반환된 실제 도구 실행 영수증(Tool Receipt)이 체커 조건 `C_{k,g} = 1`을 충족할 때만 `done`으로 변경된다.

상세 실행 런타임 루프 → [스니펫: runtime_loop.md](../source/git/snippets/Recursive_Experiential-Working_Memory_Evolution_for_Long-Horizon_Agent_Harnesses_2026_arxiv__runtime_loop.md)

### 3. 과제 간(Cross-Task) 제한적 스킬 메모리 진화

- **구조화된 실행 추적 (Structured Trace `Γ_k`)**:
  - 단순 텍스트 로그가 아닌 `Γ_k = {(w_t, E_t, a_t, o_t, w̃_{t+1}, c_t, w_{t+1})}_{t=1}^L`를 생성하여, 실패 발생 시점의 상태, 주입된 스킬, 실행 도구, 체커 판정을 일대일로 연결한다.
- **추적 기반 고장 국소화 (Trace-Based Failure Localization)**:
  - 고정된 Meta-Agent 진단기 `A_fixed`가 실패 궤적을 분석하여 결함이 발생한 구체적 컴포넌트 집합 `Z_k ⊆ {E, W, ρ, C}`를 식별한다.
    - `E` 귀인: 필요한 도메인 정책/스킬 카드 결여 또는 스킬 본문 오류.
    - `W` 귀인: 목표 항목 누락, 부적절한 필드 구조, 상태 전이 규칙 결함.
    - `ρ` 귀인: 스킬 호출 타이밍 지연, 부적절한 트리거 조건, 라우팅 누락.
    - `C` 귀인: 유효한 상태 전이의 거부 또는 무효한 상태 전이의 허용.
- **부품별 국소 패치 (Component-Specific Patching)**:
  - 식별된 컴포넌트 `Z_k`만을 수정하고 나머지 컴포넌트는 바이트 단위로 보존하는 국소 패치 연산 `⊕_{Z_k}`를 수행하여 무분별한 전면 수정을 원천 차단한다.
- **검증 게이트 승인 (Validation-Gated Patch Admission)**:
  - 제안된 후보 메모리 `M_k^+`는 고정된 코드 게이트 `G_fixed`의 검증을 거친다:
    1. 실패 원본 과제 `x_k`의 재실행 성공 확인.
    2. 보류 개발셋 `D_dev` 상에서 페어드 부트스트랩(3,000회 재샘플링) 유의성 검정 수행: 95% 신뢰구간 하한 `lo > 0`.
    3. 기존에 성공하던 앵커 과제의 성능 퇴보 개수 제한: `n_regressed ≤ reg_cap`.
    4. 테스트셋 정답 파라미터 누출 방지(`leakage_check`) 및 처방된 메커니즘의 실제 발화 여부 검증(`fingerprint_verify`).

상세 게이트 및 통계 프로토콜 구현체 → [스니펫: metaagent_gates.md](../source/git/snippets/Recursive_Experiential-Working_Memory_Evolution_for_Long-Horizon_Agent_Harnesses_2026_arxiv__metaagent_gates.md)

상세 수식 및 정형 정의 원문 → [발췌: source/paper/...md](../source/paper/Recursive_Experiential-Working_Memory_Evolution_for_Long-Horizon_Agent_Harnesses_2026_arxiv.md)

---

## Experiments & Results

### 1. Benchmark Datasets & Models

- **벤치마크 데이터셋**:
  1. **τ²-Retail (114 tasks)**: 이중 제어(dual-control) 대화 환경에서 복잡한 정책 제약(반품, 교환, 취소, 주소변경 등)과 데이터베이스 쓰기 도구를 처리하는 장기 고객 지원 벤치마크.
  2. **τ²-Airline (50 tasks)**: 항공편 예약 변경, 취소 수수료 계산, 바우처 적용 등 엄격한 정책 규칙 검증이 요구되는 대화형 도구 사용 벤치마크.
  3. **SkillFlow (166 tasks, 20 families)**: 20개 과제 계열별로 공통의 실행 흐름을 공유하며 지속적인 절차적 스킬 발견과 재사용을 평가하는 벤치마크.
  4. **Terminal-Bench 2.1 (46 tasks)**: 격리된 CLI 환경에서 복잡한 소프트웨어 빌드, 환경 설정, 시스템 엔지니어링 문제를 다루는 고난도 터미널 벤치마크.
- **평가 대상 언어 모델 (10 Models)**:
  - Deployment 모델: `doubao-seed-2-0-pro` (메모리 진화 루프가 실행된 기준 모델).
  - Open-weight 계열: `Qwen3.6-3B/8B/27B/35B`, `Granite-4.1-8B/20B`, `gpt-oss-20B/120B`.
  - Frontier 모델: `GPT-5.6 Sol`, `Claude Opus 5`, `Gemini 3.7 Flash`.

### 2. Overall Performance

모든 실험은 베이스 모델의 가중치를 완전히 고정한(frozen) 상태에서 수행되었다. 진화 루프는 `doubao-seed-2-0-pro` 상에서 단 1회 실행되어 스킬 메모리 패키지를 생성하였으며, 타 모델들은 이 패키지를 그대로 장착하여 추론에 활용하였다.

| 모델 | 벤치마크 | Base Agent (%) | Recuris (Evolved) (%) | Net Delta (pp) | 95% CI |
|---|---|---|---|---|---|
| **doubao-seed-2-0-pro** (Deploy) | τ²-Retail | 58.1 | **81.4** | **+23.3** | [+15.8, +30.8] |
| **doubao-seed-2-0-pro** (Deploy) | τ²-Airline | 52.0 | **66.0** | **+14.0** | [+1.0, +27.0] |
| **doubao-seed-2-0-pro** (Deploy) | SkillFlow | 34.6 | **51.4** | **+16.8** | [+9.3, +24.4] |
| **GPT-5.6 Sol** | τ²-Retail | 58.3 | **76.1** | **+17.8** | [+9.4, +26.1] |
| **Claude Opus 5** | τ²-Retail | 72.4 | **87.9** | **+15.6** | [+8.1, +23.0] |
| **Gemini 3.7 Flash** | τ²-Retail | 73.5 | **78.3** | **+4.8** | [-2.3, +11.8] |
| **Qwen3.6-27B** | SkillFlow | 42.2 | **58.7** | **+16.6** | [+8.4, +24.7] |
| **Qwen3.6-35B** | SkillFlow | 35.3 | **48.8** | **+13.5** | [+5.4, +21.7] |
| **gpt-oss-120B** | SkillFlow | 38.0 | **50.6** | **+12.7** | [+4.8, +20.5] |

- **주요 결과 요약**:
  - 총 37개 완료 모델-벤치마크 페어 중 35개에서 일관된 개선 확인.
  - 최상위 프론티어 모델인 Claude Opus 5를 τ²-Retail에서 72.4%에서 **87.9%**로 끌어올리며, 단일 모델 자체 능력으로 달성 불가능했던 SOTA 수준을 경신.

---

## In-Depth Analysis & Findings

### 1. 상호작용 지평선(Horizon) 길이에 따른 성능 분석

과제의 본질적 해결 턴 길이에 따라 4분위수(Q1: 단기 ~ Q4: 초장기)로 분류하여 성능 격차를 분석한 결과, 상호작용 길이가 길어질수록 Recuris의 우위가 단조 증가하였다.

```
Task Success Rate (%) across Interaction Horizon Quartiles (τ²-Retail)
100% ┌─────────────────────────────────────────────────────────────┐
     │                                                     83.3%   │  ● Recuris (Evolved)
 80% │   80.0%             82.8%             80.0%           ●     │
     │     ●                 ●                 ●                   │
 60% │                                                             │  ■ Base Agent
     │                                               51.1%         │
 40% │   63.0%             58.6%                             ■     │
     │     ■                 ■                                     │
 20% │                                       35.3%                 │
     │                                         ■                   │
  0% └─────────────────────────────────────────────────────────────┘
          Q1 (Short)        Q2 (Mid-Low)      Q3 (Mid-High)     Q4 (Longest)
          [Δ = +17.0]       [Δ = +24.2]       [Δ = +44.7]       [Δ = +32.2]
```

- **이해(Read) vs 실행(Write) 결함의 분리**:
  - 정보 조회를 위한 도구 호출 비율(Read-Action Recall)은 베이스 에이전트와 Recuris 모두 모든 분위수에서 **88.0% ~ 97.9%**로 매우 높게 유지됨. 즉, 장기 과제 실패는 지식을 몰라서가 아님.
  - 필수 데이터베이스 변경 호출 비율(Required-Write Recall)에서 베이스 에이전트는 42%의 에피소드에서 쓰기 도구를 단 한 번도 호출하지 않은 반면, Recuris는 쓰기 누락율을 16%로 대폭 축소하며 +26.7%p의 Write Recall 우위를 달성함.

### 2. EM–WM 결합 메커니즘 절제 연구 (Ablation)

동일한 10개 스킬 라이브러리를 보유한 상태에서 호출 통제 주체와 메모리 결합 방식을 비교하였다.

| 구성 (Configuration) | τ²-Retail 성공률 (%) | 필수 쓰기 실행률 (%) | 에피소드당 토큰 소비량 (Tokens / Success) |
|---|---|---|---|
| **Base Agent** (메모리 없음) | 57.5 | 55.7 | 125k |
| **EM Only** (상태 추적 없이 스킬만 주입) | 59.5 (+2.0) | 57.3 | 138k |
| **WM Only** (스킬 없이 상태 보드만 유지) | 81.4 (+23.9) | 79.8 | 104k |
| **Model-Controlled** (매 턴 10개 스킬 전체 프롬프트 주입) | 65.6 (+8.1) | 61.1 | 147k |
| **Recuris (EM + WM Verified Coupling)** | **83.6 (+26.1)** | **82.4** | **101k** |

- **핵심 발견**: 스킬 라이브러리 전체를 프롬프트에 상시 주입하고 모델이 알아서 쓰게 한 방식(Model-Controlled)은 스킬이 전혀 없는 WM-Only(81.4%)보다 오히려 낮은 점수(65.6%)를 기록하고 토큰 비용은 46% 폭증함. 스킬의 단순 가용성(Availability)이 중요한 것이 아니라, **"언제 스킬을 컨텍스트에 넣을 것인가"를 하네스가 작업 상태에 기반하여 통제(State-Grounded Invocation)하는 것이 결정적 메커니즘**임이 입증됨.

### 3. 도메인별 핵심 메커니즘의 이중 해리 (Double Dissociation)

작업 상태를 유지하는 하위 4대 메커니즘(Pre-Write Review, Status Board, Gate Termination, Truth Guard)을 하나씩 제거하며 평가한 결과, 도메인에 따라 결정적인 메커니즘이 완전히 상이함을 발견하였다.

| 제거된 컴포넌트 | τ²-Retail 성능 저하 (pp) | τ²-Airline 성능 저하 (pp) | 판정 및 메커니즘 특성 |
|---|---|---|---|
| **Pre-Write Review 제거** | -0.7 (영향 없음) | **-13.5 (치명적)** | 사전 검증: Airline의 복잡한 정책 제약 준수에 필수 |
| **Status Board 제거** | **-17.3 (치명적)** | +0.2 (영향 없음) | 가시적 상태: Retail의 다중 목표 누락 방지에 필수 |
| **Truth Guard 제거** | 0.0 (영향 없음) | 0.0 (영향 없음) | 사후 감사: 이미 실행된 도구 결과를 되돌릴 수 없어 무의미 |

- **시사점**: 단일한 고정 하네스 구조는 도메인에 따라 최적이 될 수 없다. 따라서 사전에 하네스 구조를 고정하지 않고, 실패 궤적 `Γ`로부터 결함 부품을 읽어내어 동적으로 진화시키는 `A_fixed`의 필요성이 실증됨.

### 4. 고장 국소화(Failure Localization) 정확도 비교

알려진 고의 결함(Injected Faults)을 스킬 메모리에 주입하고 결함 부품을 역추적하는 벤치마크 평가:

| 판정 입력 조건 | 전체 정확도 (Accuracy) | Macro Precision | 주요 실패 양상 |
|---|---|---|---|
| **최종 결과만 제공 (Outcome Only)** | 13.0% | 27.6% | 임의 추측 수준(33.3% 기준선 이하) |
| **원시 대화 로그 제공 (Raw Trajectory)** | 37.0% | 36.8% | 스킬 미호출(Invocation fault)을 전혀 감지 못함 (0%) |
| **구조화된 실행 추적 제공 (`Γ_k`)** | **64.8%** | **64.4%** | 상태 타임라인과 체커 결정을 통해 정확한 귀인 달성 |

---

## Comparison with Existing Paradigms

```
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                      AGENT SELF-IMPROVEMENT PARADIGM COMPARISON                       │
├──────────────────────┬────────────────────────┬───────────────────────────────────────┤
│ 패러다임             │ 대표 연구 / 방식       │ 주요 특징 및 장단점                   │
├──────────────────────┼────────────────────────┼───────────────────────────────────────┤
│ Monolithic History   │ Reflexion, Mem0,       │ - 전체 상호작용 또는 초기 프롬프트로   │
│ Memory Retrieval     │ Promptbreeder          │   스킬 검색.                          │
│                      │                        │ - 장기 턴에서 컨텍스트 오염 극심.     │
│                      │                        │ - 실패 시 전체 프롬프트 재작성 위험.  │
├──────────────────────┼────────────────────────┼───────────────────────────────────────┤
│ Unconstrained Self-  │ Voyager, Alita,        │ - 에이전트가 코드/스킬을 자유롭게 작성. │
│ Modifying Code Agent │ AlphaEvolve            │ - 런타임 회귀 및 Misevolution 취약.   │
│                      │                        │ - 검증 게이트 부재 시 기능 파괴 발생. │
├──────────────────────┼────────────────────────┼───────────────────────────────────────┤
│ Recuris (State-      │ Recuris (본 연구)      │ - WM과 EM을 결합한 상태 기반 인출.    │
│ Grounded Recursive   │                        │ - M=(E,W,ρ,C) 부품별 정밀 국소 패치.  │
│ Evolution)           │                        │ - 결정론적 코드 게이트로 회귀 차단.   │
│                      │                        │ - 모델 가중치 불변, 타 모델 무수정    │
│                      │                        │   전이성(Zero-shot Transfer) 보장.    │
└──────────────────────┴────────────────────────┴───────────────────────────────────────┘
```

---

## Analysis & Insights

### Strengths & Significance
1. **RSI의 안전하고 결정론적인 경계 설정**: 모델 가중치나 하네스 엔진 코드 전체를 임의 수정하게 두지 않고, `M = (E, W, ρ, C)`라는 명확한 4대 패치 공간과 통계적 부트스트랩 검증 게이트(`held_out_paired_gate`)를 구축하여 안전한 자기개선을 달성함.
2. **단일 모델 진화 메모리의 강력한 크로스 모델 전이성**: 고비용 프론티어 모델로 매번 진화 루프를 돌릴 필요 없이, 중간 규모 오픈소스 모델(`doubao-seed-2-0-pro`)에서 진화시킨 스킬 메모리가 GPT-5.6 Sol, Claude Opus 5 등 최상위 모델에서도 즉각적인 대폭 성능 향상(+15.6 ~ +17.8pp)을 유도함을 입증.
3. **상태 기반 인출의 토큰 및 연산 효율성**: 전체 스킬을 프롬프트에 쏟아붓는 방식 대비 필요한 순간에만 선택적으로 스킬을 주입함으로써 토큰 소모량을 대폭 절감하고 환각 및 상태 전이 혼선을 방지.

### Limitations
1. **공유 구조가 없는 고립 과제에서의 한계**: Terminal-Bench 2.1과 같이 과제 간 공유되는 정책이나 도구 규칙이 전무한 도메인에서는 크로스 태스크 스킬 진화가 수렴하지 않으며, 단일 과제 내 시험시점 적응(TTA) 모드로 제한됨.
2. **메모리 패키지의 단조 증가 성향**: 현재 게이트 정책 하에서 라운드가 진행됨에 따라 스킬 수가 누적 증가(51개 추가, 0개 삭제)하여 중복 카드가 잔존하는 경향이 있음 (향후 가지치기/통합 연산자 필요).

---

## References & Relative Links
- **원본 논문 및 코드**:
  - Paper: [arXiv:2608.24876](https://arxiv.org/abs/2608.24876)
  - GitHub: [Gen-Verse/Recuris](https://github.com/Gen-Verse/Recuris) / 로컬 서브모듈: `source/git/Recuris_Gen-Verse`
- **로컬 발췌 및 코드 스니펫**:
  - 논문 핵심 수식 및 결과 발췌: [source/paper/...md](../source/paper/Recursive_Experiential-Working_Memory_Evolution_for_Long-Horizon_Agent_Harnesses_2026_arxiv.md)
  - 런타임 루프 스니펫: [runtime_loop.md](../source/git/snippets/Recursive_Experiential-Working_Memory_Evolution_for_Long-Horizon_Agent_Harnesses_2026_arxiv__runtime_loop.md)
  - 메타에이전트 게이트 스니펫: [metaagent_gates.md](../source/git/snippets/Recursive_Experiential-Working_Memory_Evolution_for_Long-Horizon_Agent_Harnesses_2026_arxiv__metaagent_gates.md)
  - 스킬 메모리 규격 스니펫: [skillmemory_spec.md](../source/git/snippets/Recursive_Experiential-Working_Memory_Evolution_for_Long-Horizon_Agent_Harnesses_2026_arxiv__skillmemory_spec.md)
