# Recursive Experiential–Working Memory Evolution for Long-Horizon Agent Harnesses — 핵심 발췌

> 출처: [분석 문서](../../report/[paper][git]_Recursive_Experiential-Working_Memory_Evolution_for_Long-Horizon_Agent_Harnesses_2026_arxiv.md) / 원본: [arXiv:2608.24876](https://arxiv.org/abs/2608.24876) · [GitHub: Gen-Verse/Recuris](https://github.com/Gen-Verse/Recuris)

---

## 1. Problem Setup & Skill Memory Formulation

Frozen LLM `π_θ`와 도구 집합 `T`로 구성된 에이전트 하네스(agent harness)에서 장기 과제(long-horizon task) `x`를 해결할 때:

```
a_t ~ π_θ(· | h_t, w_t, E_t)
o_t = Env(a_t)
```

- `h_t`: 대화/상호작용 히스토리
- `w_t`: 스텝 `t`에서의 작업 상태 (working state)
- `E_t`: invocation policy에 의해 선택된 경험 스킬 집합 (experiential skills)
- `a_t`: 사용자 대면 메시지 또는 도구 호출 (tool call)
- `o_t`: 환경 관측값 또는 도구 실행 결과
- `τ = {(a_t, o_t)}_{t=1}^L`: 로우 궤적 (raw trajectory)
- `y ∈ {0, 1}`: 과제 최종 성공/실패 바이너리 레이블

진화 라운드 `k`에서 Recuris의 Skill Memory `M_k`는 4개 튜플로 정의된다:

```
M_k = (E_k, W_k, ρ_k, C_k)
```

1. `E_k` (Experiential Memory): 과거 실패 경험으로부터 추출된 재사용 가능한 에이전트 스킬 카드 집합
2. `W_k` (Working-Memory Specification): 과제 인스턴스별 작업 상태 `w_t`를 유지하기 위한 스키마 및 상태 갱신 제안 규칙
3. `ρ_k` (Invocation Policy): 특정 실행 이벤트(`e_t`) 발생 시 `E_k`로부터 어떤 스킬 카드를 컨텍스트에 전달할지 결정하는 호출 정책
4. `C_k` (Checker Set): 환경 관측 `o_t`가 제안된 상태 전이를 실제로 뒷받침하는지 검증하는 결정론적 완료 술어(completion predicate) 집합

---

## 2. Within-Task Verified EM–WM Coupling

과제 내부에서 구조화된 실행 추적(Structured Execution Trace) `Γ_k`를 기록:

```
Γ_k = {(w_t, E_t, a_t, o_t, w̃_{t+1}, c_t, w_{t+1})}_{t=1}^L
```

- `w̃_{t+1}`: 제안된 차기 상태 (`U_{W_k}(w_t, a_t, o_t)`)
- `c_t`: 체커 검증 결정 집합
- `w_{t+1}`: 고정 커널 `K`에 의해 커밋된 최종 상태

### State-Grounded Skill Invocation

```
E_t = ρ_k(w_t, e_t; E_k)
```

- `e_t`: 실행 이벤트 (예: 상태 변경 도구 호출 작성 시점 `PRE_WRITE`, 턴 경계 `TURN_START` 등)
- Call-time Invocation: 상태 변경 도구 호출 작성 시 해당 도구 이름에 매칭되는 스킬만 선별 주입
- Boundary Invocation: 턴 경계 시점에서 현재 미해결 목표(`pending`)에 매칭되는 스킬 주입

### Evidence-Grounded State Update

```
w̃_{t+1} = U_{W_k}(w_t, a_t, o_t)
w_{t+1} = K(w_t, w̃_{t+1}, c_t)
```

- 목표 `g`의 상태는 체커 `C_{k,g}(w_t, w̃_{t+1}, a_t, o_t) = 1`을 만족할 때만 `pending`에서 `done`으로 전이된다.
- 모델의 자체 성공 주장이나 도구 시도 자체는 완료 증거로 인정되지 않으며, 실제 도구 영수증(receipt) 또는 환경 결과값만 증거로 인정된다.

---

## 3. Bounded Cross-Task Skill Memory Evolution

과제 간(Cross-Task) 자기개선 루프는 고정된 Meta-Agent와 결정론적 코드 게이트를 통해 이루어진다:

```
Failure Diagnosis:
Z_k = {z_j}_{j=1}^{J_k},  where (f_j, z_j) = A_fixed(x_k, Γ_k, y_k)

Component Patching:
M_k^+ = M_k ⊕_{Z_k} P_fixed(Z_k, Γ_k)

Gated Admission:
M_{k+1} = G_fixed(M_k^+, M_k; x_k, D_dev) ? M_k^+ : M_k
```

- `Z_k ⊆ {E, W, ρ, C}`: 고정 진단 절차 `A_fixed`에 의해 결함이 특정된 컴포넌트 서브셋
- `⊕_{Z_k}`: 진단된 컴포넌트만 수정하고 나머지 컴포넌트는 바이트 단위로 보존하는 국소 패치 연산자
- `G_fixed`: 실패 원본 과제 `x_k` 해결 여부 및 보류 개발셋 `D_dev` 상의 페어드 부트스트랩 유의성(CI > 0) + 회귀 상한(`n_dn ≤ reg_cap`)을 만족할 때만 승인

---

## 4. Key Experimental Results

1. **4개 벤치마크 10개 모델 성능 (Overall)**:
   - 총 37개 완료 모델-벤치마크 페어 중 35개에서 성능 향상 달성.
   - τ²-Retail: GPT-5.6 Sol 58.3 → 76.1 (+17.8), Claude Opus 5 72.4 → 87.9 (+15.5), Deployment 모델(doubao-seed-2-0-pro) 58.1 → 81.4 (+23.3).
   - SkillFlow: Qwen3.6-27B 42.2 → 58.7 (+16.6), Qwen3.6-35B 35.3 → 48.8 (+13.5), Deployment 모델 34.6 → 51.4 (+16.8).
2. **장기 상호작용 지평선(Horizon)에 따른 이점**:
   - 상호작용 길이가 길어질수록 베이스라인과의 격차가 확대되어 최장 과제 4분위수에서 최대 +32.2p ~ +44.7p 격차 기록.
   - Read-action recall은 모든 모델/구간에서 88~98%로 차이가 없으나, Required-write recall에서 베이스라인 대비 +26.7p 압도. (이해 부족이 아닌 실행/완료의 문제 규명).
3. **고장 국소화(Failure Localization) 정확도**:
   - 최종 결과(outcome)만 볼 때: 13.0%
   - 로우 궤적(raw trajectory) 볼 때: 37.0%
   - 구조화된 실행 추적 `Γ` 활용 시: 64.8% (Macro Precision 64.4%)
4. **Meta-Agent 구현체 독립성 (Convergence)**:
   - Claude Code 기반 Meta-Agent vs DeepSeek Harness 기반 Meta-Agent: 86개 홀드아웃 과제에서 각각 +11.92p vs +10.47p/+9.30p로 수렴 (paired difference p=0.72).
