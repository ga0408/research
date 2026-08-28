# CaSKG: Counterfactual-Causal Skill Graphs for Scalable Agent Skill Retrieval — 핵심 발췌

> 출처: [분석 문서](../../report/[paper][git]_CaSKG_Counterfactual-Causal_Skill_Graphs_for_Scalable_Agent_Skill_Retrieval_2026_arxiv.md) / 원본: [arXiv:2608.25500](https://arxiv.org/abs/2608.25500) · [GitHub Code](https://github.com/ZhiyuanLi218/Caskg)

---

## 1. 논문 개요 및 초록 (Abstract & Problem Definition)

### 1.1 초록 (Abstract)
> "재사용 가능한 스킬 라이브러리(Skill Library)는 대규모 언어 모델(LLM) 에이전트가 다양한 태스크 전반에서 절차적 지식(Procedural Knowledge)을 재사용할 수 있도록 지원하지만, 동시에 메모리 접근을 까다로운 검색 문제로 전환시킨다.
> 전체 라이브러리를 프롬프트에 주입(Full-library prompting)하면 검색 커버리지는 유지되나 극심한 컨텍스트 비용과 주의 분산이 발생하고, 벡터 검색(Vector retrieval)은 간결한 이웃 스킬을 반환하지만 스킬들을 상호 독립적인 텍스트 단위로 취급하여 절차적 의존성을 놓친다. 그래프 기반 검색(Graph-based retrieval)은 워크플로 맥락을 복원할 수 있으나 관련성을 전파하는 엣지(Edge)들이 신뢰할 수 있을 때에만 유효하다.
>
> 본 논문에서는 검색 이전에 절차적 관계를 인과적으로 보정하는 반사실적-인과 스킬 그래프 프레임워크인 **CaSKG (Counterfactual-Causal Skill Graphs)**를 제안한다. CaSKG는 먼저 의미적(Semantic), 어휘적(Lexical), 입출력 인터페이스(Input/Output), 구조적(Structural) 증거와 복구 증거 및 선택적 LLM 판정관을 활용하여 고재현율(High-recall)의 방향성 후보 그래프를 구축한다. 이어서 소스 스킬의 제거(Removal), 대체(Substitution), 순서 역전(Reordering)을 수행하는 방향 조건부 텍스트 반사실적 프로브(Direction-conditioned textual counterfactual probes)를 적용하고, 베이지안 평활화(Bayesian smoothing)로 증거를 통합한 후 상태 필터링된 가중 그래프를 발행한다.
>
> CaSKG 그래프는 오프라인에서 구축되며 다운스트림 에이전트 정책이나 환경 인터페이스의 변경 없이 즉시 적용된다. ALFWorld ID-140 및 ScienceWorld U211의 6개 LLM 백본 평가에서 CaSKG는 12개 모델-벤치마크 조합 전체에서 최고 성능을 달성하였다. 기존 SOTA인 Graph-of-Skills (GoS) 대비 6개 모델 매크로 평균 ScienceWorld 점수를 72.62점에서 80.50점으로, ALFWorld 성공률을 80.01%에서 86.79%로 향상시킴과 동시에 두 벤치마크 모두에서 평균 환경 상호작용 스텝 수를 일관되게 감소시켰다."

---

## 2. 후보 스킬 그래프 유도 (Candidate Skill Graph Induction)

### 2.1 다중 소스 이종 증거 및 초기 결합 점수 (Eq. 1)

스킬 라이브러리 `S = {s_1, ..., s_n}`에 대해 방향성 후보 관계 집합 `C ⊆ S × S`를 생성한다. 각 후보 엣지 `(s_i, s_j) ∈ C`는 `s_i → s_j`의 가설적 지원 관계를 나타낸다.

활성 신호 집합 `A_ij`와 각 신호의 가중치 `λ_k ≥ 0`, 정규화된 신호값 `φ_k(i, j) ∈ [0, 1]`, 구조적 신호값 `φ_struct(i, j)`, 구조적 임계값 `τ_str` 및 보존 계수 `η_str`에 대해 초기 연관 점수 `A_ij`는 다음과 같이 정의된다:

```
A~_ij = clip[0, 1]( ( ∑_{k ∈ A_ij} λ_k · φ_k(i, j) ) / ( ∑_{k ∈ A_ij} λ_k ) )

A_ij = max( A~_ij, η_str · φ_struct(i, j) )   [if φ_struct(i, j) > τ_str]
     = A~_ij                                  [otherwise]
```

- `A_ij ∈ [0, 1]`는 초기 후보 가중 그래프 `(C, A)`의 엣지 가중치로 기능한다.
- 후보 집합 `C`에서 계산 예산에 따라 검증 프론티어 `F ⊆ C`를 선별하여 반사실적 검증으로 전달하며, 나머지 `C \ F`는 즉시 기각되지 않고 미검증 상태로 유지된다.

---

## 3. 방향 조건부 텍스트 반사실적 프로브 (Counterfactual Probing)

### 3.1 3대 반사실적 프로브 정의 (Eq. 2)

선별된 후보 엣지 `(s_i, s_j) ∈ F`에 대해 LLM을 통해 3가지 상호보완적 텍스트 반사실적 변형 `P_m(s_i, s_j)`을 가하고 지지 점수 `e^(m)_ij ∈ [0, 1]`를 측정한다:

1. **제거 프로브 (Removal Probe) — 필요성 (Necessity) 측정**:
   ```
   P_rem(s_i, s_j) = (∅, s_j)
   ```
   소스 스킬 `s_i`를 사용 불가능하게 만들었을 때 타깃 스킬 `s_j`의 실행이 저해되는지(Impairment) 검증.

2. **대체 프로브 (Substitution Probe) — 소스 특이성 (Specificity) 측정**:
   ```
   P_sub(s_i, s_j) = (s~_i, s_j)
   ```
   소스 스킬 `s_i`를 어휘·의미적 중복이 적은 임의의 대체 스킬 `s~_i`로 치환했을 때 워크플로가 붕괴하거나 성능이 저하되는지(Degradation) 검증.

3. **순서 역전 프로브 (Reordering Probe) — 방향성 및 순서 의존성 (Directionality) 측정**:
   ```
   P_ord(s_i, s_j) = (s_j, s_i)
   ```
   스킬의 실행 순서를 반대로 뒤집었을 때(`s_j` 실행 후 `s_i`) 워크플로의 논리적·물리적 일관성이 상실되는지(Loss of Coherence) 검증.

각 프로브의 출력 점수 `e^(m)_ij ∈ [0, 1]` (`m ∈ {rem, sub, ord}`)는 점수가 높을수록 해당 변형에 의해 가설적 의존성이 크게 훼손됨을 뜻하므로, `s_i → s_j` 방향성 의존성에 대한 긍정적 지지 증거로 일관되게 정렬된다.

---

## 4. 베이지안 엣지 캘리브레이션 및 상태 게이팅 발행 (Bayesian Calibration & Publication)

### 4.1 베이지안 증거 누적기 (Beta Accumulator) (Eq. 3, 4, 5)

증거의 극성 `z^(m)_ij` 및 증거 질량 `δ^(m)_ij` (최소 증거 바닥값 `ε_e > 0`):

```
z^(m)_ij = I[ e^(m)_ij > 0.5 ]

δ^(m)_ij = max( 2 · |e^(m)_ij - 0.5|, ε_e )
```

사전 분포 `Beta(1, 1)`에 기반한 사후 증거 파라미터 `α_ij, β_ij` 및 평활화된 관계 신뢰도 점수 `c_ij`:

```
α_ij = 1 + ∑_{m ∈ {rem, sub, ord}} z^(m)_ij · δ^(m)_ij

β_ij = 1 + ∑_{m ∈ {rem, sub, ord}} (1 - z^(m)_ij) · δ^(m)_ij

c_ij = α_ij / (α_ij + β_ij)
```

### 4.2 4단계 상태 게이팅 및 최종 발행 가중치 (Eq. 6, 7, 8)

대칭적 확인 임계값 `τ_c ∈ (0.5, 1)`에 따른 엣지 평가 상태 `σ_ij`:

```
σ_ij = confirmed    if (s_i, s_j) ∈ F ∧ c_ij > τ_c
     = rejected     if (s_i, s_j) ∈ F ∧ c_ij < 1 - τ_c
     = uncertain    if (s_i, s_j) ∈ F ∧ (1 - τ_c ≤ c_ij ≤ τ_c)
     = unvalidated  if (s_i, s_j) ∈ C \ F
```

발행 감쇄 계수 `ρ_ij`와 최종 엣지 가중치 `w^(pub)_ij`:

```
b_ij = max( A_ij, c^_ij, ε_w )

ρ_ij = 1         (if σ_ij = confirmed)
     = ρ_unc     (if σ_ij = uncertain, 0 < ρ_unc < 1)
     = ρ_scaf    (if σ_ij = unvalidated ∧ (s_i, s_j) ∈ E_scaf, 0 < ρ_scaf < ρ_unc)
     = 0         (otherwise; rejected or unselected unvalidated)

w^(pub)_ij = clip[ε_w, 1]( ρ_ij · b_ij )   [if ρ_ij > 0]
           = 0                            [if ρ_ij = 0]
```

최종 발행 그래프: `G_pub = (S, E_pub, W, Σ)`, 여기서 `E_pub = { (s_i, s_j) ∈ C : w^(pub)_ij > 0 }`.

---

## 5. 태스크 조건부 스킬 검색 (Task-Conditioned Skill Retrieval)

### 5.1 개인화된 페이지랭크 확산 (Personalized PageRank) (Eq. 9)

태스크 쿼리 `q`로부터 유도된 역순위 가중 시드 분포 `π_q`, `G_pub`의 행 정규화 전이 행렬 `T`, 재시작 계수 `γ ∈ (0, 1)`:

```
p^(t+1) = γ · π_q + (1 - γ) · T^T · p^(t)
```

수렴 후의 한계 분포 `p`에 따라 상위 스킬들을 순위화하여 다운스트림 LLM 에이전트의 컨텍스트로 반환한다.

---

## 6. 핵심 실험 결과 (Table 1, Table 2, Table 3)

### Table 1: Skill1000 벤치마크 종합 결과 (ALFWorld ID-140 & ScienceWorld U211)

| 백본 모델 | 방법론 | ALFWorld ID-140 R(%) ↑ | ALFWorld Steps ↓ | ScienceWorld U211 R ↑ | ScienceWorld Steps ↓ |
|---|---|---|---|---|---|
| **MiniMax-M2.7** | Vanilla Skills | 42.90% | 22.54 | 45.90 | 21.73 |
| | Vector Skills | 45.70% | 22.84 | 43.21 | 21.45 |
| | GoS (Graph-of-Skills) | 63.60% | 19.69 | 55.85 | 18.91 |
| | **CaSKG (Ours)** | **73.57%** | **18.44** | **68.33** | **17.45** |
| **GLM-5.2** | Vanilla Skills | 95.00% | 11.05 | 75.50 | 17.03 |
| | Vector Skills | 96.43% | 10.12 | 77.07 | 16.65 |
| | GoS (Graph-of-Skills) | 95.71% | 9.91 | 80.33 | 15.75 |
| | **CaSKG (Ours)** | **97.86%** | **9.69** | **85.11** | **14.52** |
| **Kimi-K2.6** | Vanilla Skills | 77.90% | 16.07 | 72.23 | 18.91 |
| | Vector Skills | 90.00% | 13.49 | 72.58 | 17.55 |
| | GoS (Graph-of-Skills) | 93.60% | 13.08 | 76.82 | 16.15 |
| | **CaSKG (Ours)** | **95.00%** | **12.34** | **83.88** | **15.43** |
| **Qwen3.5-397B-A17B** | Vanilla Skills | 79.30% | 15.60 | 63.72 | 18.34 |
| | Vector Skills | 78.60% | 15.49 | 62.60 | 18.51 |
| | GoS (Graph-of-Skills) | 88.60% | 14.15 | 63.18 | 17.08 |
| | **CaSKG (Ours)** | **92.14%** | **11.60** | **74.97** | **15.56** |
| **DeepSeek-V4-Flash** | Vanilla Skills | 72.86% | 16.91 | 64.84 | 18.49 |
| | Vector Skills | 78.57% | 16.89 | 68.65 | 18.39 |
| | GoS (Graph-of-Skills) | 77.86% | 17.09 | 73.45 | 16.20 |
| | **CaSKG (Ours)** | **86.43%** | **14.41** | **83.40** | **15.61** |
| **GPT-5.6-Luna** | Vanilla Skills | 72.86% | 17.74 | 84.09 | 14.99 |
| | Vector Skills | 55.00% | 22.06 | 84.09 | 14.40 |
| | GoS (Graph-of-Skills) | 60.71% | 21.86 | 86.08 | 14.22 |
| | **CaSKG (Ours)** | **75.71%** | **17.79** | **87.33** | **13.18** |
| **6개 모델 매크로 평균** | GoS | 80.01% | 15.96 | 72.62 | 16.39 |
| | **CaSKG** | **86.79% (+6.78%p)** | **14.05 (-1.91)** | **80.50 (+7.88점)** | **15.29 (-1.10)** |

### Table 2: 스킬 라이브러리 규모별 민감도 (ALFWorld ID-140)

| 백본 모델 | 스킬 수 | CaSKG R(%) | GoS R(%) | ΔR (%p) | CaSKG Steps | GoS Steps |
|---|---|---|---|---|---|---|
| **MiniMax-M2.7** | 200 | 57.14% | 50.00% | +7.14 | 20.73 | 22.21 |
| | 500 | 67.86% | 45.00% | +22.86 | 19.95 | 23.07 |
| | 1,000 | 73.57% | 63.60% | +9.97 | 18.44 | 19.69 |
| | 2,000 | 70.00% | 54.29% | +15.71 | 18.71 | 21.32 |
| **Qwen3.5-397B-A17B** | 200 | 85.00% | 76.43% | +8.57 | 14.50 | 16.25 |
| | 500 | 94.29% | 72.86% | +21.43 | 12.48 | 16.94 |
| | 1,000 | 92.14% | 88.60% | +3.54 | 11.60 | 14.15 |
| | 2,000 | 91.43% | 77.86% | +13.57 | 12.31 | 16.47 |

### Table 3: 컴포넌트 소거 실험 (MiniMax-M2.7, Skill1000, ALFWorld ID-140)

| 변형 모델 (Variant) | 후보 엣지 \|C\| | 검증 엣지 \|F\| | 발행 엣지 \|E_pub\| | 성공률 R(%) ↑ | 평균 스텝 Steps ↓ |
|---|---|---|---|---|---|
| **Full CaSKG** | 9,937 | 500 | 3,292 | **73.57%** | **18.44** |
| **Semantic-only 후보 유도** | 3,982 | 500 | 2,698 | 67.14% | 19.21 |
| **w/o LLM Judge 후보 정제** | 9,753 | 500 | 3,188 | 71.43% | 18.79 |
| **Publish all candidates (미필터링)** | 9,937 | 0 | 9,937 | 71.43% | 18.74 |
