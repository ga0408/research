# Does On-Policy Distillation Really Distill? From Noisy Teacher to Self-Improvement — 핵심 발췌

> 출처: [분석 문서](../../report/[paper][git]_Does_On-Policy_Distillation_Really_Distill_From_Noisy_Teacher_to_Self-Improvement_2026_Purdue.md) / 원본: [arXiv:2608.31046](https://arxiv.org/abs/2608.31046) / 코드: [GitHub](https://github.com/DripNowhy/On-Policy-Self-Adaptation)

---

## 1. Background & Preliminaries: On-Policy Distillation (OPD)

온폴리시 증류(On-Policy Distillation, OPD)는 학생 모델(`π_s`)이 생성한 궤적에 대해 학생과 교사 모델(`π_t`) 간의 역방향 쿨백-라이블러(Reverse Kullback–Leibler, KL) 발산을 최소화하며, K1 추정기(estimator)를 통해 다음과 같이 정식화된다:

```
KL(π_s ‖ π_t) = E_{y ~ π_s(·|x)} [ ∑_{i=1}^{|y|} (log π_s(y_i | x, y_{<i}) - log π_t(y_i | x, y_{<i})) ]
```

실제 강화학습(RL) 목적함수에서 OPD는 각 토큰 위치 `i`에 대한 토큰 단위 어드밴티지(advantage) `A_i`를 제공하는 형태로 표현된다:

```
L_OPD = - E [ (1 / |y|) ∑_{i=1}^{|y|} A_i · log π_s(y_i | x, y_{<i}) ]

where A_i = log(π_t(y_i | x, y_{<i}) / π_s(y_i | x, y_{<i})) = log π_t(y_i | x, y_{<i}) - log π_s(y_i | x, y_{<i})
```

---

## 2. Noisy Teacher Supervision in OPD

### 2.1 정의 및 노이즈 측정
교사는 학생 정책이 생성한 prefix(`y_{<i}`)를 조건으로 토큰을 평가해야 하므로, 본질적으로 교사 관점에서는 오프폴리시(off-policy) 상태에 놓이게 된다.
- **노이즈 정의**: 검증기(verifier)로 정답 여부를 판정할 수 있는 박스 정답(`\boxed{}`) 토큰에 대해, 정답 trajectory의 정답 토큰에 음수 어드밴티지(`A_i < 0`)를 부여하거나 오답 trajectory의 토큰에 양수 어드밴티지(`A_i > 0`)를 부여하는 경우를 노이즈로 정의.
- **교사 모델 크기에 따른 노이즈 비율 (Qwen3-1.7B 학생 기준)**:
  - `Qwen3-4B-Instruct` 교사: 정답 trajectory 노이즈 20.4%, 오답 trajectory 노이즈 40.8% → 전체 노이즈 비율 **30.6%**
  - `Qwen3-30B-A3B-Instruct` 교사: 정답 trajectory 노이즈 43.4%, 오답 trajectory 노이즈 26.0% → 전체 노이즈 비율 **34.7%**
  - `Qwen3-235B-A22B-Instruct` 교사: 정답 trajectory 노이즈 **97.8%** (정답 토큰의 97.8%에 음수 어드밴티지 부여), 오답 trajectory 노이즈 3.4% → 전체 노이즈 비율 **50.6%**
- **시사점**: 교사 모델의 파라미터 규모가 커질수록 학생과 교사 간의 분포 불일치(distributional mismatch)가 심화되어, 교사는 학생 궤적의 거의 모든 토큰에 무차별적으로 음수 어드밴티지를 부여함.

### 2.2 노이즈 둔감성 (Noise Insensitivity)
- 궤적을 노이즈 포함 여부로 분할하여 (1) Standard OPD (전체), (2) Clean only (노이즈 없는 궤적만), (3) Noisy only (노이즈 궤적만) 3가지로 학습 비교.
- **결과**: 세 가지 설정 모두 AIME24 Avg@4에서 유사한 수렴 속도와 유사한 최종 성능(~35%)에 도달.
- **연구 질문**: 교사 신호가 극도로 오염되어 있어도 학생이 동일하게 향상된다면, OPD의 성능 향상은 지식 전달(knowledge distillation)이 아닌 다른 원인에서 기인하는 것이 아닌가?

---

## 3. Disentangling the Source of Improvement

### 3.1 토큰 기여도 분석: 그래디언트 소실과 저확률 토큰
로짓 `z_t^v`에 대한 OPD 목적함수의 기울기는 다음과 같다:

```
- ∂L_OPD / ∂z_t^v ∝
    A_t · (1 - π_s(v | x, y_{<t}))    (if v = y_t, 샘플링된 토큰)
   -A_t · π_s(v | x, y_{<t})           (if v ≠ y_t, 샘플링되지 않은 토큰)
```

- **기울기 소실 조건**: (1) 어드밴티지 크기 `|A_t|`가 극히 작거나, (2) 샘플링된 토큰의 학생 확률 `π_s(y_t)`가 1에 근접할 때.
- **토큰 분포 관측**:
  - 전체 토큰의 29.2%는 정확히 `A = 0`, 51.7%는 `|A| ≤ 10^-4` 범위에 집중.
  - 학생이 높은 로그 확률(high-logp)을 부여한 상위 20% 토큰의 97.5%는 `|A| ≤ 10^-4`로 실질적 그래디언트가 0.
  - 상위 20% 또는 40%의 high-logp 토큰만으로 학습하면 원래 어드밴티지든 무작위 어드밴티지(`A ~ U[-1, 1]`)든 성능 향상이 전혀 발생하지 않음.
- **결론**: OPD 학습 신호의 대부분은 **학생 로그 확률이 낮은 하위 토큰(low-logp tokens)**에서 발생함.

### 3.2 학습 신호 분석: 단일 고정 음수 어드밴티지의 충분성
- 고정 어드밴티지 실험 (학생 로그 확률 하위 20% 토큰 대상):
  - `A = -0.5` 고정 음수 어드밴티지: AIME24 Avg@4가 꾸준히 향상되어 표준 교사 기반 OPD와 동일한 성능 곡선 및 동일한 응답 길이 확장(~12k 토큰) 달성.
  - `A = +0.2` 고정 양수 어드밴티지: 40스텝 만에 응답 길이가 0으로 붕괴하고 그래디언트 놈이 폭발(gradient explosion)하며 무작위 토큰 열을 생성하는 정책 붕괴 발생.
- **결론**: OPD의 실질적 이점은 교사의 지식 전달이 아니라, **학생 자신이 확률적 샘플링으로 뽑아낸 저확률 테일 토큰을 억제(suppression)**하는 데서 나옴. 이는 외부 교사 모델이 전혀 필요하지 않음을 시사함.

---

## 4. On-Policy Self-Adaptation (OPSA) Methodology

### 4.1 엔트로피 적응형 어드밴티지 (Entropy-Adaptive Advantage)
로그 확률이 낮은 토큰은 (1) 모델이 여러 후보 사이에서 불확실한 경우(high-entropy reasoning fork)와 (2) 확신하는 분포에서 우연히 샘플링된 비정상 테일(low-entropy tail)의 두 가지가 존재함. 엔트로피 `H_i`를 통해 신호 강도를 조절:

```
H_i = - ∑_{v ∈ V} π_θ(v | x, y_{<i}) · log π_θ(v | x, y_{<i})

r_i = 2 · (H_i - H_min) / (H_max - H_min) - 1 ∈ [-1, 1]

A_i^dyn = A_i^fix - (1/4) · δ · r_i
```

`A_i^fix = -3/4`, `δ = 1`로 설정하면:

```
A_i^dyn = -1/2 - (H_i - H_min) / (2 · (H_max - H_min)) ∈ [-1.0, -0.5]
```

- 높은 엔트로피 토큰(추론 갈림길): `A_i^dyn ≈ -1.0` (강한 학습 신호)
- 낮은 엔트로피 토큰: `A_i^dyn ≈ -0.5` (기본 억제 신호)

### 4.2 OPSA 최종 목적함수
학생 궤적 내에서 온폴리시 로그 확률이 가장 낮은 하위 20% 토큰 집합 `S_lowest20`에만 엔트로피 적응형 음수 어드밴티지를 적용:

```
L_OPSA = - E [ (1 / |S_lowest20|) ∑_{i ∈ S_lowest20} A_i^dyn · log π_θ(y_i | x, y_{<i}) ]
```

- 교사 순전파(forward pass) 불필요, 검증 보상(verifiable reward) 불필요, 정답 라벨 불필요, 힌트 프롬프트 불필요.

### 4.3 OPSA의 확률 재분배 4가지 작용 기제
1. **고엔트로피 위치의 테일 토큰 (High-Entropy Tail)**:
   - 샘플링된 저확률 토큰에 음수 어드밴티지 부여 → 해당 토큰 확률 억제 (`0.14 → 0.08`), 상위 헤드 토큰 집합 확률 증가 (`0.86 → 0.92`).
2. **고엔트로피 위치의 헤드 토큰 (High-Entropy Head)**:
   - 샘플링된 헤드 토큰에 음수 어드밴티지 부여 → 경쟁 관계에 있는 다른 유력 헤드 토큰들로 확률 질량을 균등하게 재분배 (예: `So`, `Wait`, `But` 사이의 확률 균등화), 탐색 다양성 보존.
3. **저엔트로피 위치의 테일 토큰 (Low-Entropy Tail)**:
   - 잘못 샘플링된 저확률 토큰 억제 (`0.09 → 0.03`) → 고신뢰도 정답 토큰으로 확률 집중 (`0.91 → 0.97`).
4. **저엔트로피 위치의 헤드 토큰 (Low-Entropy Head)**:
   - 상위 로그 확률 토큰은 하위 20%에 포함되지 않아 업데이트 대상에서 제외 → 모델의 기존 고신뢰도 지식 및 정밀도 완벽 보존.

---

## 5. Experimental Results

### 5.1 벤치마크 결과 비교 (Qwen3-1.7B, Non-Thinking Mode)
- 학습 데이터: DAPO-17k (문제 텍스트만 사용, 라벨/정답 미사용)

| Method | Superv. Type | AIME24 Avg@32 | AIME24 Pass@32 | AIME25 Avg@32 | AIME25 Pass@32 | HMMT25 Avg@32 | HMMT25 Pass@32 | Avg Avg@32 | Avg Pass@32 |
|---|---|---|---|---|---|---|---|---|---|
| Base Model | None | 13.44 | 40.00 | 9.69 | 30.00 | 5.73 | 23.33 | 9.62 | 31.11 |
| + GRPO | RLVR (Reward) | 33.96 | 70.00 | 25.31 | 50.00 | 15.10 | 43.33 | 24.79 | 54.44 |
| + TTRL | Self-Consist. | 19.90 | 30.00 | 9.79 | 30.00 | 5.73 | 23.33 | 11.81 | 27.78 |
| + OPD | 4B Teacher | 32.08 | 73.33 | 20.52 | 50.00 | 13.85 | 40.00 | 22.15 | 54.44 |
| + OPSD | Hint Teacher | 33.33 | 73.33 | 22.50 | 53.33 | 14.90 | 43.33 | 23.58 | 56.67 |
| **+ OPSA (Ours)** | **None (Zero)** | **48.85** | **80.00** | **35.31** | **66.67** | **23.33** | **50.00** | **35.83** | **65.56** |

- **상대 향상도 (Base 대비)**:
  - AIME24 Avg@32: `+263.5%` (+35.41 pt), Pass@32: `+100.0%` (+40.00 pt)
  - AIME25 Avg@32: `+264.4%` (+25.62 pt), Pass@32: `+122.2%` (+36.67 pt)
  - HMMT25 Avg@32: `+307.2%` (+17.60 pt), Pass@32: `+114.3%` (+26.67 pt)

### 5.2 모델 스케일 확장 결과
- **Qwen3-4B**:
  - AIME24 Avg@32: `23.33 → 62.08` (+38.75 pt, +166.1%), Pass@32: `56.67 → 83.33`
  - AIME25 Avg@32: `20.52 → 58.44` (+37.92 pt, +184.8%), Pass@32: `56.67 → 83.33`
- **Qwen3.5-9B**:
  - AIME24 Avg@32: `76.35 → 87.81` (+11.46 pt), Pass@32: `93.33 → 96.67`
  - HMMT25 Avg@32: `44.48 → 67.40` (+22.92 pt, +51.5%), Pass@32: `86.67 → 93.33`

### 5.3 OOD 일반화 및 효율성
- **MBPP+ (Code)**: Qwen3-1.7B `58.24 → 59.44`, Qwen3-4B `66.93 → 68.35`, Qwen3.5-9B `77.33 → 79.27`
- **GPQA-Diamond (Q&A)**: Qwen3-1.7B `27.92 → 32.40` (+16.0%), Qwen3-4B `38.46 → 41.29`, Qwen3.5-9B `70.53 → 73.70`
- **학습 스텝 시간(Step Time)**:
  - GRPO: 186.2초 (샘플 16개 롤아웃)
  - OPD: 61.2초 (교사 순전파 필요)
  - **OPSA: 22.8초** (교사 없음, 단일 롤아웃 기반 최적화로 약 8배 빠름)

---

## 6. Key Analysis & Ablations

### 6.1 성찰적 장문 추론의 자발적 발현 (Reflective Long-Form Reasoning)
- OPSA 학습 진행에 따라 응답 길이가 점진적으로 증가하며, 자가 점검/성찰 키워드(`wait`, `however`, `but`, `alternatively`, `hmm`, `perhaps`, `check`, `might`, `actually`) 빈도가 급증함.
- 응답 길이와 AIME 정확도 간 강한 양의 상관관계 (`r = 0.88` for 1.7B, `r = 0.82` for 4B).
- **갈림길 토큰 마스킹 실험 (Masking Fork Tokens)**: 상위 5개 후보에 성찰 단어가 포함된 갈림길 토큰을 업데이트에서 마스킹하면, 응답 길이 증가와 정확도 향상이 완전히 사라지고 300스텝 부근에서 붕괴됨.

### 6.2 생성 다양성 보존 (Jaccard Distance)
- 4-gram 단위 pairwise Jaccard Distance 분석 결과, 생성 길이가 길어질수록 베이스 모델과의 다양성 격차가 0으로 수렴. OPSA는 전체 엔트로피를 낮추면서도 핵심 갈림길에서 트리 형태(tree-style)의 다중 탐색 다양성을 완벽히 유지함.

### 6.3 RLVR 콜드 스타트 (Cold-Start for GRPO)
- OPSA로 사전 적응된 Qwen3-4B 체크포인트를 초기 모델로 삼아 DAPO-17k에서 GRPO를 추가 학습한 결과, 40스텝 만에 AIME24 Avg@4가 추가로 약 9 포인트 상승하며 매우 안정적으로 수렴함.
