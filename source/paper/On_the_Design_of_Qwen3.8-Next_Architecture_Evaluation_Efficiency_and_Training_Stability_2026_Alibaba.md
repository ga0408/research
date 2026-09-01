# On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability — 핵심 발췌

> 출처: [분석 문서](../../report/[paper]_On_the_Design_of_Qwen3.8-Next_Architecture_Evaluation_Efficiency_and_Training_Stability_2026_Alibaba.md) / 원본: [arXiv:2608.30320](https://arxiv.org/abs/2608.30320)

---

## 1. Abstract & Core Model Specifications

### Qwen3.8-Flash-Next 아키텍처 핵심 제원
- **모델 형태**: Sparse Mixture-of-Experts (MoE) Language Model
- **총 파라미터 (Total Parameters)**: 125B
- **토큰당 활성 파라미터 (Activated Parameters per Token)**: 6B
- **가속기 외 호스트 메모리 오프로드 파라미터 (Off-Accelerator Host Memory)**: 51B (N-gram Embedding Tables)
- **선행 연구 대비 효율**: 선행 397B-A17B 모델(Qwen3.7-Plus) 대비 14개 사전학습 벤치마크 중 8개에서 우위, 나머지에서 최대 2.6점 차이로 대등한 성능을 달성하면서, 활성 파라미터 1/3, 학습 토큰 수 1/3, 학습 FLOPs 약 1/9 수준으로 절감.
- **핵심 3대 설계 축**:
  1. **Token Mixing**: Gated DeltaNet(GDN) 3개 레이어와 Global Attention 1개 레이어의 레이어별 주기적 하이브리드(3:1 비율) 구성. 지속 사전학습(CPT) 및 서빙 시 Global Attention을 Qwen Sparse Attention(QSA, micro-block lightweight indexer 기반)으로 대체.
  2. **Gated Residual (GR)**: 단일 레지듀얼 스트림을 4개 브랜치(`nr = 4`)로 확장하고, RMSNorm 기반 채널별 게이트를 통합한 GatedNorm 방식의 원소별(elementwise) 읽기 게이트와 스칼라 쓰기 게이트를 결합하여 깊이에 따른 장거리/단거리 경로 동적 라우팅 및 고학습률 안정성 확보.
  3. **N-gram Embedding**: 가속기 HBM이 아닌 호스트 CPU 메모리에 51B 크기의 N-gram 임베딩 테이블을 분리 배치하고 비동기 프리페칭(Asynchronous Prefetching)을 통해 Layer 2에서 주입.
  4. **Muon Optimizer & Scaling Law**: 행렬 직교화 기반 Muon 옵티마이저를 2D 선형 변환에 적용하고, 융합 파라미터 분할, 배치 크기 웜업 제거, 상향된 최적 학습률 및 배치 크기 정렬.

---

## 2. Token Mixing: Gated DeltaNet (GDN) Hybrid & Qwen Sparse Attention (QSA)

### 2.1 Gated DeltaNet (GDN) 점화식 및 수식

```
[Gated Delta Recurrence (Head 단위)]

eS_{t-1} = α_t · S_{t-1}
e_t = v_t - eS_{t-1}^T · k_t
S_t = eS_{t-1} + β_t · k_t · e_t^T
y_t = S_t^T · q_t

등가 수식:
S_t = α_t · (I - β_t · k_t · k_t^T) · S_{t-1} + β_t · k_t · v_t^T
```

여기서:
- `S_t ∈ R^{dk × dv}`: 헤드별 고정 크기 순환 상태 행렬 (Recurrent State Matrix).
- `α_t ∈ (0, 1)`: 데이터 의존적 전역 감쇠 게이트 (Data-dependent Decay Gate).
- `β_t ∈ (0, 1)`: 델타 갱신 제어 쓰기 게이트 (Write Gate).
- `e_t = v_t - eS_{t-1}^T k_t`: 현재 키 `k_t`에 대해 기존 상태가 예측한 값과의 잔차 오차(Residual Error). 기존 연관성을 덮어쓰고(erase) 새로운 잔차만 기록.

#### GDN 파라미터화 및 게이팅
```
q_t = L2Norm(SiLU(ShortConv(W_q · x_t)))
k_t = L2Norm(SiLU(ShortConv(W_k · x_t)))
v_t = SiLU(ShortConv(W_v · x_t))

β_t = σ(W_β · x_t)
α_t = exp[-exp(A) · softplus(W_α · x_t + b_α)]

o_t = W_o · [σ(W_z · x_t) ⊙ RMSNorm(y_t)]
```
- Short Convolution: 순환 상태 압축 전 국소 귀납 편향(Local Inductive Bias) 제공.
- L2 Normalization: `q_t`, `k_t`의 크기를 제한하여 랭크-1 델타 상태 전이 안정화.
- Bounded Sigmoid Output Gate: SiLU 대비 바운디드 시그모이드 게이트 `σ(W_z · x_t)`를 적용하여 출력 크기 발산 억제.
- Zero-Centered RMSNorm: RMSNorm 가중치의 비정상적 비대화 방지.

---

### 2.2 Qwen Sparse Attention (QSA)

#### 압축 경량 인덱서 (Compressed Lightweight Indexer)
- 인덱서 구조: Multi-Query Attention (MQA, 쿼리 헤드 `H = 4`, 공유 키 헤드 1개).
- 키 시퀀스를 `r = 4` 크기의 비중첩 마이크로 블록으로 분할 후 평균 풀링:
```
bq_i^h = RMSNorm(W_Q^h · x_i),    k_i = W_K · x_i

bkb_b = RMSNorm(AvgPool(k_{p_b : p_b + r - 1})),    p_b = b · r

q_i^h = PRoPE(bq_i^h, i)
kb_b = PRoPE(bkb_b, p_b)
```
- **Partial RoPE 순서**: 키 압축(AvgPool)을 먼저 수행한 뒤 블록 시작 위치 `p_b`에 단일 RoPE를 적용하여 상이한 회전 위상이 평균화되는 왜곡을 원천 방지.

#### 블록 인과적 중요도 스코어링 (Block-Causal Scoring)
```
I_{i,b} =
  ∑_{h=1}^H ReLU(⟨ q_i^h, kb_b ⟩),    if p_b + r - 1 ≤ i
  -∞,                                    otherwise
```
- 토큰 예산 `K = 2048`, 블록 압축비 `r = 4`일 때 최대 `KB = K / r = 512`개 블록 선택.
- 최종 어텐션 대상 토큰 집합 `S_i`:
```
S_i = Expand(Top-KB(I_{i, :})) ∪ { r · ⌊(i + 1) / r⌋, ..., i }
```

#### 2단계 CPT 학습 절차
- **Stage 1: 인덱서 워밍업 (Indexer Warm-up)**: 백본을 동결하고 인덱서만 학습 (1,000 스텝, 256K 시퀀스 8개/스텝, 약 2B 토큰, LR = `1 × 10^-3`). Dense Attention의 어텐션 맵 `a_{i,j}`를 블록 합산한 `hat{a}_{i,b}`를 교사로 하여 KL 발산 손실 최소화:
```
L_KL = (1 / N) · ∑_i D_KL(hat{a}_{i, :} || Softmax(I_{i, :}))
```
- **Stage 2: 희소 공동 학습 (Sparse Training)**: 백본과 인덱서를 함께 8,000 스텝 학습 (256K 시퀀스 96개/스텝, 약 200B 토큰, LR = `2.5 × 10^-5`). 선택된 Top-KB 블록 내에서 재정규화된 교사 분포에 대해 KL 손실 적용.

---

## 3. Gated Residual (GR): 다중 브랜치 레지듀얼 및 GatedNorm

### 3.1 GR 수식 정식화
단일 레지듀얼 벡터를 `nr = 4`개 브랜치 `R = [R_1, ..., R_nr] ∈ R^{nr × d}`로 확장.

```
[독립 브랜치 RMSNorm]
hat{R}_i = RMSNorm(R_i; γ_i),    i = 1, ..., nr

[원소별 읽기 게이트 (Elementwise Read Gate)]
G = unvec(σ(W_u · SiLU((1 / nr) · W_d · vec(hat{R})))) ∈ R^{nr × d}

[블록 입력 합성]
x = (1 / nr) · ∑_{i=1}^{nr} G_i ⊙ hat{R}_i

[블록 연산 및 스칼라 쓰기 게이트 (Scalar Write Gate)]
y = F(x)
s = 2 · σ((1 / nr) · W_w · vec(hat{R})) ∈ R^{nr}
R_i' = R_i + s_i · y
```
- 저랭크 바틀넥 차원: `r_rank = d / 8`.
- `vec(hat{R}) ∈ R^{nr · d}`, `W_d ∈ R^{r_rank × nr · d}`, `W_u ∈ R^{nr · d × r_rank}`.
- `W_w ∈ R^{nr × nr · d}`.
- **Hres 제거**: 브랜치 간 선형 혼합 연산자 `H_res`를 완전 배제하여 블록당 레지듀얼 상태 전수 읽기 메모리 트래픽을 제거하고 추론 효율 및 수치 안정성 극대화.
- **FP8 저장 지원**: 게이트가 레지듀얼 크기를 좁은 범위로 유지하므로 레지듀얼 브랜치를 FP8로 저장하여 메모리 대역폭을 50% 절감.

### 3.2 경로 분해 및 분석 (Path Decomposition)
블록 `u`가 블록 `v`의 입력에 기여하는 정량적 크기:
```
a_{u→v} = (1 / nr) · ∑_{c=1}^{nr} G_c^{(v)} ⊙ γ_c ⊙ (s_c^{(u)} · y^{(u)} / rms(R_c^{(v)}))

π_{uv} = ||a_{u→v}|| / (∑_{u' < v} ||a_{u'→v}||)
```
- **발견된 구조**: 4개 브랜치 중 1개 브랜치(`b0`)는 Layer 0 GDN/MLP 출력을 집중 보존하여 깊은 레이어(Layer 10~19)의 Softmax Attention 레이어로 전달하는 장거리 전용 고속도로(Long-range Highway, 평균 스킵 10.9 레이어)로 작동.
- 나머지 3개 브랜치는 국소 단거리 연결(Local Short-range Connections, 평균 스킵 1.2~3.5 레이어)을 전담.

---

## 4. N-gram Embedding Layer & Scaling Dynamics

### 4.1 N-gram 임베딩 구조 및 배치
- 멀티헤드 해싱(Multi-head Hashing)을 이용해 로컬 토큰 n-gram 키로 호스트 메모리 테이블을 조회.
- **레이어 배치**: Layer 2에 단일 N-gram 임베딩 레이어를 배치. 첫 번째 레이어 연산 시간 동안 호스트 메모리로부터 비동기 프리페칭을 완료하여 연산-통신 중첩 실현.

### 4.2 N-gram 어휘 확장과 메트릭 괴리 (Loss vs Benchmark Disentanglement)
- **고정 파라미터 예산 내 MoE 전문가와의 배분 (Fixed Budget)**: N-gram 슬롯을 늘리고 MoE 전문가를 줄이면 10× (25% 파라미터)에서 Loss가 최저를 기록하지만, OOD Uncheatable PPL 및 다운스트림 벤치마크는 개선되지 않음.
- **호스트 메모리 추가 파라미터 확장 (20V ~ 200V)**:
  - N-gram 어휘를 20V에서 200V로 확장 시 Pre-training Loss는 1.553 → 1.526으로 단조 감소.
  - 반면 수학/코딩/추론 벤치마크(MATH, GSM8K, BBH)는 50V~100V 구간에서 포화(saturation) 및 미세 진동.
  - 중국어 벤치마크(C-Eval 71.75→74.94, CMMLU 72.29→73.24)는 지속적으로 향상.

---

## 5. Optimization & Training Stability

### 5.1 Muon 옵티마이저 설계 디테일
- **적용 대상**: 2D 선형 변환 행렬 가중치 (Attention q/k/v/o, GDN q/k/v/o, MoE Routed/Shared Experts fc1/fc2, N-gram K/V 프로젝션).
- **AdamW 유지 대상**: Token Embeddings, LM Output Head, MoE Router, GR Low-rank Projections (`W_d, W_u, W_w`), N-gram Embedding Tables (Adam, no weight decay).
- **Polar Express Newton-Schulz 반복**: 8회 반복으로 극대화된 직교화 정확도 및 그래디언트 스파이크 억제. 스케일링 팩터 `γ(A, B) = 0.2 · √(max(A, B))`.
- **융합 파라미터 분할 (Splitting Fused Parameters)**: Megatron의 융합된 QKV, GDN 입력 프로젝션, SwiGLU fc1(Gate+Up)을 직교화 전 헤드/연산자 단위로 분할하여 독립 Newton-Schulz 수행.
- **Canzona 분산 시스템**: DP 랭크 간 NS FLOPs를 균등 분할하는 `α`-balanced static partitioner 및 TP 랭크 간 비동기 All-to-All 통신을 통한 행렬 재구성 파이프라인. CUDA Graph로 100+개 서브 매트릭스 커널 런치 오버헤드 제거.

### 5.2 하이퍼파라미터 스케일링 법칙 및 배치 웜업 제거
- **배치 크기 스케일링**: 4T 토큰 예산에서 최적 배치 크기가 12.6M → 25.2M으로 상향.
- **배치 크기 웜업 (Batch-Size Warmup) 불필요**: 초기 배치 크기를 작게 시작하여 웜업하는 방식은 상수 배치 크기 대비 최종 손실이 열세(`2.5 × 10^-4` 패널티)하며 18.8% 더 많은 옵티마이저 스텝과 학습 시간을 소모. Muon 환경에서는 처음부터 목표 배치 크기로 학습하는 것이 우수.
- **학습률 스케일링**: 48레이어 156B-A7B MoE에서 최적 학습률이 `1.76 × 10^-3`으로 상향되며, `1 / √2`에서 `√2` 범위의 평탄한 최적 영역(flat bowl) 형성.

### 5.3 스트레스 테스트 및 GatedNorm 안정화 메커니즘
- **4× 최적 학습률 스트레스 테스트**:
  - AdamW + Qwen3.5 구조: 10k 스텝당 183회 손실 스파이크 발생, 19,932 스텝 중 213회 클리핑 임계치(0.5) 초과.
  - Muon + Qwen3.8-Flash-Next (GR + GatedNorm): 손실 스파이크 0회, 클리핑 임계치 초과 0회.
- **GatedNorm 메커니즘**:
  - 미게이팅 베이스라인은 고학습률에서 스케일 조정을 위해 활성화 이상치(Activation Outliers)를 비정상적으로 키우며 붕괴.
  - GatedNorm의 곱셈 게이트(Multiplicative Gate)가 직접적이고 유연한 재스케일링을 제공하여 활성화 크기를 엄격히 통제하고 스파이크를 방지.
  - 프로덕션 학습 전체에서 qk-clip이나 SwiGLU-clip 등 인위적 클리핑 없이 완전 무결한 학습 궤적 달성.
