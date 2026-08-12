> [paper] https://arxiv.org/abs/2608.07408

# Addressable Memory for Video World Models

## Summary & Outline

본 논문은 대화형 비디오 월드 모델(Interactive Video World Models)의 장기 비주얼 지속성(Visual Persistence) 문제를 해결하기 위해, 학습 없이(training-free) KV 캐시 메모리의 주소 지정 가능성(Addressability)과 정보 보존성(Informativeness)을 동시에 보장하는 **WorldTrace** 프레임워크를 제안한다.

### Outline
- **Problem & Motivation**: 생성 길이가 학습 컨텍스트 한계를 초과할 때 RoPE 오프셋이 분포 외(OOD) 영역으로 이탈하여 메모리 인출이 실패하는 Addressability 병목과, 텐서 평균화 시 RoPE 위상 상쇄(Phase Cancellation)로 정보가 손실되는 비주얼 캐시 압축의 한계 분석.
- **Contributions**: Slot Indexing 기반의 가상 위치 할당 (Def. 1), 정규 공간(canonical space) 내 키 평균화 기반의 **WorldTrace-Field** (Def. 2), 랜드마크 기반 고정 키 추적의 **WorldTrace-Landmark**, 그리고 닫힌 루프(closed-loop) 장기 탐색 평가 벤치마크인 **LoopBench** 제안.
- **Method**: 메모리를 최근 윈도우($N_r$)와 요약 캐시($N_s$)로 분할하고, Slot Rank 기반의 고정 가상 위치에 Canonical Unrotated Keys를 매핑하는 수학적 아키텍처.
- **Experiments**: Matrix-Game-2 (1.3B) 및 LingBot-World (14B) 롤아웃 환경에서 비주얼 일관성(+15.5%) 및 재방문 에피소드 회상(+19.5%)의 대폭 향상 입증.

---

## Problem & Motivation

### 연구 배경
자기회귀(AR) 비디오 월드 모델(Matrix-Game-2, LingBot-World 등)은 이전 프레임의 KV 캐시를 참조하여 인터랙티브 게임 엔진 및 로봇 시뮬레이션용 비디오를 롤아웃 방식으로 생성한다. 에이전트가 탐색 중 과거에 방문했던 장소로 돌아올 때(revisit), 생성된 비디오가 이전의 시각적 모습과 구조적으로 동일하게 유지되는 **비주얼 지속성(Visual Persistence)**이 핵심적이다.

### 풀고자 하는 문제 (Task)
- **Task**: Interactive Long-Horizon Video World Model Generation & Visual Persistence Under Scene Revisits.

```
[Sliding Window Cache] ──> 과거 프레임 파기 ──> 재방문 시 원래 장소 모습 망각 (Visual Drift)
[Naive KV Compression] ──> RoPE 위상 상쇄 ──> Key 벡터 무력화 (Phase Cancellation)
[Block-Relative Shift] ──> 슬롯 오프셋 이탈 ──> 모든 과거 메모리가 동일 위치로 붕괴 (Block-Relative Collapse)
```

### 기존 접근의 한계
1. **Addressability Failure (OOD RoPE Offsets)**:
   - 쿼리 시점 $q$에서 과거 토큰 $t$를 참조할 때 상대 거리 $\delta_{q,k} = q - t$가 학습 컨텍스트 범위를 넘어가면, RoPE(Rotary Positional Embedding) 인코딩이 훈련 분포 범위를 벗어나 주의 집중(attention) 인출이 불가능해진다.
2. **Phase Cancellation Failure in Compressed Keys**:
   - 기존의 회전 공간(rotated space)에서 키들을 단순히 평균화하면, 각 프레임의 RoPE 회전각 $\theta_f t_m$ 차이로 인해 주파수 성분 간 상쇄(Phase Cancellation)가 발생하여 신호가 소실된다.
3. **Block-Relative Shift Collapse**:
   - 상대 위치 오프셋을 최대 학습 범위로 캡(clamping)하는 기존 방식은 장기 롤아웃 시 모든 요약 슬롯이 동일한 최소 위치로 붕괴하여 개별 메모리 슬롯을 구분할 수 없다 (Remark 1).

---

## Contributions

1. **Addressability & Compression 결합 병목 규명**: 장기 생성 실패가 단지 과거 프레임의 저장 유무가 아니라 "메모리 슬롯이 주소 지정 가능한 위치에 배치되었는가"와 "압축 시 위상 상쇄가 방지되었는가"의 복합적 원인임을 수학적으로 증명 (Sec. 2).
2. **Training-Free WorldTrace Framework 제안**:
   - **Virtual Position Assignment (Def. 1)**: Slot Rank 기반의 오프셋 정의를 통해 롤아웃 길이에 관계없이 모든 슬롯을 항상 학습 분포 내부(in-distribution) 위치에 고정.
   - **WorldTrace-Field (Def. 2)**: Canonical Space(unrotated)에서의 평균화 후 Virtual Position으로의 재인코딩을 통해 연속적인 시각적 일관성 유지.
   - **WorldTrace-Landmark**: 씬 진입(scene-entry) 전환 이벤트를 정규 키 코사인 거리로 감지하고 랜드마크 프레임을 동적으로 갱신하여 에피소드 회상 구현.
3. **LoopBench Benchmark 제안**: 장기 탐색 후 초기 방문 장소로 복귀하는 닫힌 루프(ABA, ABCA, ABCDA) 롤아웃 시 비주얼 지속성을 정밀 측정하는 벤치마크 설계.

---

## Method

### 1. Overall Cache & Virtual Position Architecture

WorldTrace는 전체 주의집중 윈도우 $L_{\text{attn}}$을 $N_r$개의 최근 윈도우(Recent Window)와 $N_s$개의 요약 캐시(Summary Cache)로 분할한다 ($N_s + N_r = L_{\text{attn}}$).

```
   ┌──────────────────────────────────────────────────────────────────────────────────┐
   │                          Local Attention Window L_attn                           │
   ├──────────────────────────────────────────────────────┬───────────────────────────┤
   │                 Summary Cache S (N_s slots)          │ Recent Window R (N_r)     │
   │               Virtual Positions (t^v_s)              │ Verbatim Latent Frames    │
   ├──────────────┬──────────────┬──────────┬─────────────┼──────────────┬────────────┤
   │   Slot s=0   │   Slot s=1   │   ...    │ Slot s=N_s-1│  Slot r=0    │  Slot r=1  │
   │   q - 5      │   q - 4      │          │   q - 2     │    q - 1     │     q      │
   └──────────────┴──────────────┴──────────┴─────────────┴──────────────┴────────────┘
```

#### Definition 1 (WorldTrace Slot Indexing)
쿼리 위치 $q$에서 요약 슬롯 $s$ ($s = 0, \dots, N_s - 1$)의 가상 위치 $t^v_s$는 슬롯 순위(rank)에 의해 고정된다:
$$t^v_s = q - (L_{\text{attn}} - 1 - s)$$

- 임의의 롤아웃 길이 $N$에서도 오프셋 $(L_{\text{attn}} - 1 - s)$은 일정하므로 모든 요약 슬롯이 항상 학습 분포 범위 $[t^v_{\text{min}}, t^v_{\text{max}}]$에 유지된다.

---

### 2. WorldTrace-Field: Canonical Key Averaging

과거 프레임 세트 $\{t_1, \dots, t_M\}$을 하나의 요약 슬롯 $s$로 압축할 때, RoPE 회전을 제거한 정규 공간(canonical space)에서 평균화한 후 Target Virtual Position $t^v_s$로 다시 인코딩한다.

```
 Source Frames {K_{t_m}} ──[ Unrotate R(-θ t_m) ]──> Canonical Keys {K_can}
                                                             │
                                                     [ Group Average 1/M ]
                                                             │
                                                             ▼
 Target Compressed Key K_field ◄──[ Re-rotate R(θ t^v_s) ]── Canonical Mean Key
```

#### Definition 2 (WorldTrace-Field Operator)
$$K^{\text{field}}_f(t^v) = R(\theta_f t^v) \cdot \frac{1}{M} \sum_{m=1}^{M} R(-\theta_f t_m) K^f_{t_m}$$

- **Mean Attention Preservation (Prop. 2)**: 정규 공간 평균화는 위상 상쇄를 방지하여, 주의 집중 소프트맥스 적용 이전의 평균 주의집중 점수(mean pre-softmax attention score)를 완벽히 보존한다.

---

### 3. WorldTrace-Landmark: Scene-Entry Landmark Traces

에피소드 회상을 위해 환경의 씬(scene) 변화 지점을 자동 감지하여 verbatim 정규 키를 저장한다.

1. **Scene-Entry Detector**:
   - 연속된 프레임 간 정규 키 코사인 거리를 측정:
     $$\text{dist}(t) = 1 - \cos\left(K_{\text{can}}(t), K_{\text{can}}(t-1)\right)$$
   - $\text{dist}(t) > \tau$ 인 경우 해당 프레임을 Landmark로 등록.
2. **Frozen Canonical Storage & Dynamic Re-rotation**:
   - 등록된 랜드마크의 정규 키 $K^{\text{landmark}}_{\text{can}}$를 고정 보존하며, 쿼리 $q$ 발생 시 슬롯 가상 위치 $t^v_s$로 동적 회전 적용:
     $$K^{\text{landmark}}_s(t^v_s) = R(\theta_f t^v_s) K^{\text{landmark}}_{\text{can}}$$

상세 발췌 → [excerpt](../source/paper/Addressable_Memory_for_Video_World_Models_2026_NVIDIA.md)

---

## Experiments & Results

### Benchmark Datasets & Setup
- **Evaluation Backbone**: Matrix-Game-2 (1.3B 디스틸드 자기회귀 비디오 월드 모델, $L_{\text{train}} = 6$ AR chunks, 3D-RoPE 적용) 및 LingBot-World (14B).
- **LoopBench Benchmark**:
  - **ABA**: 직진 후 180도 반전하여 출발 지점 A로 복귀 ($N=16$).
  - **ABCA**: 삼각 궤적 복귀 ($N=17$).
  - **ABCDA**: 사각형 궤적 복귀 ($N=16$).
- **Evaluation Metrics**:
  - **TempSSIM** ($\uparrow$): 연속 프레임 간 SSIM (시각적 일관성).
  - **Local Scene Drift** ($\downarrow$): 연속 청크 간 CLIP 피처 거리 (드래프트 감소 측정).
  - **Position-Aligned CLIP (PAC)** ($\uparrow$): 동역학적으로 매칭되는 복귀 시점과 최초 방문 시점 간 CLIP 유사도 (에피소드 회상 정확도).

---

### Quantitative Results

#### 1. 장기 롤아웃 시 시각적 일관성 비교 (MG2-1.3B, Horizon $N=48$ Chunks)

| Cache Method | TempSSIM ($\uparrow$) | Local Scene Drift ($\downarrow$) |
| :--- | :---: | :---: |
| Sliding Window (Default Baseline) | 0.472 | 0.0305 |
| Canonical Averaging + Block-Relative | 0.530 | 0.0339 |
| Canonical Averaging + Centroid-Linear | 0.479 | 0.0297 |
| **WorldTrace-Field (Ours)** | **0.545** | **0.0295** |

- **분석**: Block-Relative는 슬롯 붕괴로 인해 Sliding Window 수준으로 퇴화하며, Centroid-Linear는 가상 위치가 $N$에 의존하여 지터가 발생함. 반면 **WorldTrace-Field**는 롤아웃 $N=48$에서도 안정적인 일관성을 달성함 (+15.5% 상대 향상).

#### 2. LoopBench 장기 재방문 에피소드 회상 (PAC Metric)

```
[Sliding Window]     PAC = 0.638  (원래 장소 복귀 시 씬 붕괴 및 다른 구조 생성)
[WorldTrace-Landmark] PAC = 0.833  (+19.5% 향상, 최초 방문 씬 A의 구조 및 색상 완벽 복원)
```

---

## Analysis

### Strengths & Significance
1. **Zero-Retraining (Training-Free)**: 모델 파라미터 재학습이나 파인튜닝 없이, 추론 시점의 KV 캐시 포지셔닝 및 정규화 변환만으로 장기 메모리 한계를 극복함.
2. **수학적 엄밀성**: Slot Rank 가상 위치 할당과 Canonical Key Averaging의 결합이 Mean Attention Score를 보존함을 명확히 정식화.
3. **LoopBench 표준 제시**: 그동안 비디오 월드 모델 평가에서 미비했던 "장기 탐색 후 원래 위치 복귀 시의 비주얼 지속성"을 객관적으로 측정하는 신규 벤치마크 정립.

### Limitations
1. **Lossy Compression**: $N_s$ 슬롯에 $T/N_s$ 비율로 프레임을 압축하므로 롤아웃이 극도로 길어지면 슬롯 내부 프레임의 세부 디테일이 점진적으로 블러링됨.
2. **Landmark Slot Capacity**: 랜드마크 슬롯 수가 $N_s$개로 제한되어 있어, $N_s$개 이상의 서로 다른 씬을 거칠 경우 가장 오래된 랜드마크가 밀려나 복원 불가.

### Future Work & Improvements
1. **Geometry-Aware Canonical Keys**: 카메라 포즈 변환(Plücker / Warped-RoPE)과의 결합을 통해 공간 3D 좌표계를 맞춘 후 canonical averaging 수행.
2. **Learned Scene-Entry Policy**: 씬 변화 감지 역치 $\tau$를 고정하지 않고 에이전트 액션 및 세그멘테이션 로짓 기반의 학습된 정책 적용.

---

## References

- **Paper URL**: [https://arxiv.org/abs/2608.07408](https://arxiv.org/abs/2608.07408)
- **Project Webpage**: [https://research.nvidia.com/labs/sil/projects/WorldTrace/](https://research.nvidia.com/labs/sil/projects/WorldTrace/)
- **Excerpt File**: [Addressable_Memory_for_Video_World_Models_2026_NVIDIA.md](../source/paper/Addressable_Memory_for_Video_World_Models_2026_NVIDIA.md)
