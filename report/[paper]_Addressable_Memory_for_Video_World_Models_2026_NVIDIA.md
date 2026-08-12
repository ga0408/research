> [paper] https://arxiv.org/abs/2608.07408

# Addressable Memory for Video World Models

## Summary & Outline

본 논문은 대화형 비디오 월드 모델(Interactive Video World Models)의 장기 비주얼 지속성(Visual Persistence) 문제를 해결하기 위해, 파인튜닝 없이(training-free) KV 캐시 메모리의 주소 지정 가능성(Addressability)과 정보 보존성(Informativeness)을 동시에 보장하는 **WorldTrace** 프레임워크를 제안한다.

### Outline
- **Problem & Motivation**: 생성 길이가 길어질 때 위치 인코딩(RoPE) 오프셋이 학습 범위를 벗어나 과거 관측을 읽지 못하는 Addressability 병목과, 키 평균화 시 발생하는 RoPE 위상 상쇄(Phase Cancellation)로 인해 시각적 일관성이 무너지는 문제 분석.
- **Contributions**: Slot Rank 기반의 고정 가상 위치 할당 (Def. 1), 정규 공간(canonical space) 내 키 평균화 기반 **WorldTrace-Field** (Def. 2), 랜드마크 기반 고정 키 추적의 **WorldTrace-Landmark**, 닫힌 루프(closed-loop) 재방문 벤치마크 **LoopBench** 제시.
- **Method**: 메모리를 최근 윈도우($N_r$)와 요약 캐시($N_s$)로 분할하고, 롤아웃 길이에 구속받지 않는 가상 위치 할당과 Canonical Unrotated Keys 매핑 아키텍처.
- **Experiments & Results**: 롤아웃 $N=48$ 환경에서 비주얼 일관성(+15.5%) 및 장시간 탐색 후 출발 장소 재방문 시 에피소드 회상(+19.5%) 대폭 향상.

---

## Problem & Motivation (풀고자 하는 문제)

### 직관적 상황 예시: "게임/로봇 탐색 후 원래 장소로 돌아왔을 때"

3D 로봇/게임 월드 모델에서 에이전트가 출발지 **Point A (빨간 차고)**에서 출발하여 한참 동안 탐색($B \to C \to D$)한 뒤, 다시 출발지 **Point A**로 돌아왔다고 가정한다.

```
 [출발] Point A (빨간 차고) ──> Point B (나무 숲) ──> Point C (다리) ──> Point D (동굴) ──> [복귀] Point A (???)
```

- **바람직한 결과 (Visual Persistence)**: Point A로 돌아왔을 때, 이전에 본 **'빨간 차고'**의 형태와 색상이 동일하게 생성되어야 함.
- **기존 방식의 실패 (Sliding Window / Naive Cache)**: 
  - 과거 Point A의 메모리를 잊어버리거나 읽지 못해, Point A에 도착했을 때 빨간 차고 대신 **'파란 집'**이나 **'전혀 다른 건물'**을 그려냄 (시각적 붕괴).

---

### 무엇이 문제인가? (2가지 핵심 병목)

```
========================================================================================
1. Addressability Failure (위치 주소 지정 실패)
========================================================================================
 [Query: t=100] ────────────────────── 상대 거리 δ = 100 - 0 = 100 ─────────────────────► [Memory: t=0 (Point A)]
                                (학습된 RoPE 범위: Max 18 초과!)
                                  ==> Attention이 과거 메모리를 읽지 못하고 무시함!

========================================================================================
2. Phase Cancellation Failure (위상 상쇄 실패)
========================================================================================
 Frame t=1 Key: ↗ (RoPE θ_1 회전)
 Frame t=5 Key: ↙ (RoPE θ_5 회전)  ──[ Naive Average ]──> Summed Key: · (0으로 억제!)
                                                       ==> 메모리 내용 상쇄/파괴!
```

1. **Addressability Failure (위치 주소 지정 실패)**:
   - 비디오 월드 모델은 temporal RoPE를 통해 과거 프레임과의 상대 거리 $\delta = q - t$를 계산함.
   - 탐색 시간이 길어져 오프셋 $\delta$가 학습 윈도우 $\Delta t_{\text{train}}$를 넘어서면, KV 캐시에 과거 Point A가 저장되어 있더라도 주의집중(Attention) 메커니즘이 이 메모리를 찾지 못함 (Addressation 불가).
2. **Phase Cancellation (위상 상쇄)**:
   - 메모리 용량을 줄이기 위해 서로 다른 시점의 프레임 Key들을 단순 평균하면, RoPE로 회전된 각도가 서로 반대 방향을 가리켜 벡터가 상쇄됨 (신호 파괴).

---

## Method (어떻게 해결했는가?)

WorldTrace는 파인튜닝 없이 추론 시점(inference-time)에서 **"메모리를 어디에 둘 것인가(Addressability)"**와 **"무엇을 어떻게 저장할 것인가(Informativeness)"**를 완벽히 해결한다.

```
+--------------------------------------------------------------------------------------------------+
|                                    WorldTrace Cache Architecture                                 |
+--------------------------------------------------------------------------------------------------+
                                     Attention Window L_attn
                   ┌──────────────────────────────────────────┬────────────────────────┐
                   │       Summary Cache S (N_s slots)        │ Recent Window R (N_r)  │
                   │      Fixed Virtual Positions (t^v_s)     │ Verbatim Latent Frames │
                   └──────────────────────────────────────────┴────────────────────────┘
                                         ▲                                ▲
                                         │                                │
                       [Canonical Key Operator]                  [Verbatim Recent]
                                         │
        ┌────────────────────────────────┴────────────────────────────────┐
        │                                                                 │
  [WorldTrace-Field]                                             [WorldTrace-Landmark]
  - 정규 공간(Unrotated)에서 평균 계산                          - 씬 전환 지점(Scene-Entry) 감지
  - 위상 상쇄 없는 시각적 일관성 유지                             - 랜드마크 프레임 정규 키 고정 저장
```

---

### 1. 가상 위치 할당 (Virtual Position Assignment)

- **아이디어**: 과거 메모리 슬롯을 절대 시간 $t$ 대신, 항상 모델이 학습했던 안전한 상대 거리 범위 내의 **슬롯 순위(Slot Rank)** 위치 $t^v_s$에 배치한다.
- **수식**: $t^v_s = q - (L_{\text{attn}} - 1 - s)$
- **효과**: 롤아웃이 1000 프레임 이상으로 길어져도, 모델은 항상 훈련 받은 인-디스트리뷰션(In-Distribution) 오프셋 범위 내에서 과거 요약 슬롯을 안전하게 주소 지정(Address)할 수 있음.

---

### 2. 해결 방식 A: WorldTrace-Field (연속적 시각 일관성)

시간 흐름에 따라 변하는 배경과 움직임을 매끄럽게 보존하기 위한 방식이다.

```
[Source Frames Keys] ───► 1) RoPE 회전 제거: R(-θ_m) * K_m ───► Canonical Space (정규 공간)
                                                                       │
                                                               2) 위상 상쇄 없는 평균 계산
                                                                       │
[Compressed Key] ◄─── 3) Target 가상 위치로 재인코딩: R(θ t^v_s) ◄───┘
```

1. **Unrotate (정규화)**: 각 프레임 Key에서 기존 RoPE 회전을 제거하여 정규 공간(Canonical Space)으로 되돌린다.
2. **Canonical Averaging (평균화)**: 정규 공간에서는 모든 Key의 방향이 정렬되어 있으므로, 위상 상쇄 없이 안전하게 평균을 구한다.
3. **Re-rotate (재인코딩)**: 평균화된 Key를 해당 요약 슬롯의 가상 위치 $t^v_s$ 각도로 새로 인코딩한다.

---

### 3. 해결 방식 B: WorldTrace-Landmark (재방문 에피소드 회상)

장소 재방문 시 과거의 주요 장소를 선명하게 기억해내는 방식이다.

```
Incoming Frame ──► [Consecutive Distance Check: cos_dist(K_can(t), K_can(t-1)) > τ]
                          │
                          ├──► [YES]: Scene-Entry Event! (새 장소/방문 전환 감지)
                          │           ==> 정규 키(Canonical Key)를 Frozen 상태로 Landmark Slot에 보관!
                          │           ==> 쿼리 시점에 Virtual Position t^v_s 로 동적 회전하여 인출!
                          └──► [NO] : 일반 이동 (유지)
```

1. **Scene-Entry Detection**: 프레임 간 정규 키 코사인 거리를 측정하여 갑작스러운 배경 변화(장소 전환 등)가 일어난 지점을 **Landmark**로 지정한다.
2. **Frozen Canonical Key**: 랜드마크 프레임의 정규 키를 고정 보존하다가, 쿼리가 발생할 때 가상 위치 $t^v_s$로 동적 회전하여 완벽한 시각적 모습을 복원한다.

상세 발췌 → [excerpt](../source/paper/Addressable_Memory_for_Video_World_Models_2026_NVIDIA.md)

---

## Experiments & Results (검증 결과)

### 1. LoopBench 벤치마크 (장소 재방문 검증)

WorldTrace의 효과를 검증하기 위해, 카메라가 다양한 궤적(ABA, ABCA, ABCDA)으로 탐색한 후 최초 장소 $A$로 돌아오는 **LoopBench** 벤치마크를 구축했다.

```
  [ABA Geometry]             [ABCA Geometry]            [ABCDA Geometry]
  A <=========> B            A ─────────► B             A ─────────► B
  (직진 후 180도 반전)       │            │             │            │
                             └◄── C ◄─────┘             D ◄───────── C
```

---

### 2. 성능 비교 표

#### (1) 장기 롤아웃 시 시각 일관성 (MG2-1.3B, $N=48$ Chunks)

| 구분 | TempSSIM ($\uparrow$, 일관성) | Local Scene Drift ($\downarrow$, 붕괴율) | 특이사항 |
| :--- | :---: | :---: | :--- |
| **Sliding Window (기존)** | 0.472 | 0.0305 | 과거 프레임 망각으로 씬 변경 |
| **Block-Relative (기존)** | 0.530 | 0.0339 | 모든 메모리가 한 지점으로 붕괴 |
| **WorldTrace-Field (Ours)** | **0.545** | **0.0295** | **+15.5% 일관성 향상 & 최소 붕괴** |

#### (2) 장시간 탐색 후 최초 장소 $A$ 복귀 시 회상 정확도 (LoopBench ABA Revisit)

```
========================================================================================
Position-Aligned CLIP (PAC Metric, ↑)
========================================================================================
Sliding Window Baseline : [██████████████░░░░░░] 0.638 (원래 장소 복귀 시 씬 무너짐)
WorldTrace-Landmark     : [████████████████████] 0.833 (+19.5% 향상, 원래 장소 완벽 복원!)
```

---

## Analysis (종합 평가)

### 장점 & 의의
1. **파인튜닝 없는 적용 (Zero-Retraining)**: 기존 훈련된 비디오 월드 모델의 가중치를 고정한 채, 추론 시점의 KV 캐시 포지셔닝과 정규화 변환만으로 장기 메모리 구축.
2. **수학적 보장**: Canonical Space 키 평균화가 소프트맥스 주의집중 이전의 평균 점수(Mean Attention Score)를 보존함을 명확히 증명.

### 한계점 & 확장 방향
1. **슬롯 용량의 한계**: 요약 슬롯 $N_s$ 개수가 고정되어 있어, 수십 개 이상의 씬을 넘나들 경우 오래된 랜드마크가 밀려날 수 있음.
2. **향후 과제**: 카메라 3D 포즈 정보(Plücker embedding)와 연동한 공간 정렬 메모리 확장.

---

## References

- **Paper URL**: [https://arxiv.org/abs/2608.07408](https://arxiv.org/abs/2608.07408)
- **Project Webpage**: [https://research.nvidia.com/labs/sil/projects/WorldTrace/](https://research.nvidia.com/labs/sil/projects/WorldTrace/)
- **Excerpt File**: [Addressable_Memory_for_Video_World_Models_2026_NVIDIA.md](../source/paper/Addressable_Memory_for_Video_World_Models_2026_NVIDIA.md)
