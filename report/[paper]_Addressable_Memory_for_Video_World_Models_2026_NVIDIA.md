> [paper] https://arxiv.org/abs/2608.07408

# Addressable Memory for Video World Models

## Summary & Outline

본 논문은 대화형 비디오 월드 모델(Interactive Video World Models)의 장기 비주얼 지속성(Visual Persistence) 문제를 해결하기 위해, 파인튜닝 없이(training-free) KV 캐시 메모리의 주소 지정 가능성(Addressability)과 정보 보존성(Informativeness)을 동시에 보장하는 **WorldTrace** 프레임워크를 제안한다.

### Outline
- **Problem & Motivation**: 생성 길이가 길어질 때 위치 인코딩(RoPE) 오프셋이 학습 범위를 벗어나 과거 관측을 읽지 못하는 Addressability 병목과, 키 평균화 시 발생하는 RoPE 위상 상쇄(Phase Cancellation)로 인해 시각적 일관성이 무너지는 문제 분석.
- **Theoretical Context & Paradigm Comparison**: LLM의 컨텍스트 윈도우 한계와의 연관성 및 텍스트 기반 RAG(Retrieval-Augmented Generation) 패러다임과 비디오 라텐트 KV 캐시 직접 조작 방식 간의 아키텍처적 차이 분석.
- **Contributions**: Slot Rank 기반의 고정 가상 위치 할당 (Def. 1), 정규 공간(canonical space) 내 키 평균화 기반 **WorldTrace-Field** (Def. 2), 랜드마크 기반 고정 키 추적의 **WorldTrace-Landmark**, 닫힌 루프(closed-loop) 재방문 벤치마크 **LoopBench** 제시.
- **Method & Experiments**: 메모리 가상 위치 할당 및 Canonical Unrotated Keys 매핑 아키텍처를 바탕으로, 일반 일방향 장기 롤아웃($N=48$)에서의 비주얼 일관성(+15.5%) 및 루프 재방문 벤치마크(LoopBench)에서의 에피소드 회상(+19.5%) 대폭 향상 입증.

---

## Context Window Limitations & Paradigm Comparison

### 1. LLM Context Window 한계와 비디오 월드 모델의 연관성
LLM에서 입력 텍스트가 훈련 시 지정된 컨텍스트 윈도우(Context Size)를 넘어서면 이전 대화나 정보를 잊어버리는 것과 마찬가지로, 자기회귀(Autoregressive) 비디오 월드 모델 역시 생성 프레임 길이가 훈련 컨텍스트 한계($L_{\text{train}}$)를 넘어서면 과거 시각 관측을 상실하게 된다.

그러나 비디오 월드 모델에서는 단순한 메모리 용량 초과를 넘어 **두 가지 고유한 기술적 병목**이 결합되어 나타난다:
1. **RoPE 오프셋 이탈에 따른 Addressability 붕괴**: 프레임 생성이 길어질수록 쿼리와 과거 키 간의 상대 위치 거리($\delta = q - t$)가 훈련 범위를 벗어나, 과거 관측이 KV 캐시에 저장되어 있음에도 주의집중(Attention) 점수가 0으로 수렴하여 인출에 실패한다.
2. **위상 상쇄(Phase Cancellation)로 인한 신호 파괴**: 캐시 용량을 줄이기 위해 프레임 Key 텐서를 단순 평균할 경우, RoPE 회전각 차이로 인해 시각 신호 벡터들이 서로 상쇄된다.

---

### 2. LLM RAG(Retrieval-Augmented Generation)와 WorldTrace의 구조적 비교
텍스트 LLM 분야에서는 컨텍스트 한계를 극복하기 위해 문서를 청킹하여 외부 Vector DB/FTS에 저장하고 쿼리 시점에 상위 K개 텍스트 조각을 검색해 프롬프트에 재주입하는 RAG 패러다임을 주로 사용한다. 반면 실시간 비디오 월드 모델 생성에서는 다음과 같은 이유로 RAG 방식 대신 **WorldTrace의 추론-내부 KV 캐시 텐서 조작 방식**이 필수적이다.

```
+--------------------------------------------------------------------------------------------------+
|                            LLM RAG vs Video WorldTrace Comparison                                |
+--------------------------------------------------------------------------------------------------+
  [1. LLM RAG (Text / Discrete Memory)]
  Text / Context ──► [Chunking & Vector DB/FTS Search] ──► Top-K Text Snippets ──► Prompt Injection
  - 불연속 텍스트 조각 검색 및 텍스트 프롬프트 새로 인코딩 (실시간 비디오 생성엔 오버헤드 큼)

  [2. Video WorldTrace (Continuous Latent / KV Cache Direct Manipulation)]
  KV Cache (Tensor) ──► [Unrotate Canonical Space] ──► [Virtual Position Slot Mapping] ──► Direct Attention
  - Vector DB 없이 추론 연산 내부에서 텐서(Key-Value) 레벨 직접 제어 & 지연 시간 Zero (Training-Free)
```

- **연속적 라텐트 텐서 조작**: RAG가 불연속적인 텍스트 조각(Discrete Text)을 다루는 반면, WorldTrace는 비디오 모델 내부의 고차원 연속 라텐트 텐서(Continuous Latent KV Tensors)를 직접 다룬다.
- **Zero-Latency 인프라**: 외부 DB 검색 및 텍스트 재인코딩 오버헤드가 발생하는 RAG와 달리, WorldTrace는 모델 추론 연산 내부에서 텐서 회전을 풀고 가상 위치에 매핑하므로 실시간 프레임 생성 속도에 지장을 주지 않는다 (추론 오버헤드 6% 미만).
- **RoPE 위상 정렬 (Phase Alignment)**: RAG에서는 검색된 텍스트에 새 위치 인코딩이 부여되지만, 비디오 KV 캐시 텐서 조작에서는 RoPE 회전을 역으로 제거한 정규 공간(Canonical Space)에서 평균을 계산하여 위상 상쇄 없이 주의집중 점수를 보존한다.

---

## Background: RoPE(Rotary Position Embedding)와 범위 초과(OOD)

### 1. RoPE (회전 위치 인코딩)의 직관적 개념
RoPE는 모델이 토큰(프레임)의 위치를 기억하도록 **벡터를 2차원 평면상에서 특정 각도($\theta \times t$)만큼 시계 바늘처럼 회전시키는 기술**이다.

```
 [t = 0 시점 (12시)]     [t = 1 시점 (1시)]     [t = 2 시점 (2시)]     [t = 6 시점 (6시)]
       12시                     1시                    2시                    6시
        │                      ↗                      ➔                      │
        │ 0°                   │ 30°                  │ 60°                  ▼ 180°
```

- **상대 위치(Relative Position) 계산**: 
  - 쿼리(현재 질문 시점 $q$)와 과거 키(답변 시점 $t$) 간의 Attention(주의집중) 점수를 계산할 때, 모델은 절대 시간 $t$가 아니라 두 시각 바늘의 **'각도 차이(상대 거리 $\delta = q - t$)'**를 보고 정보를 읽어온다.

---

### 2. RoPE 오프셋 범위 초과(Out-of-Distribution, OOD) 현상

```
==================================================================================================
1) 훈련 시점 (Training Window) : 최대 거리 18 까지만 학습 (각도 차이 0° ~ 180° 범위)
==================================================================================================
 [현재 질문 q=18] ── (각도 차이 δ = 18 - 0 = 18) ──► [과거 메모리 t=0] ──► 정상 작동! (익숙한 각도)

==================================================================================================
2) 생성 시점 (Long Horizon)   : 생성이 길어져 q=100 도달 (시계 바늘이 수 바퀴이나 돎)
==================================================================================================
 [Query q=100] ── (각도 차이 δ = 100 - 0 = 100) ────► [Memory t=0] ──► ❌ 읽기 실패! (OOD 범위 초과!)
                                                                      (메모리가 저장되어 있어도 
                                                                       Attention 점수가 0에 가까워짐)
```

- **문제 발생 원인**:
  - 모델을 훈련시킬 때 비디오 길이는 최대 6개 청크(약 18 프레임) 수준이었으므로, 모델은 상대 오프셋 $\delta \le 18$ 범위 내의 각도 차이만 보고 과거를 읽도록 학습되었다.
  - 하지만 실시간 생성 길이가 길어져 $q = 100$ 시점에 도달하면, $t=0$ 시점의 과거 메모리를 읽으려 할 때 오프셋이 $\delta = 100$으로 커진다.
  - **시계 바늘이 뱅글뱅글 수없이 돌아 훈련할 때 단 한 번도 본 적 없는 낯선 각도 오프셋(OOD)**이 발생한다.
  - 결과적으로 과거 메모리가 KV 캐시에 물리적으로 저장되어 있더라도, Attention 메커니즘이 이 메모리를 찾지 못해 **'메모리가 없는 것처럼 무시'**하게 된다 (Addressability Failure).

---

## Problem & Motivation (풀고자 하는 문제)

### 시각적 예시: "탐색 후 출발 지점으로 복귀할 때의 비주얼 지속성 (Visual Persistence)"

![Figure 1: LoopBench Revisit Visual Comparison](../source/paper/figures/fig1_loopbench_revisit_crop.png)

> **그림 1 분석 (논문 Figure 1 원본)**: 
> - **경로**: 출발지 A(초록 숲/도로, $t=0$) ──> B ($t=4$) ──> C ($t=8$) ──> D ($t=12$) ──> 다시 **출발지 A 복귀** ($t=16$).
> - **Sliding Window (기존)**: 과거 A의 메모리를 잊어버리거나 위치 오프셋 이탈로 인출하지 못해, 출발지로 돌아왔을 때 원래 모습이 아닌 **완전히 무너진 가짜 도로/건물(빨간색 테두리)**을 생성함.
> - **WorldTrace (Ours)**: 랜드마크 정규 키 메모리를 유지하여, 긴 탐색 복귀 후에도 최초 A 시점의 시각적 모습과 **구조적으로 정확히 일치하는 오리지널 씬(초록색 테두리)**을 복원함.

---

## Method (어떻게 해결했는가?)

![Figure 2: WorldTrace Memory Architecture Overview](../source/paper/figures/fig2_worldtrace_overview_crop.png)

> **그림 2 분석 (논문 Figure 2 원본)**: 
> - **전체 구조**: 전체 주의집중 윈도우 $L_{\text{attn}}$를 최근 윈도우 $\mathcal{R}$ ($N_r$ 슬롯)과 먼 과거 요약 캐시 $\mathcal{S}$ ($N_s$ 슬롯)로 분할.
> - **WorldTrace-Field (좌측 하단)**: 과거 프레임들의 RoPE 회전을 역으로 풀어서 정규 공간(Canonical Domain)에서 평균화한 뒤, 각 슬롯의 Virtual Position $t^v_s$로 재회전하여 시각적 일관성 유지.
> - **WorldTrace-Landmark (우측 하단)**: 씬 전환 시점의 정규 키(Canonical Key)를 Frozen 상태로 보관하고 쿼리 시점에 $t^v_s$로 동적 회전하여 재방문 회상 달성.

---

### 1. 가상 위치 할당 (Virtual Position Assignment)

- **아이디어**: 과거 메모리 슬롯을 절대 시간 $t$ 대신, 항상 모델이 학습했던 안전한 상대 거리 범위 내의 **슬롯 순위(Slot Rank)** 위치 $t^v_s$에 배치한다.
- **수식**: $t^v_s = q - (L_{\text{attn}} - 1 - s)$
- **효과**: 롤아웃이 1000 프레임 이상으로 길어져도, 모델은 항상 훈련 받은 인-디스트리뷰션(In-Distribution) 오프셋 범위 내에서 과거 요약 슬롯을 안전하게 주소 지정(Address)할 수 있다.

---

### 2. 해결 방식 A: WorldTrace-Field (일반 일방향 장기 비디오 생성용)

- **적용 대상**: 루프나 시점 anchor 재방문이 없는 **모든 일반적인 일방향 장시간 비디오 생성**.
- **원리**: 롤아웃 시간이 길어짐에 따라 과거 프레임들을 정규 공간(Canonical Space)에서 슬롯별로 이동 평균(Moving Average)하여 보존한다.
- **결과**: 재방문이 없더라도 프레임 붕괴(drift)를 방지하고 시각 일관성(TempSSIM)을 지속 유지한다.

---

### 3. 해결 방식 B: WorldTrace-Landmark (장소 재방문 / 씬 전환용)

- **적용 대상**: 에이전트가 탐색 중 과거 방문 장소로 돌아오는 **재방문(Scene Revisit) 상황**.
- **원리**: 씬 전환 시점의 정규 키를 Frozen 상태로 보관하다가, 쿼리 시점에 가상 위치 $t^v_s$로 동적 회전하여 완벽한 시각적 모습을 복원한다.

상세 발췌 → [excerpt](../source/paper/Addressable_Memory_for_Video_World_Models_2026_NVIDIA.md)

---

## Experiments & Results (실험 구조 검증)

### 1. 일반 일방향 장기 생성과 장소 재방문 검증 구조
본 논문은 기술의 범용성을 입증하기 위해 (1) 시점 anchor 재방문이 없는 일반 일방향 장기 비디오 생성과 (2) 장소 재방문(LoopBench) 두 가지 영역으로 나누어 검증을 진행했다.

```
+--------------------------------------------------------------------------------------------------+
|                                    WorldTrace Experiment Scope                                   |
+--------------------------------------------------------------------------------------------------+
                                                 │
        ┌────────────────────────────────────────┴────────────────────────────────────────┐
        │                                                                                 │
  [1. 일반 일방향 장기 롤아웃 실험 (Sec 4.2)]                      [2. 닫힌 루프 재방문 평가 (Sec 4.3)]
  - 조건: 재방문/시점 anchor 없는 일방향 생성 ($N=48$ Chunks)       - 조건: $A \to B \to A$, $A \to B \to C \to A$ 등
  - 적용: WorldTrace-Field                                         - 적용: WorldTrace-Landmark & LoopBench
  - 성과: TempSSIM +15.5% 향상 & 시각적 붕괴 방지                 - 성과: PAC +19.5% 에피소드 회상 향상
```

---

### 2. LoopBench 벤치마크 궤적 (Sec 4.3 재방문 평가)

![Figure 3: LoopBench Benchmark Geometries](../source/paper/figures/fig3_loopbench_geometries_crop.png)

> **그림 3 분석 (논문 Figure 3 원본)**: 
> LoopBench는 재방문 시점의 회상 성능을 정밀 평가하기 위한 전용 벤치마크 기하학 구조임 ($ABA \times 8$, $ABCA \times 5,5,7$, $ABCDA \times 4,4,4,4$).

---

### 3. 일반 일방향 장기 롤아웃 프레임 비교 (Sec 4.2 - N=48 AR Chunks)

![Figure 4: 48-chunk Long Rollout Frame Comparison](../source/paper/figures/fig4_long_rollout_crop.png)

> **그림 4 분석 (논문 Figure 4 원본)**: 
> - **루프 재방문이 없는 단순 일방향 롤아웃 환경**에서 $N=48$ 청크(장시간) 생성 시 비교.
> - **기존 방식들 (Sliding Window, Block-Relative, Centroid)**: 청크 $n=18$ 시점부터 시각적 왜곡 및 드래프트(빨간 테두리)가 심각하게 발생함.
> - **WorldTrace-Field (Ours)**: 루프 재방문이 없는 일방향 비디오 생성에서도 롤아웃 끝($n=48$)까지 도로와 건물의 구조가 일관되게 유지됨.

---

### 4. 정량적 성과 표

| 구분 | 적용 조건 | TempSSIM ($\uparrow$, 일관성) | Local Scene Drift ($\downarrow$, 붕괴율) | LoopBench Revisit PAC ($\uparrow$, 회상률) |
| :--- | :--- | :---: | :---: | :---: |
| **Sliding Window (기존)** | 공통 Baseline | 0.472 | 0.0305 | 0.638 |
| **WorldTrace-Field (Ours)** | **일반 일방향 장기 생성** | **0.545** (+15.5%) | **0.0295** | - |
| **WorldTrace-Landmark (Ours)** | **루프 장소 재방문** | - | - | **0.833** (+19.5%) |

---

## References

- **Paper URL**: [https://arxiv.org/abs/2608.07408](https://arxiv.org/abs/2608.07408)
- **Project Webpage**: [https://research.nvidia.com/labs/sil/projects/WorldTrace/](https://research.nvidia.com/labs/sil/projects/WorldTrace/)
- **Excerpt File**: [Addressable_Memory_for_Video_World_Models_2026_NVIDIA.md](../source/paper/Addressable_Memory_for_Video_World_Models_2026_NVIDIA.md)
