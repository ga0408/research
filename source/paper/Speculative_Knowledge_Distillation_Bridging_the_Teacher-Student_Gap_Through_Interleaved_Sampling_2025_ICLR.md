# Speculative Knowledge Distillation — 핵심 발췌

> 출처: [분석 문서](../../report/[paper][git]_Speculative_Knowledge_Distillation_Bridging_the_Teacher-Student_Gap_Through_Interleaved_Sampling_2025_ICLR.md) / 원본: [arXiv:2410.11325](https://arxiv.org/abs/2410.11325) (ICLR 2025)

## Problem Setup — 발산 메트릭 (Eq. 1)

교사 `M_t`, 학생 `M_s`. 과제 T의 (x,y) 쌍에 대해 **토큰 단위 분포 발산의 평균**을 최소화:

```
D(M_t || M_s)(y | x) = (1/L_y) Σ_{i=1}^{L_y} D( M_T(·|y_<i, x) || M_s(·|y_<i, x) )   (Eq. 1)
```

- `L_y`: 출력 길이. 학습 목적 = Eq.1 최소화로 학생이 교사 모방.
- D는 KL(기본, forward KL D_KL(P_teacher||Q_student)), 역방향 KL, JSD 모두 테스트(Appendix A).

## 왜 기존 KD가 실패하는가 (motivation)

- **Supervised KD**: 고정 정답 데이터셋 학습 → 학습 분포 ≠ 추론 시 학생 생성 분포 (train-inference mismatch). 학생이 과거 실수를 교정하는 법을 못 배움.
- **On-policy KD**: 학생 자기 생성 샘플로 학습 → mismatch는 해소되지만, 학생이 **저품질/OOD 샘플** 생성(특히 학습 초기) → 교사가 익숙하지 않은 상태에 부정확한 피드백. 또한 학생 초기화에 매우 민감.

이론적 근거(모방학습, Ross et al. 2011 DAgger): **샘플 생성을 교사→학생으로 점진적 이동**하는 것이 효과적. SKD는 이것을 speculative sampling으로 실현.

## Method — SKD Algorithm 1 (γ=1 단순화; 실제 구현 γ=5)

```
입력: 학생 M_s, 교사 M_t, 프롬프트 {x_j}, 디코딩 길이 α, top-K, 발산 D
for step = 1..N:
  for i = 1..α:
    (1) 학생에서 샘플:  y_i ~ M_s(·|y_<i, x_j)
    (2) if y_i ∉ top_K( M_t(·|y_<i, x_j) ):    # 교사 top-K 밖이면 거부
          y_i ~ M_t(·|y_<i, x_j)               # 교사에서 재샘플하여 교체
  (3) 경사하강법으로 D(M_t || M_s)(y | x) (Eq.1) 최소화
```

핵심 설계:
- **수락 기준 = 학생 토큰이 교사의 top-K 안에 있는가.** 표준 speculative decoding의 확률비율 `min(1,p/q)` 기반 수락은 결국 교사 분포에서 샘플링하게 되어 **supervised KD로 퇴화**. SKD는 학생 분포를 유지하면서 "교사가 생성할 법한" 토큰만 걸러낸다.
- **거부 시**: 후속 토큰 폐기 후 교사 분포에서 재샘플 → 교체. autoregressive 모델에서 조기 오류 전파 방지.
- **자연스러운 transition (implicit curriculum)**:
  - 학습 초기 학생 품질 낮음 → 대부분 거부/교사 교체 → ≈ supervised KD.
  - 학습后期 학생 품질 향상 → 대부분 수락 → ≈ on-policy KD.
  - K→0 (전부 거부) = supervised KD; K→∞ (전부 수락) = on-policy KD. SKD는 두 극단의 **일반화**.

## 하이퍼파라미터 / Setup

- 모델族: Gemma (교사 7B-it → 학생 2B), Qwen2 (교사 7B-it → 학생 0.5B). 교사는 과제 SFT, 학생 초기화 = IT 또는 SFT.
- 과제: Flores-200 Assamese→English 번역(COMET), DialogSum 요약(ROUGE-L), GSM8K 산술추론(정확도), UltraInteract 수학 지시수행(MATH/GSMplus/ASDiv/SVAMP 평가; task-agnostic).
- 데이터: 과제별 ~1K(저자원 100), task-agnostic 1K/10K.
- K=25 고정(탐색 없음, top-k 샘플링 통상값). lr=1e-5, warmup 0.1, dropout 0.1, 3 epoch(번역 10), greedy 디코딩 평가. 8×A100 80GB.

## Results — Table 1 (IT 학생 초기화, K=25)

Gemma 7B→2B / Qwen 7B→0.5B:

| Baseline | Trans(COMET) | Summ(ROUGE-L) | GSM8K(Acc) | Trans'(COMET) | Summ' | GSM8K' |
|---|---|---|---|---|---|---|
| SFT | 72.5 | 31.7 | 18.7 | 57.6 | 29.2 | 31.8 |
| Supervised KD | 73.3 | 34.1 | 22.5 | 57.5 | 31.3 | 35.0 |
| On-policy KD | 36.1 | 34.1 | 25.3 | 53.0 | 31.2 | 33.9 |
| ImitKD | 74.8 | 34.9 | 26.2 | 55.9 | 30.9 | 33.2 |
| **SKD** | **75.3** | **35.0** | **29.1** | **57.1** | **31.7** | **36.6** |

- On-policy KD는 IT 초기화 번역에서 붕괴(COMET 36.1) — 초기화 민감성.
- Gemma 7B→2B 기준 SFT 대비: 번역 +41.8%, 요약 +230%, 산술 +160%. 수학 지시수행: MATH +198%, GSMplus +360%.

## 주요 발견 (Results §5 + Appendix)

- **§5.3 초기화**: on-policy KD는 나쁜 학생 초기화에서 저조(낮은 수준에 갇힘); SKD는 IT·SFT 모두 robust. ImitKD(시퀀스 단위 혼합) < SKD(토큰 단위 혼합).
- **§5.4 저자원(100)**: on-policy 저품질로 약세; SKD > supervised·on-policy 전 과제. SFT 과적합이 post-KD 성능 저하 → end-to-end SKD(SFT 우회)가 저자원에서 유리.
- **§5.5 두 단계 ablation**(전반 supervised KD + 후반 on-policy): 놀랍게도 둘 중 하나만 쓴 것보다도 종종 열세. 시퀀스 단계 단순 혼합은 토큰 단위 자연 confidence 부재로 data discrepancy → SKD의 adaptive 토큰 단위 혼합이 필수임을 시사.
- **Appendix B**: K∈[5,50] 넓은 범위가 두 baseline 모두 초과; 극단값(너무 작/큰)은 비최적. 선정 K=25.
- **Appendix C**: adaptive K(감소)는 constant K=25에 못 미침 — 인위적 supervised행동 강제가 off-policy 토큰 도입해 상해. 모델이 자연 전이함이 더 낫다.
- **Appendix K**: SKD 학생을 speculative decoding draft로 사용 시 token acceptance 71→85% 향상 → 1.2× 가속.
- **Appendix L/M**: 거부율 분석; SKD가 교사 직접 샘플링 대비 연산 50% 절감(추정).
- **Appendix N**: SKD가 모든 과제/초기화에서 최저 validation loss.

## Baseline 정의 (§3.1)

- **Supervised FT**: `L_SFT = -log p_s(y|x)`, 고정 데이터셋.
- **SeqKD**: 교사 생성 시퀀스로 MLE (SFT가 항상 우세해 제외).
- **Supervised KD (Sanh 2020)**: 고정 정답 y에 대해 Eq.1.
- **On-policy KD (Agarwal 2024)**: 학생 self-generate 시퀀스에 Eq.1.
- **ImitKD (Lin 2020)**: 학생 생성 vs 정답을 50:50 시퀀스 단위 무작위 혼합 후 Eq.1 (supervised+on-policy naive 결합).
