> [paper][git] https://github.com/bingyang-lei/Draft-OPD · https://arxiv.org/abs/2605.29343

# Draft-OPD — On-Policy Distillation for Speculative Draft Models

> **원본 관계**: 본 분석은 both 타입. 논문(arXiv:2605.29343)이 방법론·실험의 지적 기반이고, git repository가 그 구현체. 두 원본 모두 본문에서 상호 참조된다. 논문 핵심 발췌 → [excerpt](../source/paper/Draft-OPD_On-Policy_Distillation_for_Speculative_Draft_Models_2026_arxiv.md).

## Overview

Draft-OPD는 speculative decoding용 **training-based draft model**(EAGLE-style / DFlash-style)을 on-policy distillation(OPD)로 post-training하는 프레임워크다. 핵심 관찰: 기존 SFT는 고정된 target trajectory로 학습하지만, 추론 시 acceptance length(τ)를 결정하는 상태는 **drafter 자기 정책이 만든 block** → "offline-to-inference mismatch"로 SFT가 빠르게 plateau. Draft-OPD는 target-assisted rollout으로 안정적인 continuation을 얻되, 검증에서 노출된 **거부된 draft 오류 위치(anchor)**를 replay해 draft가 실제로 범한 오류 상태에서 교사 감독을 받게 한다. accepted token엔 forward KL, rejected token엔 reverse KL을 적용하는 acceptance-aware objective로 학습해, EAGLE-3 대비 23%, DFlash 대비 13% 향상(5×+ 무손실 가속)을 달성한다.

## Architecture — 전체 동작 flow

Draft-OPD는 학습 시 두 모델(교사 target `p_θ`, 학생 draft `q_φ`)이 협력하며, 추론 시엔 표준 무손실 speculative decoding을 그대로 쓴다(출력 분포 보존). 학습 1스텝은 아래 3페이즈로 구성된다.

```
                       Draft-OPD 학습 1스텝 (3 coordinated designs)
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                                                                           │
 │  ① ROLLOUT (target-assisted, error-position 수집)                        │
 │     prompt x ──> q_φ가 K-block 제안 ──> p_θ 병렬 검증                     │
 │       수용 r_m 토큰 / 거부 (r_m<K)                                        │
 │       verified continuation y (target 품질) + 각 block 시작 위치 = anchor │
 │       (a_m) 기록. rejected draft token/PB도 별도 수집                     │
 │              │                                                            │
 │              v                                                            │
 │  ② REPLAY (모든 anchor에서 drafting 재현, log-prob 계산)                  │
 │     anchor m ──> context c_m=(x, y_≤a_m) ──> draft block d_m 재생성       │
 │       각 d_{m,k} : student log q_{m,k} / teacher log p_{m,k}             │
 │       검증 결과로 partition → I_acc (수용) / I_rej (거부+suffix)          │
 │              │                                                            │
 │              v                                                            │
 │  ③ LOSS (acceptance-aware KL)                                            │
 │     I_acc  : forward KL  D_KL(p‖q)   (target-weighted, reliable)         │
 │     I_rej  : reverse KL  w_k·D_KL(q‖p), w_k=γ^(k-1) (draft-weighted)     │
 │     L = (λ_acc·L_acc + λ_rej·L_rej)/(λ_acc+λ_rej),  λ=1, γ=0.8          │
 │              │                                                            │
 │              v                                                            │
 │  학생(draft) 가중치 업데이트 (감독 KL 직접 역전파, 정책경사 아님)        │
 └─────────────────────────────────────────────────────────────────────────┘

 추론(inference): Draft-OPD가 바꾼 건 draft model 가중치뿐 → 표준 DFlash
 speculative decoding 절차 그대로 → target 출력 분포 무손실 보존
```

### 왜 표준 OPD는 draft model에 안 통하는가 (Figure 2)

| 접근 | 문제 | 결과 |
| --- | --- | --- |
| (a) Draft-only self-rollout | EAGLE/DFlash drafter는 standalone 생성기 아님 → 반복·저품질 붕괴 | 감독 신호 불가 |
| (b) Naive target-assisted rollout | 무손실 검증이 verified continuation을 target 분포로 고정 → rejected token(가장 정보력 높은 오류) 버려짐 | on-policy signal 소실, 결국 target-trajectory KL SFT로 퇴화 |

Draft-OPD는 (b)의 안정성은 취하되, **rejected 오류 위치를 replay로 되살려** draft가 실제로 행동한 상태에서 학습하는 점이 핵심 차별점.

## Problem & Motivation

- **연구 배경**: LLM 모델·생성 길이 증대로 추론 비용 급증. Speculative decoding은 가벼운 draft model이 token block을 제안하면 큰 target model이 병렬 검증해, target 출력 분포를 보존하면서 비싼 target 디코딩 스텝을 줄인다. 가속은 draft-target 정합도(accepted length τ)에 크게 의존.
- **풀고자 하는 문제**: training-based draft model(EAGLE-3, DFlash)의 SFT plateau. SFT는 target 생성 trajectory로 학습하지만 추론에선 draft 자기 정책 block이 평가받음 → 상태 mismatch.
- **기존 접근의 한계**: SFT warm-up 후 continued offline SFT가 τ 개선 없이 plateau; OPD data로 SFT하면 오히려 τ 감소. 표준 OPD는 학생이 full sequence를 자기 정책으로 rollout한다고 가정하지만, draft module은 short block 제안용이라 standalone rollout이 불안정.

## Contributions

1. **한계 식별**: offline SFT가 빠른 plateau에 도달하는 이유 = offline target trajectory vs draft-induced inference state의 mismatch.
2. **OPD 적용 불가 이유 설명**: draft-only rollout 불안정, target-assisted rollout은 on-policy signal 제거.
3. **Draft-OPD 제안**: error-position replay로 verification-time 오류에서 draft model을 post-training. 안정적 rollout + draft-policy 오류 보존을 동시 달성.
4. **실증**: thinking model 5×+ 무손실 가속, matched FLOPs에서 EAGLE-3/DFlash 대비 23%/13% 향상. SGLang serving throughput 검증까지.

## Method

### Design 1 — Rollout with error-position collection

Target-assisted speculative decoding로 고품질 rollout `y`를 얻는 동시에, 각 draft block의 **시작 위치를 anchor `a_m`**로 기록. step m에서 verified prefix 끝 = `a_m`, draft가 K-token block 제안, target이 `r_m`개 수용하면 다음 anchor = `a_m + r_m`. anchor는 "draft가 추론 중 실제로 block을 제안했던 상태"를 표식하되, rollout 전체는 여전히 target 품질 샘플. rollout 시 이미 계산한 target hidden state를 replay에 재사용 가능하다.

### Design 2 — Replay for log-probability computation

모든 anchor로부터 drafting을 **replay**해 token-level log-prob를 계산. replay context `c_m = (x, y_≤a_m)`에서 draft block `d_m`을 재생성. 각 drafted token `d_{m,k}`에 대해:
- student log-prob `log q_{m,k}(d_{m,k}) = log q_φ(d_{m,k} | c_m, d_{m,<k})`
- teacher log-prob `log p_{m,k}(d_{m,k}) = log p_θ(d_{m,k} | c_m, d_{m,<k})` (동일 draft prefix에서 평가)

검증 결과가 자연스럽게 토큰을 partition: `I_acc = {(m,k): 1≤k≤r_m}`, `I_rej = {(m,k): r_m<k≤K}`(첫 실패 토큰 + 이후 suffix). 이 replay는 "verified 최종 토큰으로 학습"과 다르다 — **거부된 위치까지 포함해 교사가 평가**하는 게 핵심.

코드 구현 → [anchor_replay snippet](../source/git/snippets/Draft-OPD_2026_arxiv__anchor_replay.md)

### Design 3 — Acceptance-aware distillation objective

| 토큰군 | KL 방향 | 공식 | 의미 |
| --- | --- | --- | --- |
| Accepted (`I_acc`) | **forward KL** `D_KL(p‖q)` | `L_acc = (1/|I_acc|) Σ D_KL(p_{m,k}‖q_{m,k})` | target-weighted, 검증 통과한 reliable 상태에서 부합 강화 |
| Rejected (`I_rej`) | **reverse KL** `D_KL(q‖p)` | `L_rej = (1/Z) Σ w_k D_KL(q_{m,k}‖p_{m,k})` | draft-weighted, draft 자기 high-prob mode가 target과 불동의 시 벌점 |
| Position decay | — | `w_k = γ^{k-1}`, γ=0.8 | 블록 내 초기 오류가 acceptance에 더 큰 영향 → 후속 rejected suffix는 가중치 감소 |

최종: `L_Draft-OPD = (λ_acc L_acc + λ_rej L_rej)/(λ_acc + λ_rej)`, 실험에선 `λ_acc=λ_rej=1`.

**KL 방향 근거(Appendix B)**: accepted는 target-weighted cross-entropy `E_{y~p}[-log q(y)] = H(p) + D_KL(p‖q)` → forward KL. rejected는 draft-weighted disagreement `E_{y~q}[log(q/p)] = D_KL(q‖p)` → reverse KL. 단일 KL 방향은 이 역할 구분을 무시(all-forward는 거부 오류를 reliable 상태처럼, all-reverse는 검증 통과를 error 상태처럼 취급).

코드 구현 → [distillation_loss snippet](../source/git/snippets/Draft-OPD_2026_arxiv__distillation_loss.md)

## Repository Architecture

```
Draft-OPD_bingyang-lei/
├── README.md
├── install.sh                  # sglang-dflash + verl editable install
├── data/                       # OPD 학습 prompt pool (apos/code/math/gsm8k 혼합 16K)
├── fig/
├── verl/                       # 학습 스택 (verl 프레임워크 fork)
│   ├── verl/
│   │   ├── models/transformers/
│   │   │   └── dflash_student.py   # ① anchor plan + replay + rejected-draft log-prob 수집
│   │   ├── workers/config/distillation.py  # ② DistillationLossConfig (모든 하이퍼파라미터)
│   │   └── trainer/distillation/losses.py   # ③ acceptance-aware KL 손실 결합
│   └── examples/on_policy_distillation_trainer/
│       ├── run_qwen_gsm8k_forward-ins.sh    # 공식 진입점(래퍼) → λ, γ, KL 방향 등 노출
│       ├── run_qwen_gsm8k.sh                # 본체 (Hydra config → verl.trainer.main_ppo)
│       └── run_qwen-30b-a3b.sh              # 30B-A3B용 변형
├── sglang-dflash/              # DFlash/SGLang 런타임 (학생+교사 rollout 엔진)
└── diffusion/                  # 평가 유틸리티
    ├── dflash/                 # 메인 벤치마크 워크플로우 (benchmark_sglang.py 등)
    ├── sglang-metrics.py
    └── run-sglang-metric.sh
```

학생(draft)은 `actor_rollout_ref`로, 교사(target)는 `distillation.teacher_models`로 별도 SGLang rollout 엔진에 할당. 학생 쪽 rollout 자체에 `speculative_algorithm=DFLASH` + `speculative_draft_model_path`를 켜두어, **rollout 단계가 곧 target-assisted speculative 검증**이 되도록 설계(=Design 1의 anchor 수집이 자연 발생).

학습 진입점 상세 → [training_entrypoint snippet](../source/git/snippets/Draft-OPD_2026_arxiv__training_entrypoint.md)

### 논문 ↔ 코드 파라미터 매핑

| 논문 기호 | 코드 설정 | 기본값 |
| --- | --- | --- |
| `λ_acc = λ_rej = 1` | `response_stream_weight`, `rejected_draft_stream_weight` | 1.0 |
| `γ = 0.8` (position decay) | `REJECTED_DRAFT_POSITION_DECAY` + `_ENABLED` | 0.8 / True |
| reverse KL on rejected | `REJECTED_DRAFT_USE_REVERSE_KL` | True |
| 감독 KL(정책경사 아님) | `USE_POLICY_GRADIENT`, `loss_mode` | False / "k3" |
| "Random Anchors" ablation | `RANDOM_RESPONSE_ANCHOR_ENABLED` / `_SEED` | False / 42 |
| block size 16 | DFlash config (`LM_HEAD_CHUNK_SIZE=512` 등) | — |

## Experiments & Results

### Benchmark Datasets
- 수학: GSM8K, MATH-500, AIME25
- 코드/SE: MBPP, HumanEval, SWE-bench Lite
- OOD: MT-Bench
- OPD 학습 pool: 16K prompt (GSM8K 2K + MATH 5K + AoPS 4K + CodeAlpaca 5K), 응답은 online target 생성.

### Setup
- 모델: Qwen3-4B/8B/30B-A3B-Thinking-2507. draft는 5 layers(4B/8B), 8 layers(30B-A3B). block size 16. H200 GPU.
- Baseline: EAGLE-3, DFlash (동일 data mixture, **matched FLOPs budget**로 SFT 학습량을 OPD 총량에 맞춤).
- Draft-OPD는 SFT 6 epoch checkpoint에서 OPD init 후 8 epoch.
- 메트릭: speedup ratio, 평균 acceptance length τ. (출력 분포 무손실이므로 품질은 별도 보고 안 함)

### Results

**Main (Table 1, thinking mode, temp 0)**

| Model | Method | Mean Speedup | τ |
| --- | --- | --- | --- |
| Q3-4B | EAGLE-3 / DFlash / **Draft-OPD** | 3.87× / 4.33× / **4.86×** | 5.33 / 5.51 / **5.96** |
| Q3-8B | EAGLE-3 / DFlash / **Draft-OPD** | 4.06× / 4.34× / **4.89×** | 5.64 / 5.19 / **5.73** |

Thinking 기준 EAGLE-3 대비 23%, DFlash 대비 13% 향상, 5×+ 무손실 가속. Non-thinking에선 평균 τ 6.33, 5.17× 가속 유지. temp 0.6에서도 일관되게最快(단 Q3-8B temp0.6에선 EAGLE-3 τ가 최고이나 순차 drafting이라 wall-clock은 느림).

**Ablation (Table 3, Qwen3-4B, MATH-500)**

| 변형 | Speedup | τ | 시사점 |
| --- | --- | --- | --- |
| **Draft-OPD** | **5.55×** | **6.57** | — |
| w/o Weight Decay | 5.13× | 6.18 | 후속 rejected suffix 가중치 감소 필요 |
| All-reverse KL | 5.11× | 6.14 | worst — accepted에 reverse KL 부적합 |
| All-forward KL | 5.34× | 6.35 | rejected에 forward KL은 오류 signal 약화 |
| Random Anchors | 5.04× | 6.08 | 거부 위치 집중이 무작위보다 우수 |

- **Training data ablation (Figure 4)**: OPD prompt로 교사 응답 SFT만 하면 gains 없음 → 향상은 추가 데이터가 아니라 on-policy distillation 자체에서 옴.
- **Naive target-assisted rollout (Table 4)**: 4.63×→4.29× (-7.3%) → 안정적 rollout만으론 부족, draft-induced error 보존이 필수.

**SGLang serving (Table 2)**: 모든 concurrency/task/model에서 DFlash 대비 throughput 향상, τ 평균 11.2% 증가. concurrency 32에서도 gain이 감소하지 않음(오히려 증가). Qwen3-30B-A3B에서 최대 17% 가속.

### Findings & Implications
- Draft model은 "검증 시점의 오류"에서 직접 학습할 때 SFT plateau를 넘는다. 데이터 양이 아니라 **on-policy 상태 노출**이 핵심.
- Accepted/rejected KL 방향 분리가 유효 (단일 KL 방향보다 일관되게 우수).
- 초기 거부 위치가 가장 informative → position decay 의미 부여.
- Thinking mode가 non-thinking보다 draft가 어려운 건 target 자체 NLL이 높아(불확실성↑) 분포 집중도가 낮기 때문(Appendix C). Draft-OPD가 부분 완화.

## Analysis

### Strengths & Significance
- **실용성**: lossless verification을 건드리지 않고 draft 가중치만 post-training → 기존 SD 파이프라인에 드롭인 가능, 품질 보존 자동 보장.
- **근거 명확**: offline-to-inference mismatch라는 명확한 진단에서 출발, 각 설계(rollout/replay/objective)가 그 진단에 정확히 대응. Appendix B의 local objective 유도로 KL 방향 선택을 정당화.
- **견고한 ablation**: 데이터·KL종류·anchor종류·decay 각각이 독립적으로 기여함을 입증. matched FLOPs 비교로 '단지 더 학습해서'라는 반론 차단.
- **배포 검증**: 단일 GPU 벤치마크에 그치지 않고 SGLang serving(concurrency 1–32)에서 실 throughput gain 확인.

### Limitations
- **Training length**: thinking train 응답을 4096으로 cap, eval은 8192 → 매우 긴 생성 후반 상태 미흡 (저자 인정).
- **Evaluation scope**: Qwen3 + DFlash 아키텍처 한정. 다른 모델 패밀리/EAGLE draft 구조/SGLang 외 백엔드로의 전이는 미검증.
- **Lossless only**: approximate/lossy verification 설정으로의 확장은 future work.

### Future Work / Improvements
- 더 긴 rollout으로 OPD 훈련 확장 → 긴 reasoning 후반 오류 노출 증대.
- reasoning 전용 draft model(불확실성 높은 target 분포 대응) 연구.
- EAGLE 구조·Megatron/다른 백엔드로의 전이 검증.

## References
- 논문: [arXiv:2605.29343](https://arxiv.org/abs/2605.29343)
- 코드: [github.com/bingyang-lei/Draft-OPD](https://github.com/bingyang-lei/Draft-OPD)
- 모델: [HuggingFace collection](https://huggingface.co/collections/bingyang-lei/draft-opd)
- 기반: DFlash ([arXiv:2602.06036](https://arxiv.org/abs/2602.06036)), EAGLE-3 ([arXiv:2503.01840](https://arxiv.org/abs/2503.01840)), SpecForge, SGLang, verl
