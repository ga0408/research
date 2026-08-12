# Draft-OPD: On-Policy Distillation for Speculative Draft Models — 핵심 발췌

> 출처: [분석 문서](../../report/[paper][git]_Draft-OPD_On-Policy_Distillation_for_Speculative_Draft_Models_2026_arxiv.md) / 원본: [arXiv:2605.29343](https://arxiv.org/abs/2605.29343)

## Problem: Offline-to-Inference Mismatch

SFT는 고정된 target 생성 trajectory로 drafter를 학습하지만, speculative decoding에서는 drafter가 **자기 정책하에 제안한 block**이 검증 대상이 된다. 따라서 accepted length τ를 결정하는 상태는 offline target trajectory가 아니라 **draft-induced inference state**다.

- SFT warm-up 후 continued offline SFT는 τ가 plateau에 도달 (Figure 1).
- OPD data로 SFT를 계속하면 오히려 τ가 감소하기도 함.

## Why Standard OPD Fails for Draft Models (Figure 2)

- **(a) Draft-only rollout**: EAGLE/DFlash-style drafter는 standalone autoregressive 생성기가 아님 → self-rollout이 반복적·저품질로 붕괴.
- **(b) Naive target-assisted rollout**: 무손실 검증이 verified continuation을 target 분포를 따르게 만듦 → rejected draft token(가장 정보력 높은 오류)이 버려져 on-policy signal 소실. 결국 target trajectory KL-loss SFT로 퇴화.

## Draft-OPD Method (3 coordinated designs)

### Design 1: Rollout with error-position collection
Target-assisted rollout으로 안정적 continuation 확보. 각 draft block의 시작 위치를 **anchor** a_m 으로 기록.
- verified prefix 끝 = a_m, draft가 K-token block d_m 제안, target이 r_m 개 수용 → 다음 anchor = a_m + r_m.
- Anchor는 draft model이 실제로 행동한 (local action) 위치를 보존. rollout 자체는 고품질 target 샘플이지만, 각 anchor는 inference 시 draft가 block을 제안했던 상태.
- Rollout 시 이미 계산한 target hidden state를 replay에 재사용 가능.

### Design 2: Replay for log-probability computation
모든 anchor로부터 drafting을 replay하여 token-level student/teacher log-prob 계산.
- replay context c_m = (x, y_≤a_m). draft block d_m 을 replay.
- 각 drafted token d_{m,k}: student log q_{m,k}(d_{m,k}), teacher log p_{m,k}(d_{m,k}) on 동일 draft prefix.
- 검증 결과로 자연스럽게 partition:
  - I_acc = {(m,k) : 1 ≤ k ≤ r_m} (수용)
  - I_rej = {(m,k) : r_m < k ≤ K} (거부, 첫 실패 token + 이후 suffix)

### Design 3: Acceptance-aware distillation objective
- **Accepted tokens**: forward KL (target-weighted, reliable agreement 강화)
  L_acc = (1/|I_acc|) Σ D_KL(p_{m,k} ‖ q_{m,k})
- **Rejected tokens**: reverse KL (draft-weighted, draft 자신의 high-prob mode를 target이 불동의할 때 벌점)
  L_rej = (1/Z) Σ w_k D_KL(q_{m,k} ‖ p_{m,k})
- **Position decay** (초기 오류가 acceptance에 더 큰 영향):
  w_k = γ^{k-1}, γ=0.8
- **최종 objective** (λ_acc = λ_rej = 1):
  L_Draft-OPD = (λ_acc L_acc + λ_rej L_rej) / (λ_acc + λ_rej)

## KL 방향 근거 (Appendix B)
- Accepted: target-weighted cross-entropy J_acc(q)=E_{y~p}[-log q(y)] = H(p) + D_KL(p‖q) → forward KL.
- Rejected: draft-weighted disagreement J_rej(q)=E_{y~q}[log(q(y)/p(y))] = D_KL(q‖p) → reverse KL.
- 단일 KL 방향은 이 구분을 무시: all-forward는 rejected 오류를 reliable state처럼, all-reverse는 verified accepted를 error state처럼 취급.

## Preliminary 수식
- Speculative decoding: draft q_φ 가 block K 제안, target p_θ 가 병렬 검증, longest valid prefix 수용 (무손실).
- Standard OPD: L_OPD = E_{s_t} [D_KL(p_θ^t ‖ q_φ^t)], student 자기 정책 trajectory에서 teacher가 감독.

## Experiments
- Models: Qwen3-4B, Qwen3-8B, Qwen3-30B-A3B-Thinking-2507.
- Benchmarks: GSM8K, MATH-500, AIME25, MBPP, HumanEval, SWE-Lite, MT-Bench.
- Baselines: EAGLE-3, DFlash (동일 data mixture, matched FLOPs budget).
- OPD data: 16K prompt pool (GSM8K 2K + MATH 5K + AoPS 4K + CodeAlpaca 5K), response는 online target 생성.
- Draft: 5 layers (4B/8B), 8 layers (30B-A3B), block size 16. H200 GPU.

### Main results (Table 1, thinking mode, temp 0)
| Model | Method | Mean Speedup | τ |
| --- | --- | --- | --- |
| Q3-4B | EAGLE-3 / DFlash / Draft-OPD | 3.87× / 4.33× / 4.86× | 5.33 / 5.51 / 5.96 |
| Q3-8B | EAGLE-3 / DFlash / Draft-OPD | 4.06× / 4.34× / 4.89× | 5.64 / 5.19 / 5.73 |

Thinking mode 기준 EAGLE-3 대비 23%, DFlash 대비 13% 향상. 5× 이상 무손실 가속.

### Ablations (Table 3, Qwen3-4B)
| Method | MATH-500 speedup | τ |
| --- | --- | --- |
| Draft-OPD | 5.55× | 6.57 |
| w/o Weight Decay | 5.13× | 6.18 |
| All-reverse KL | 5.11× | 6.14 |
| All-forward KL | 5.34× | 6.35 |
| Random Anchors | 5.04× | 6.08 |

- Training data: OPD data로 단순 SFT(교사 응답)는 gains 없음 → on-policy distillation 자체가 효과.
- Naive target-assisted rollout (Table 4): 4.63×→4.29× (-7.3%) → draft-induced error 보존 중요.

### SGLang serving (Table 2)
모든 concurrency/task/model에서 DFlash 대비 throughput 향상, τ 평균 11.2% 증가. Qwen3-30B-A3B에서 최대 17% 가속.

## Limitations
- Training length: thinking mode train 응답 4096 cap, eval 8192 → 긴 생성 후반 상태 미흩.
- Evaluation scope: Qwen3 + DFlash architecture 한정.
- Lossless decoding만 다룸 (approximate/lossy verification은 future work).

## Thinking-Mode Drafting Gap (Appendix C)
Thinking mode 응답이 next-token NLL이 더 높음 (target 자체 불확실성 ↑) → draft가 덜 집중된 target 분포를 맞추기 어려움. Draft-OPD가 부분 완화.
