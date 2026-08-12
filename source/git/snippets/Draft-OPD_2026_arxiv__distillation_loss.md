# Acceptance-Aware Distillation Loss

> 출처: [분석 문서](../../../report/[paper][git]_Draft-OPD_On-Policy_Distillation_for_Speculative_Draft_Models_2026_arxiv.md) / submodule 경로: `source/git/Draft-OPD_bingyang-lei/verl/verl/trainer/distillation/losses.py`

## 설명

Draft-OPD의 acceptance-aware KL objective를 구현한 핵심 손실 함수군. 두 토큰 스트림을 구분해 처리한다.

- **Response stream** (accepted/verified token): forward KL — target 분포로 가중치를 둬 verified 상태에서 drafter가 target에 부합하도록.
- **Rejected draft stream** (거부된 draft suffix token): reverse KL (`rejected_draft_use_reverse_kl=True`) — draft 자신의 높은 확률 mode가 target과 다를 때 벌점.

`_combine_sampled_reverse_forward_losses`가 `force_reverse_kl` 플래그로 rejected stream엔 항상 reverse KL(sampled k1/k3 estimator)을 적용하고, response stream엔 forward/reverse 조합을 적용한다. 거부 token의 position decay 가중치는 `_build_rejected_draft_position_weights`에서 `γ^{offset-1}`로 계산된다 (offset = draft block 내 위치).

최종 결합(`distillation_loss`)은 두 스트림을 가중 평균한다: `(w_resp·resp_sum + w_rej·rej_sum) / (w_resp·count + w_rej·eff_count)`. `response_stream_weight` / `rejected_draft_stream_weight`가 논문의 λ_acc, λ_rej 역할(실험에선 둘 다 1).

## 코드

```python
# verl/verl/trainer/distillation/losses.py — rejected-token position decay weights
def _build_rejected_draft_position_weights(
    *, model_output, rejected_draft_mask, loss_config
):
    enabled = bool(getattr(loss_config, "rejected_draft_position_decay_enabled", True))
    if not enabled:
        return rejected_draft_mask.to(dtype=torch.float32), False

    decay = float(getattr(loss_config, "rejected_draft_position_decay", 0.9))
    offsets = model_output.get("opd_rejected_draft_offsets")
    # w_k = gamma^(offset - 1)
    exponents = (offsets.to(dtype=torch.float32) - 1.0).clamp_min(0.0)
    weights = torch.pow(offsets.new_tensor(decay, dtype=torch.float32), exponents)
    return weights * rejected_draft_mask.to(dtype=torch.float32), True


# per-stream KL combination: force_reverse_kl forces sampled reverse-KL for rejected tokens
def _combine_sampled_reverse_forward_losses(
    *, student_log_probs, teacher_log_probs, loss_config, mask, stream_name,
    force_reverse_kl=False,
):
    reverse_weight = _loss_weight(loss_config, "reverse_kl_weight", 1.0)
    forward_weight = _loss_weight(loss_config, "forward_kl_weight", 0.0)
    sampled_loss_mode = _sampled_kl_loss_mode(loss_config)
    if force_reverse_kl:
        reverse_losses = kl_penalty(
            logprob=student_log_probs, ref_logprob=teacher_log_probs,
            kl_penalty=sampled_loss_mode,
        )
        return reverse_losses, {f"distillation/{stream_name}_reverse_kl_loss": ...}

    # response stream: combine reverse + local bernoulli forward KL by weights
    total_losses = None
    if reverse_weight > 0:
        reverse_losses = kl_penalty(logprob=student_log_probs,
            ref_logprob=teacher_log_probs, kl_penalty=sampled_loss_mode)
        total_losses = reverse_losses * reverse_weight
    if forward_weight > 0:
        forward_losses = _local_bernoulli_forward_kl(
            student_log_probs=student_log_probs,
            teacher_log_probs=teacher_log_probs, loss_config=loss_config)
        total_losses = (total_losses or 0) + forward_losses * forward_weight
    return total_losses, ...


# final aggregation: weighted average of accepted(response) + rejected streams
# (supervised, non-policy-gradient path)
response_weight = float(getattr(loss_config, "response_stream_weight", 1.0))
rejected_weight = float(getattr(loss_config, "rejected_draft_stream_weight", 1.0))
response_sum = (distillation_losses * effective_response_mask).sum()
rejected_sum = (rejected_draft_losses * rejected_loss_weights).sum()
denom = (response_weight * global_response_count
         + rejected_weight * global_rejected_effective_count)
distillation_loss = (response_weight * response_sum
                     + rejected_weight * rejected_sum) / denom
```
