# OPSA Core Implementation Snippets

> 소스 위치: [source/git/On-Policy-Self-Adaptation_DripNowhy/slime/slime/backends/megatron_utils/opsa.py](../../git/On-Policy-Self-Adaptation_DripNowhy/slime/slime/backends/megatron_utils/opsa.py)
> 분석 문서: [report/[paper][git]_Does_On-Policy_Distillation_Really_Distill_From_Noisy_Teacher_to_Self-Improvement_2026_Purdue.md](../../../report/[paper][git]_Does_On-Policy_Distillation_Really_Distill_From_Noisy_Teacher_to_Self-Improvement_2026_Purdue.md)

---

## 1. OPSA 어드밴티지 및 마스크 계산 (`compute_opsa`)

```python
# File: slime/slime/backends/megatron_utils/opsa.py

def compute_opsa(
    log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    *,
    token_fraction: float,
    mode: str,
    entropies: list[torch.Tensor] | None = None,
    advantage_min: float = -1.0,
    advantage_max: float = -0.5,
    fixed_advantage: float | None = None,
) -> OPSAOutput:
    """Compute OPSA advantages for a DP-local packed response batch.

    Valid response tokens from every sample in log_probs are concatenated conceptually.
    The lowest floor(token_fraction * N) actor log-probability tokens are selected.
    Unselected tokens receive zero advantage and a zero loss mask.
    """
    _validate_inputs(log_probs, loss_masks, entropies)

    device = log_probs[0].device
    flat_log_probs = torch.cat([value.detach().to(device=device, dtype=torch.float32) for value in log_probs])
    flat_valid_mask = torch.cat([value.to(device=device).bool() for value in loss_masks])
    flat_advantages = torch.zeros_like(flat_log_probs, dtype=torch.float32)
    flat_opsa_mask = torch.zeros_like(flat_log_probs, dtype=torch.float32)

    valid_indices = torch.nonzero(flat_valid_mask, as_tuple=False).flatten()
    valid_count = valid_indices.numel()
    if valid_count > 0:
        # 하위 token_fraction (기본 20%) 개수 계산
        selected_count = max(1, math.floor(token_fraction * valid_count))
        selected_count = min(selected_count, valid_count)
        
        # 학생 모델의 온폴리시 log-probability 기준 오름차순 정렬
        order = torch.argsort(flat_log_probs[valid_indices], stable=True)
        selected_indices = valid_indices[order[:selected_count]]
        flat_opsa_mask[selected_indices] = 1.0

        if mode == "fixed":
            flat_advantages[selected_indices] = fixed_advantage
        else:
            # 엔트로피 기반 동적 어드밴티지 계산: A_i^dyn = A_max + (A_min - A_max) * entropy_rank
            flat_entropies = torch.cat([value.detach().to(device=device, dtype=torch.float32) for value in entropies])
            selected_entropies = flat_entropies[selected_indices]
            entropy_range = selected_entropies.max() - selected_entropies.min()
            if entropy_range <= 1e-12:
                entropy_rank = torch.ones_like(selected_entropies)
            else:
                entropy_rank = (selected_entropies - selected_entropies.min()) / entropy_range
            
            # advantage_max = -0.5, advantage_min = -1.0
            # flat_advantages = -0.5 + (-1.0 - (-0.5)) * entropy_rank = -0.5 - 0.5 * entropy_rank
            flat_advantages[selected_indices] = advantage_max + (advantage_min - advantage_max) * entropy_rank

    split_sizes = [value.numel() for value in log_probs]
    advantages = [
        value.reshape_as(log_prob).to(dtype=log_prob.dtype)
        for value, log_prob in zip(flat_advantages.split(split_sizes), log_probs, strict=True)
    ]
    opsa_masks = [
        value.reshape_as(loss_mask).to(dtype=loss_mask.dtype)
        for value, loss_mask in zip(flat_opsa_mask.split(split_sizes), loss_masks, strict=True)
    ]
    return OPSAOutput(advantages=advantages, loss_masks=opsa_masks, metrics=metrics)
```

---

## 2. 외부 보상 및 참조 모델 완전 배제 설정

```python
# File: slime/slime/backends/megatron_utils/opsa.py

def requires_reference_model(args) -> bool:
    """Return whether training must instantiate a separate reference model."""
    if args.advantage_estimator == "opsa":
        return False
    return args.kl_coef != 0 or args.use_kl_loss

# File: slime/slime/rollout/opsa.py

async def reward_func(args, sample, **kwargs):
    """Return zero task reward; OPSA supplies the token-level learning signal."""
    if isinstance(sample, list):
        return [0.0] * len(sample)
    return 0.0
```
