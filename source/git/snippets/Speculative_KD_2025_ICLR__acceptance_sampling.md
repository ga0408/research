# SKD Acceptance Sampling — 핵심 수락/교체 로직

> 출처: [분석 문서](../../../report/[paper][git]_Speculative_Knowledge_Distillation_Bridging_the_Teacher-Student_Gap_Through_Interleaved_Sampling_2025_ICLR.md) / 원본: `google-research/google-research` monorepo `speculative_kd/transformers/utils.py`
> (monorepo 규모가 커 submodule 대신 URL 참조: https://github.com/google-research/google-research/tree/master/speculative_kd)

## 설명

이 함수 `_speculative_kd_sampling`이 **논문 Algorithm 1의 구현체**이자 SKD의 핵심이다. HuggingFace `transformers`의 speculative decoding 검증 루틴을 SKD용으로 교체한 것이다.

동작 흐름:
1. 학생(draft/assistant)이 γ개 후보 토큰을 제안 → `candidate_logits`. 학생 분포 `q = softmax(candidate_logits)`에서 **multinomial 샘플링**으로 각 위치마다 학생 토큰 1개씩 뽑는다 (`q_selected_tokens`).
2. 교사가 후보 시퀀스를 한 번 forward → `new_logits`. 교사 분포 `p = softmax(new_logits)`.
3. **수락 기준(top-K)**: 교사 분포의 상위 K 토큰 집합 `p_selected_tokens`를 구하고, 학생이 샘플링한 토큰이 그 집합에 포함되면 수락(`is_accepted`). (top-p 옵션도 지원 — Appendix C의 adaptive K 실험용이 아닌 top-K/top-P 트렁케이션 선택.)
   - 표준 speculative decoding은 수락 확률 `min(1, p/q)` 비율 기반이라 교사 분포에서 샘플링하게 되어 supervised KD로 퇴화. SKD는 **"학생 토큰이 교사 top-K 안에 있는가"** 로 판정하여 학생 분포를 유지한다.
4. `n_matches` = 첫 거부 이전까지 연속 수락된 토큰 수 (Algorithm 1의 `n`).
5. **거부 시 교체**: 거부 위치 `n_matches`에서 교사 분포 `p_prime = p[:, n_matches, :]`로부터 **multinomial 재샘플링** `t`. 최종 `valid_tokens` = (수락된 학생 토큰들) + (교사 재샘플 토큰 t).

반환 `plus_enable`/`trigger_enable`은 토큰 교체 발생 여부로, 상위 `_assisted_decoding` 루프에서 `cor_count`/`tot_count`에 누적 → 최종 `correction_rate`(거부율) 계산에 쓰인다.

## 코드

```python
# speculative_kd/transformers/utils.py:4033
def _speculative_kd_sampling(
    candidate_input_ids,
    candidate_logits,
    candidate_length,
    new_logits,
    is_done_candidate,
    teacher_k,
    teacher_p,
):
    """Applies sampling as in the speculative decoding paper (algorithm 1)."""
    new_candidate_input_ids = candidate_input_ids[:, -candidate_length:]
    # q_i / p_i: assistant(학생) / model(교사) probabilities
    q = candidate_logits.softmax(dim=-1)
    # ancestral sampling으로 학생 토큰 1개씩 샘플링
    q_selected_tokens = torch.multinomial(q[0, :, :], num_samples=1)

    p = new_logits.softmax(dim=-1)
    # 학생 토큰이 교사의 top-k 안에 있는지 검사
    if teacher_k > 0:
        # last token at teacher should be removed
        p_selected_tokens_tuple = torch.topk(p[0, :-1, :], teacher_k)
        p_selected_tokens = p_selected_tokens_tuple.indices
    elif teacher_p > 0:
        sorted_logits, p_selected_tokens = torch.sort(p[0, :-1, :], descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_logits, dim=-1)
        sorted_indices_to_remove = cumulative_probs >= teacher_p
        sorted_indices_to_remove[:, 0] = False
        p_selected_tokens[sorted_indices_to_remove] = -1
    else:
        print("We do not support other truncation for teacher other than top-k or top-p!")
        exit(1)

    is_accepted = torch.sum(
        p_selected_tokens == q_selected_tokens.repeat(1, p_selected_tokens.size(1)), dim=1
    ) > 0
    n_matches = ((~is_accepted).cumsum(dim=-1) < 1).sum()  # algorithm 1의 `n`

    if is_done_candidate and n_matches == candidate_length:
        n_matches -= 1
        valid_tokens = new_candidate_input_ids[:, : n_matches + 1]
        return valid_tokens, n_matches, True, False
    else:
        gamma = candidate_logits.shape[1]
        # 학생 토큰이 교사 분포에서 생성될 가능성이 낮으면 교사에서 재샘플링
        p_prime = p[:, n_matches, :]
        t = torch.multinomial(p_prime, num_samples=1).squeeze(1)[None, :]

        # 선택된 토큰 = (매칭된 학생 토큰들) + (다음 샘플링 토큰)
        if n_matches > 0 and n_matches == gamma:
            valid_tokens = new_candidate_input_ids[:, :n_matches]
            return valid_tokens, n_matches, False, False
        if n_matches > 0 and n_matches < gamma:
            valid_tokens = torch.cat((new_candidate_input_ids[:, :n_matches], t), dim=-1)
            return valid_tokens, n_matches, True, True
        else:
            valid_tokens = t
            return valid_tokens, n_matches, True, True
```

## 부가: adaptive K decay (`_assisted_decoding` 내, Appendix C 구현)

`expected_seq_len > 0`일 때 교사 K를 선형 감소시켜 adaptive K 실험(Appendix C)을 재현. 단, 논문은 constant K=25가 adaptive보다 우수함을 보고.

```python
# speculative_kd/transformers/utils.py:3738 (within _assisted_decoding)
if expected_seq_len > 0:
    num_token_decay = max(int(expected_seq_len / teacher_k), 1)
    num_token_per_decay = max(math.ceil(teacher_k / expected_seq_len), 1)
    cur_token_index = 0
...
# 루프 내 (line 3845)
if expected_seq_len > 0:
    if cur_token_index % num_token_decay == 0:
        teacher_k -= num_token_per_decay
        teacher_k = max(teacher_k, 1)
    cur_token_index += 1
```
