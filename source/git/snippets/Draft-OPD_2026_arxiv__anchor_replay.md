# Error-Position Anchor Replay

> 출처: [분석 문서](../../../report/[paper][git]_Draft-OPD_On-Policy_Distillation_for_Speculative_Draft_Models_2026_arxiv.md) / submodule 경로: `source/git/Draft-OPD_bingyang-lei/verl/verl/models/transformers/dflash_student.py`

## 설명

Draft-OPD의 "error-position replay"를 구현하는 anchor plan 구축 로직._rollout 단계에서 기록된 reject token 위치(`reject_token_indices`)를 anchor로 변환해, 각 anchor로부터 drafting을 재현(replay)할 수 있게 만든다.

동작 흐름:
1. **Reject → anchor 변환**: response 내 거부 위치(rejects)를 정렬·중복제거. anchor는 `-1`(prompt 끝 = 시퀀스 시작) + 각 reject 위치. 각 anchor–boundary 쌍이 하나의 draft segment를 정의.
2. **Segment 길이**: `boundary - anchor`, `draft_block_size - 1`로 상한. segment는 draft가 제안할 block 길이.
3. **Anchor → 전체 인덱스**: `full_anchor = prompt_len - 1` (anchor<0 시) 또는 `prompt_len + anchor_resp`.
4. **Random anchor ablation**: `random_response_anchor_enabled=True`면 동일 토큰 수를 보존하면서 anchor를 무작위로 재분배 (논문 Table 3 "Random Anchors" 비교용).
5. **Rejected draft anchor 추가**: rollout에서 수집한 `rejected_draft_anchor_indices`/`offsets`를 기존 anchor에 dedup해서 추가 → 거부된 draft block의 시작 위치도 replay에 포함.

`RANDOM_RESPONSE_ANCHOR_SEED`(기본 42)가 ablation 재현성을 보장. 코드의 `verl_dflash_response_anchor_stride`는 anchor 밀도 제어(1=전부 계산).

## 코드

```python
# verl/verl/models/transformers/dflash_student.py — _build_opd_anchor_plan
def _build_opd_anchor_plan(self, *, input_ids, attention_mask, prompt_lengths,
    response_lengths, reject_token_indices, draft_block_size,
    response_anchor_stride=1, random_response_anchor_enabled=False,
    random_response_anchor_seed=42, rejected_draft_anchor_indices=None,
    rejected_draft_offsets=None, rejected_draft_mask=None,
    include_rejected_draft_anchors=True, ...):

    for batch_idx in range(batch_size):
        prompt_len = int(prompt_lengths[batch_idx].item())
        response_len = int(response_lengths[batch_idx].item())
        if response_len > 0:
            raw_rejects = reject_token_indices[batch_idx]
            # 정렬 + 중복제거 + 유효 범위 필터
            rejects = sorted({int(idx.item()) for idx in raw_rejects
                              if 0 <= int(idx.item()) < response_len})
            if len(rejects) == 0:
                anchors_resp, boundaries_resp = [], []
            elif rejects[-1] < response_len - 1:
                rejects.append(response_len - 1)        # 마지막 분할 보장
                anchors_resp = [-1] + rejects          # -1 = prompt 끝(시작)
                boundaries_resp = rejects
            else:
                anchors_resp = [-1] + rejects
                boundaries_resp = rejects

        # (optional) stride로 anchor 밀도 감소
        if response_anchor_stride > 1 and anchors_resp:
            anchor_boundary_pairs = [...]
            anchors_resp = [p[0] for p in anchor_boundary_pairs]
            boundaries_resp = [p[1] for p in anchor_boundary_pairs]

        # anchor/boundary → segment
        for anchor_resp, boundary_resp in zip(anchors_resp, boundaries_resp):
            full_anchor = prompt_len - 1 if anchor_resp < 0 else prompt_len + anchor_resp
            segment_len = min(boundary_resp - anchor_resp, draft_block_size - 1)
            ...

        # ablation: random anchor 재분배 (토큰 수 보존)
        if random_response_anchor_enabled and response_len > 0 and response_segment_lens:
            response_anchors_resp, response_segment_lens = self._build_random_response_anchor_plan(
                ..., seed=random_response_anchor_seed)

        # rejected-draft anchor dedup 추가 (거부 block 시작점도 replay에 포함)
        if include_rejected_draft_anchors and rejected_draft_anchor_indices is not None:
            for item_idx in range(rejected_count):
                if not bool(rejected_draft_mask[batch_idx, item_idx].item()):
                    continue
                offset = int(rejected_draft_offsets[batch_idx, item_idx].item())
                if offset <= 0 or offset >= draft_block_size:
                    continue
                anchor_resp = int(rejected_draft_anchor_indices[batch_idx, item_idx].item())
                full_anchor = prompt_len - 1 if anchor_resp < 0 else prompt_len + anchor_resp
                if full_anchor in existing_anchors:
                    continue
                sample_anchors.append(full_anchor)
                existing_anchors.add(full_anchor)

    # 패딩 → anchor_positions, segment_lens, block_keep_mask 텐서 반환
    return anchor_positions, segment_lens, row_starts, block_keep_mask, ..., stats
```
