# LMLM Pre-training Loss Masking Logic

> 출처: [분석 문서](../../../report/[paper][git]_Pre-training_Limited_Memory_Language_Models_with_Internal_and_External_Knowledge_2025_Cornell.md) / submodule: `source/git/LMLM_kilian-group/src/lmlm/training/utils/utils_mask.py` 및 `utils_metrics.py`

## 설명

LMLM의 핵심 사전학습 메커니즘은 외부 데이터베이스로부터 조회되는 반환값(return value) 토큰 및 `<|db_end|>` 토큰을 손실 함수(cross entropy loss) 계산에서 제외(masking)하는 것.
1. `extract_dblookup_masks`: 배치 텐서에서 특수 토큰 경계(`<|db_entity|>`, `<|db_relationship|>`, `<|db_return|>`, `<|db_end|>`)를 벡터화된 스팬 매칭으로 식별.
2. `compute_pretrain_mask`: `pretrain_mask = ~(value_mask | end_token_mask) & (shift_labels != -100)`을 계산하여 반환값과 end 토큰의 gradient 역전파를 차단.
3. `compute_loss_func`: 마스킹된 위치를 제외한 토큰들에 대해서만 weighted cross-entropy 손실을 계산. 이를 통해 모델은 사실 자체를 가중치에 암기하지 않고, 사실이 필요할 때 룩업 쿼리를 생성하는 능력만 학습.

## 코드 스니펫

```python
# src/lmlm/training/utils/utils_mask.py

def extract_dblookup_masks(
    tokens: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    pretrain_mask_only: bool = False,
    include_eos: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    특수 토큰을 기준으로 entity, relationship, value, pretrain 마스크를 생성.
    """
    special_ids = {
        "entity": tokenizer.convert_tokens_to_ids(DB_START_TOKEN),
        "rel": tokenizer.convert_tokens_to_ids(DB_SEP_TOKEN),
        "return": tokenizer.convert_tokens_to_ids(DB_RETRIEVE_TOKEN),
        "end": tokenizer.convert_tokens_to_ids(DB_END_TOKEN),
        "eos": tokenizer.eos_token_id,
        "bos": tokenizer.bos_token_id,
        "pad": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    }

    B, T = tokens.shape
    device = tokens.device

    if pretrain_mask_only:  
        pad_mask = (tokens == special_ids["pad"]).to(device)
        # return 토큰부터 end 토큰까지의 스팬을 value_mask로 지정
        value_mask = get_span_mask(tokens, special_ids["return"], special_ids["end"], special_ids["eos"], bos_token_id=special_ids["bos"])
        end_token_mask = (tokens == special_ids["end"]).to(device)

        # 사전학습 마스크: 반환값 스팬과 end 토큰을 제외
        pretrain_mask = ~(value_mask | end_token_mask)
        pretrain_mask[pad_mask] = 0

        if include_eos:
            pretrain_mask[end_token_mask] = 1

        return {"pretrain": pretrain_mask}

    # 전체 마스크 카테고리 구성
    entity_mask = get_span_mask(tokens, special_ids["entity"], special_ids["rel"], special_ids["eos"], bos_token_id=special_ids["bos"])
    rel_mask    = get_span_mask(tokens, special_ids["rel"], special_ids["return"], special_ids["eos"], bos_token_id=special_ids["bos"])
    value_mask  = get_span_mask(tokens, special_ids["return"], special_ids["end"], special_ids["eos"], bos_token_id=special_ids["bos"])
    db_span     = get_span_mask(tokens, special_ids["entity"], special_ids["end"], special_ids["eos"], bos_token_id=special_ids["bos"])

    pad_mask = tokens == special_ids["pad"]
    org_mask = ~db_span
    org_mask[pad_mask] = 0

    end_token_mask = (tokens == special_ids["end"])
    pretrain_mask = ~(value_mask | end_token_mask)
    pretrain_mask[pad_mask] = 0

    return {
        "entity": entity_mask,
        "relationship": rel_mask,
        "value": value_mask,
        "dblookup": db_span,
        "org": org_mask,
        "pretrain": pretrain_mask
    }

# src/lmlm/training/utils/utils_metrics.py

def compute_loss_func(outputs, labels, num_items_in_batch, include_eos=False):
    """
    Return value 토큰이 제외된 사전학습 손실 함수
    """
    logits = outputs.logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    pretrained_mask = compute_pretrain_mask(shift_labels, include_eos=include_eos)

    loss_fct = CrossEntropyLoss(reduction='none', ignore_index=-100)
    per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(labels.size(0), -1)
    if pretrained_mask.shape != per_token_loss.shape:
        pretrained_mask = pretrained_mask.view(per_token_loss.size(0), -1)

    weighted_loss = per_token_loss[pretrained_mask != 0]

    if num_items_in_batch is None:
        return weighted_loss.mean()
    else:
        return weighted_loss.sum() / num_items_in_batch
```
