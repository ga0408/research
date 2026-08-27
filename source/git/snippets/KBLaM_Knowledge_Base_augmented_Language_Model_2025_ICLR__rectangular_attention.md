# KBLaM Rectangular Attention & Dynamic Sparsification

> 출처: [분석 문서](../../../report/[paper][git]_KBLaM_Knowledge_Base_augmented_Language_Model_2025_ICLR.md) / submodule: `source/git/KBLaM_microsoft/src/kblam/models/llama3_model.py`

## 설명

KBLaM의 직사각형 어텐션(Rectangular Attention) 및 동적 희소화(Dynamic Sparsification) 핵심 로직:
1. **직사각형 어텐션 구조**: 프롬프트의 N개 토큰 쿼리가 이전 프롬프트 토큰들과 더불어 KB의 M개 지식 토큰(Knowledge Tokens) 키-값 쌍을 어텐션할 수 있도록 키(`key_states`)와 값(`value_states`) 차원의 앞부분에 `[kb_keys, key_states]` 형태로 지식 토큰을 접합(Concatenation).
2. **독립 쿼리 헤드 (`q_proj_new` / `sep_query_head`)**: 지식 베이스 검색용 쿼리(`query_states_2`)와 프롬프트 내부 인과 어텐션용 쿼리(`query_states`)를 분리하여 학습 안정성과 검색 정확도를 향상.
3. **어텐션 스코어 스케일링 (`kb_scale_factor`)**: 지식 베이스 트리플 수(M) 증가에 따른 소프트맥스 쏠림(Prompt 토큰 정보 소실)을 방지하기 위해 `attn_weights_2 - log(kb_len) + log(C)` 보정 적용.
4. **동적 희소화 (`prune_key_value`)**: 추론 시 방대한 KB(10K+ 트리플) 중 쿼리와의 내적 점수 상위 K개 토큰만 선택하여 메모리 및 연산량을 O(K)로 제한.

## 코드 스니펫

```python
# src/kblam/models/llama3_model.py

class KblamLlamaAttention(nn.Module):
    """Multi-headed attention implemented as Rectangular attention for KBLaM"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        # 기본 LLM 프로젝션 헤드
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        # KB 전용 독립 쿼리 프로젝션 헤드 (W_tilde_Q)
        self.q_proj_new = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self._init_rope()

    def prune_key_value(self, query, kb_keys, kb_values, topk_size=20):
        """추론 시 Top-K 지식 토큰만 동적으로 선별하는 Pruning 함수"""
        assert query.requires_grad is False, "This function should only be used at test time"
        batch_size, num_heads, kb_len, head_dim = kb_keys.shape
        attn_weights = torch.matmul(query, kb_keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if topk_size >= kb_len:
            return kb_keys, kb_values, attn_weights
        with torch.autograd.no_grad():
            top_idx = attn_weights.sum((1, 2)).topk(min(kb_len, topk_size), -1)[1]
            top_idx = top_idx.view(batch_size, -1, topk_size, 1).expand(
                batch_size, num_heads, topk_size, head_dim
            )
            kb_keys = kb_keys.gather(-2, top_idx)
            kb_values = kb_values.gather(-2, top_idx)
        return kb_keys, kb_values, attn_weights[..., :topk_size]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        kb_kvs: Optional[tuple] = None,
        kb_config: Optional[KBLaMConfig] = None,
        save_attention_weights: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # 1. 프롬프트 쿼리 및 KB 전용 쿼리 계산
        query_states = self.q_proj(hidden_states)
        query_states_2 = self.q_proj_new(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        query_states_2 = query_states_2.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        kb_layer_frequency = kb_config.kb_layer_frequency
        dynamic_sparsify = kb_config.dynamic_sparsify
        topk_size = kb_config.top_k_kb
        attn_weights_2 = None

        # 2. KB Knowledge Tokens 주입 (지정된 레이어 주기마다)
        if kb_kvs is not None:
            if self.layer_idx % kb_layer_frequency == 0:
                kb_keys, kb_values = kb_kvs
                kb_idx = self.layer_idx // kb_layer_frequency
                if len(kb_keys.shape) == 2:
                    kb_len = kb_keys.shape[0]
                    kb_keys = kb_keys.reshape(kb_len, 1 + self.config.num_hidden_layers // kb_layer_frequency, -1)[:, kb_idx]
                    kb_values = kb_values.reshape(kb_len, 1 + self.config.num_hidden_layers // kb_layer_frequency, -1)[:, kb_idx]
                    kb_keys = kb_keys.view(kb_len, self.num_heads, self.head_dim).transpose(0, 1).unsqueeze(0).expand(bsz, self.num_heads, kb_len, self.head_dim)
                    kb_values = kb_values.view(kb_len, self.num_heads, self.head_dim).transpose(0, 1).unsqueeze(0).expand(bsz, self.num_heads, kb_len, self.head_dim)
                    
                    if dynamic_sparsify:
                        kb_keys, kb_values, attn_weights_2 = self.prune_key_value(query_states_2, kb_keys, kb_values, topk_size)
                    
                    # 직사각형 어텐션: 프롬프트 키/값 앞에 KB 키/값을 연결
                    key_states = torch.concat([kb_keys, key_states], dim=2)
                    value_states = torch.concat([kb_values, value_states], dim=2)

                kb_len = kb_keys.shape[2]
                kb_atten_mask = attention_mask.new_zeros(bsz, 1, q_len, kb_len)
                attention_mask = torch.concat([kb_atten_mask, attention_mask], dim=-1)

        # 3. 어텐션 가중치 계산 및 스케일링
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        sep_query_head = kb_config.sep_query_head
        kb_scale_factor = kb_config.kb_scale_factor
        if sep_query_head and kb_kvs is not None and (self.layer_idx % kb_layer_frequency == 0):
            if attn_weights_2 is None:
                attn_weights_2 = torch.matmul(query_states_2, kb_keys.transpose(2, 3)) / math.sqrt(self.head_dim)
            attn_weights = attn_weights[:, :, :, kb_len:]
            if kb_scale_factor is not None:
                # log(C) - log(M) 어텐션 스코어 스케일링
                attn_weights_2 = attn_weights_2 - np.log(kb_len) + np.log(kb_scale_factor)
            attn_weights = torch.concat([attn_weights_2, attn_weights], -1)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value
```
