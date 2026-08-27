# KBLaM Knowledge Base Encoder & Adapters

> 출처: [분석 문서](../../../report/[paper][git]_KBLaM_Knowledge_Base_augmented_Language_Model_2025_ICLR.md) / submodule: `source/git/KBLaM_microsoft/src/kblam/kb_encoder.py`

## 설명

KBLaM의 지식 베이스 인코더(`KBEncoder`)와 선형 어댑터(Linear Adapter) 구조:
1. **Key/Value 분리 인코딩**:
   - Key: `The <property> of <name>` 문자열을 사전학습 문장 임베더(`SentenceTransformer` 또는 OpenAI ada-002)로 인코딩한 후 선형 어댑터(`projector_k`)와 LayerNorm을 통과시켜 `k_tilde_m` 벡터 생성.
   - Value: `<value>` 문자열을 사전학습 문장 임베더로 인코딩한 후 선형 어댑터(`projector_v`)를 통과시켜 `v_tilde_m` 벡터 생성.
2. **동결된 백본(Frozen Backbone) & 경량 어댑터 학습**: 문장 인코더 백본은 동결하고, 소수의 선형 프로젝터 파라미터만 최적화하여 인코더 잠재 공간과 LLM의 어텐션 키/값 공간 사이의 정렬(Alignment) 학습.
3. **오프라인 사전 연산(Offline Caching)**: 고정된 대규모 KB의 기본 임베딩을 사전에 계산 및 저장하여 학습 및 추론 시 인코딩 오버헤드를 극소화.

## 코드 스니펫

```python
# src/kblam/kb_encoder.py

class KBEncoder(nn.Module, FeatureExtractionMixin):
    kb_special_token = {
        "<KB_BEGIN>": 0,
        "<KB_END>": 1,
        "<KEY_SEP>": 2,
        "<VALUE_SEP>": 3,
        "<ENTITY_SEP>": 4,
        "<KV_SEP>": 5,
    }

    def __init__(
        self,
        encoder_name: str,
        projector_type: str,
        out_dim: int,
        endpoint_url: str,
        projector_kwargs: dict = {},
        frozen_base_model: bool = True,
        device: Union[str, torch.device] = "cuda",
        get_oai_embd_online: bool = False,
    ):
        super().__init__()
        self.encoder_spec = encoder_name

        if encoder_name in ["OAI", "BigOAI"]:
            big = "Big" in encoder_name
            if get_oai_embd_online:
                if big:
                    self.gs = GPT("text-embedding-3-large", endpoint_url)
                else:
                    self.gs = GPT("ada-embeddings", endpoint_url)

                self.base_model_encode = lambda s: torch.tensor(
                    self.gs.generate_embedding(s)
                ).to(self.device)
            else:
                self.base_model_encode = None
            self.in_dim = 3072 if big else 1536
        else:
            self.base_model = SentenceTransformer(encoder_name)
            self.base_model_encode = lambda s: self.base_model.encode(
                s, convert_to_numpy=False
            )
            self.frozen_base_model = frozen_base_model
            if frozen_base_model:
                self.base_model.eval()
                for param in self.base_model.parameters():
                    param.requires_grad = False
            else:
                self.base_model.train()
            self.in_dim = self.base_model.get_sentence_embedding_dimension()

        self.out_dim = out_dim
        # Key 및 Value 전용 선형 어댑터 프로젝터
        self.projector_k = get_projector(projector_type, self.in_dim, self.out_dim, projector_kwargs)
        self.projector_v = get_projector(projector_type, self.in_dim, self.out_dim, projector_kwargs)
        self.key_layernorm = nn.LayerNorm(self.out_dim, elementwise_affine=False, bias=False)
        self.embedding = nn.Embedding(len(self.kb_special_token), out_dim)
        self.device = device
        self.to(self.device)

    def encode_key(self, S=None, base_emb=None):
        """Key 문자열 또는 사전 계산된 임베딩을 LLM Key 공간으로 투영"""
        if S:
            base_embedding = self.base_model_encode(S)
        elif base_emb is not None:
            base_embedding = torch.from_numpy(base_emb).to(self.device)
        return self.key_layernorm(self.projector_k(base_embedding)).bfloat16()

    def encode_val(self, S=None, base_emb=None):
        """Value 문자열 또는 사전 계산된 임베딩을 LLM Value 공간으로 투영"""
        if S:
            base_embedding = self.base_model_encode(S)
        elif base_emb is not None:
            base_embedding = torch.from_numpy(base_emb).to(self.device)
        return self.projector_v(base_embedding).bfloat16()

    def encode_key_value(self, key, value):
        key_embd = self.encode_key(S=key)
        value_embd = self.encode_val(S=value)
        return key_embd, value_embd

    def encode_key_value_embeddings(self, key_embd, value_embd):
        key_embd = self.encode_key(base_emb=key_embd)
        value_embd = self.encode_val(base_emb=value_embd)
        return key_embd, value_embd

    def encode_base_embeddings(self, kb: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        key_embds, value_embds = [], []
        for key, value in zip(kb[0], kb[1]):
            key_embd, value_embd = self.encode_key_value_embeddings(key, value)
            key_embds.append(key_embd)
            value_embds.append(value_embd)
        return torch.stack(key_embds), torch.stack(value_embds)
```
