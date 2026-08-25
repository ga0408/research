# LMLM Modeling & Inference with Dynamic Database Lookup

> 출처: [분석 문서](../../../report/[paper][git]_Pre-training_Limited_Memory_Language_Models_with_Internal_and_External_Knowledge_2025_Cornell.md) / submodule: `source/git/LMLM_kilian-group/src/lmlm/modeling_lmlm.py`

## 설명

LMLM의 핵심 추론 클래스인 `LlamaForLMLM` 및 `GPT2ForLMLM`의 동작 로직:
1. `generate_with_lookup`: 텍스트 생성 중 `<|db_return|>` 토큰이 생성되면 생성을 중단(stop token)하고, 직전에 생성된 `<|db_entity|> Entity <|db_relationship|> Relation <|db_return|>` 문자열을 파싱하여 `DatabaseManager`로 데이터베이스 검색을 수행.
2. 검색된 사실 값(return value)과 `<|db_end|>` 토큰을 프롬프트 컨텍스트에 추가한 후 후속 텍스트 생성을 이어감.
3. `set_logits_bias`: 특수 토큰들(`<|db_entity|>`, `<|db_relationship|>`, `<|db_return|>`, `<|db_end|>`)에 대해 `LogitBiasProcessor`를 등록하여 모델이 필요 시점에 안정적으로 룩업 문법을 방출하도록 유도.
4. `post_process`: 생성 완료 후 룩업 호출 구문을 자연스러운 일반 텍스트로 정제(정규화 및 미완성 태그 필터링).

## 코드 스니펫

```python
# src/lmlm/modeling_lmlm.py

class LlamaForLMLM(LlamaForCausalLM):
    def __init__(self, config, db_manager=None, use_special_tokens=True, threshold=0.6, fallback_policy="top1_anyway"):
        super().__init__(config)
        self.use_special_tokens = use_special_tokens
        self.fallback_policy = fallback_policy
        self.db_manager = db_manager
        if db_manager is not None:
            self.db_manager.init_topk_retriever(default_threshold=threshold)
        self.logits_processor = None

    def set_logits_bias(self, tokenizer):
        """특수 토큰 방출 확률을 높이기 위한 Logit Bias 설정"""
        if self.logits_processor is not None:
            return
        
        entity_token_id = tokenizer.convert_tokens_to_ids(DB_START_TOKEN)
        relationship_token_id = tokenizer.convert_tokens_to_ids(DB_SEP_TOKEN)
        return_token_id = tokenizer.convert_tokens_to_ids(DB_RETRIEVE_TOKEN)
        end_token_id = tokenizer.convert_tokens_to_ids(DB_END_TOKEN)

        bias = 2
        logit_bias = {
            entity_token_id: bias * 2,
            relationship_token_id: bias,
            return_token_id: bias,
            end_token_id: bias
        }
        self.logits_processor = [LogitBiasProcessor(logit_bias)]

    def generate_with_lookup(self, prompt, tokenizer, enable_dblookup, enable_postprocess=True, **kwargs):
        """
        데이터베이스 룩업을 동적으로 인터리빙하며 텍스트를 생성하는 루프
        """
        max_new_tokens = kwargs.pop("max_new_tokens", 256)
        max_lookup_limit = kwargs.pop("max_lookup_limit", 5)

        self.eval()
        device = self.device
        finished = False

        if not enable_dblookup:
            # 룩업 비활성화 시 일반 generation 수행 후 후처리
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = self.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], **kwargs)
            output_text = self.normalize_db_format(tokenizer.decode(outputs[0], skip_special_tokens=False))
            output_text = output_text.split(prompt)[-1]
            return self.post_process(output_text, tokenizer) if enable_postprocess else output_text

        self.set_logits_bias(tokenizer)

        input_text = prompt
        stop_token_ids = [
            tokenizer.convert_tokens_to_ids(DB_RETRIEVE_TOKEN),
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
        ]

        generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=stop_token_ids,
            return_dict_in_generate=False,
        )

        while not finished:
            #### Step 1: 입력 인코딩
            inputs = tokenizer(input_text, return_tensors="pt").to(device)
            input_len = inputs["input_ids"].shape[1]

            #### Step 2: DB_RETRIEVE_TOKEN 또는 EOS 만날 때까지 생성
            with torch.no_grad():
                outputs = self.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    logits_processor=self.logits_processor,
                    **generate_kwargs,
                    **kwargs
                )
            
            output_text = self._decode_with_special_tokens(outputs, tokenizer, input_len, input_text)
            input_text += output_text

            # <|db_return|>이 없으면 일반 종료
            if DB_RETRIEVE_TOKEN not in output_text:
                break

            #### Step 3: 외부 데이터베이스에서 사실 검색
            try:
                return_value = self.db_manager.retrieve_from_database(output_text)
            except DatabaseLookupError as e:
                logger.warning(f"Database lookup failed: {e}")
                return_value, _ = self.handle_dblookup_failure(output_text)

            #### Step 4: 검색된 사실값과 종료 토큰을 컨텍스트에 주입 후 이어서 생성
            input_text += return_value + DB_END_TOKEN

            if self.token_len_without_dblookups(input_text, tokenizer) >= max_new_tokens:
                finished = True
            if len(input_text.split(DB_START_TOKEN)) >= max_lookup_limit:
                finished = True
        
        output_text = input_text.split(prompt)[-1]
        if enable_postprocess:
            output_text = self.post_process(output_text, tokenizer)
        return output_text
```
