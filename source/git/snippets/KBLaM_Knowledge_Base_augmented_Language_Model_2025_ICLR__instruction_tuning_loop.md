# KBLaM Instruction Tuning & Synthetic Batch Construction

> 출처: [분석 문서](../../../report/[paper][git]_KBLaM_Knowledge_Base_augmented_Language_Model_2025_ICLR.md) / submodule: `source/git/KBLaM_microsoft/experiments/train.py`

## 설명

KBLaM의 합성 데이터셋 기반 인스트럭션 튜닝(Instruction Tuning) 및 배치 구성 로직:
1. **합성 KB 샘플링**: 120K 합성 트리플 풀에서 각 학습 샘플마다 10~100개의 무작위 트리플을 샘플링하여 샘플 전용 KB 구성 (정답 트리플 1~2개 + 나머지 distractor).
2. **4종 인스트럭션 유형 배합**:
   - 단일 엔티티 Q&A (Simple Q&A, 6 마이크로배치)
   - 다중 엔티티 Q&A (Multi-entities Q&A, 6 마이크로배치)
   - 개방형 추론 Q&A (Open-ended reasoning Q&A, 6 마이크로배치)
   - 답변 불가 Q&A (Unanswerable/Out-of-KB Q&A, 2 마이크로배치 → "Sorry, I cannot find relevant information in the KB" 출력 학습)
3. **가변 컨텍스트 스케줄러 (`context_set_size_scheduler`)**: 학습 초기에는 작은 KB 크기로 시작하여 점진적으로 KB 크기를 확장, 안정적인 수렴 유도.
4. **동결 백본 및 어댑터 최적화**: LLM 백본과 인코더는 동결하고 어댑터 파라미터 `theta`(`projector_k`, `projector_v`, `q_proj_new`)만 AdamW로 학습.

## 코드 스니펫

```python
# experiments/train.py & src/kblam/utils/train_utils.py

def train_kblam_step(model, kb_encoder, batch_data, optimizer, scheduler, accelerator):
    """
    KBLaM 1회 학습 스텝: KB 임베딩 주입 및 Q&A 정답 생성 CrossEntropy 손실 계산
    """
    model.train()
    kb_encoder.train()
    
    # 1. 배치 데이터 로드 (KB 트리플, 질문, 정답)
    input_ids = batch_data["input_ids"]          # Prompt 질문 및 템플릿 토큰
    labels = batch_data["labels"]                # 정답 A에 대해서만 Loss 계산 (질문 부분 -100 마스킹)
    kb_triples = batch_data["kb_triples"]        # 샘플별 샘플링된 KB (10~100 트리플)
    
    # 2. KB 인코더를 통한 Knowledge Tokens (Keys, Values) 생성
    # precomputed_embd가 있는 경우 캐시에서 고속 인출
    kb_keys, kb_values = kb_encoder.encode_base_embeddings(batch_data["kb_base_embds"])
    
    # 3. 모델 포워드 (Rectangular Attention 적용)
    outputs = model(
        input_ids=input_ids,
        labels=labels,
        kb_kvs=(kb_keys, kb_values),
        kb_config=model.kb_config
    )
    
    loss = outputs.loss
    accelerator.backward(loss)
    
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    
    return loss.item()

def context_set_size_scheduler(step, total_steps, min_size=10, max_size=100):
    """학습 진행에 따라 샘플당 KB 트리플 크기를 동적으로 확장하는 스케줄러"""
    progress = min(1.0, step / (total_steps * 0.5))
    current_size = int(min_size + progress * (max_size - min_size))
    return current_size
```
