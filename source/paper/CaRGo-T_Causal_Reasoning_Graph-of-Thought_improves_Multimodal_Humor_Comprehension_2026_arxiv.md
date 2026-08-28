# CaRGo-T: Causal Reasoning Graph-of-Thought improves Multimodal Humor Comprehension — 핵심 발췌

> 출처: [분석 문서](../../report/[paper][git]_CaRGo-T_Causal_Reasoning_Graph-of-Thought_improves_Multimodal_Humor_Comprehension_2026_arxiv.md) / 원본: [arXiv:2608.23172](https://arxiv.org/abs/2608.23172) · [GitHub Code](https://github.com/abhi1nandy2/CaRGo-T)

---

## 1. 문제 정의 및 수식화 (Problem Formulation)

```
입력: 이미지 I, 과업별 텍스트 프롬프트 P
출력: 예측 답변 Y_hat ⊆ F_θ(I, P)

VLM 출력 분해:
F_θ(I, P) = [R, Y_hat]
- R: 중간 추론 성분 (Reasoning Component)
- Y_hat: 최종 정답 (Final Answer)
- F_θ: 파라미터 θ를 갖는 사전학습된 Vision-Language Model
```

- 유머 이해 (Humor Understanding): 정답 `Y`는 자연어 펀치라인/설명 텍스트이며, `Y_hat`과 `Y`의 어휘적(ROUGE-L, BLEU) 및 의미론적(BERTScore) 유사도를 극대화.
- 유머 탐지 (Humor Detection): 정답 `Y ∈ {"Yes", "No"}` (또는 `{"Y", "N"}`)의 이진 분류 문제로, 분류 정확도(Accuracy) 및 Macro-F1을 평가.

---

## 2. 인과 추론 그래프 (Causal Reasoning Graph, CRG) 구조 정의

인과 추론 그래프는 코드 형태(JSON 스키마)로 직렬화되는 경량 그래프 추론 구조로, 개체(Entities)와 그 속성(Properties), 그리고 사건 간의 인과 관계(Causal Relationships)로 엄격히 분리 정의된다.

```json
{
  "entities": {
    "ENTITY_ID": {
      "properties": [
        "개체 서술어/형용사/부사적 수식어",
        "타 개체와의 비인과적 양방향 관계 (어느 한 개체에만 기재)"
      ]
    }
  },
  "causal_relationships": [
    {
      "cause": "원인 사건 (EVENT_1: 개체와 속성의 결합 형태 'X가 Z와 Y를 행함')",
      "effect": "결과 사건 (EVENT_2: 상태 변화 또는 감정/인지적 결과)"
    }
  ]
}
```

### In-Context 수동 정제 (Manual Rectification) 규칙
1. `effects` 속성을 `entities` 내부에서 제거하고 순수한 `properties`로 한정.
2. 환각되거나 불필요한 개체 노드를 축약하고 핵심 원인-결과 관계를 `causal_relationships` 리스트에 최소 단위 쌍으로 정렬.

---

## 3. 핵심 정량 실험 결과 발췌 (Benchmark Results)

### Zero-Shot 유머 이해 성능 비교 (Table 1)

| VLM | Method | SIU ROUGE-L | SIU BLEU | SIU BERTScore | SIU Avg. Score | MC ROUGE-L | MC BLEU | MC BERTScore | MC Avg. Score |
|---|---|---|---|---|---|---|---|---|---|
| MiniCPM (0-shot) | Vanilla | 0.1669 | 0.0108 | 0.8589 | 0.3455 | 0.0789 | 0.0073 | 0.8330 | 0.3064 |
| | CoT | 0.1630 | 0.0155 | 0.8586 | 0.3457 | 0.0646 | 0.0033 | 0.8303 | 0.2994 |
| | CoD | 0.1684 | 0.0137 | 0.8616 | 0.3479 | 0.0888 | 0.0059 | 0.8394 | 0.3114 |
| | CCoT | 0.1482 | 0.0086 | 0.8541 | 0.3370 | 0.0744 | 0.0040 | 0.8404 | 0.3063 |
| | **CaRGo-T** | **0.1779** | 0.0139 | 0.8594 | **0.3504** | **0.1260** | **0.0123** | **0.8503** | **0.3295** |
| GPT-4o-mini (0-shot) | Vanilla | 0.1493 | 0.0098 | 0.8525 | 0.3372 | 0.0889 | 0.0045 | 0.8450 | 0.3128 |
| | CoT | 0.0880 | 0.0052 | 0.8197 | 0.3043 | 0.1178 | 0.0076 | 0.8453 | 0.3236 |
| | CoD | 0.1388 | 0.0107 | 0.8340 | 0.3278 | 0.1282 | 0.0078 | 0.8459 | 0.3273 |
| | CCoT | 0.1184 | 0.0062 | 0.8402 | 0.3216 | 0.1315 | 0.0069 | 0.8566 | 0.3317 |
| | **CaRGo-T** | **0.2024** | **0.0185** | **0.8687** | **0.3632** | **0.1377** | **0.0080** | 0.8497 | **0.3318** |
| GPT-4o (0-shot) | Vanilla | 0.1893 | 0.0152 | 0.8667 | 0.3571 | 0.1092 | 0.0060 | 0.8512 | 0.3221 |
| | CoT | 0.1459 | 0.0082 | 0.8470 | 0.3337 | 0.1183 | 0.0064 | 0.8523 | 0.3257 |
| | CoD | 0.2064 | 0.0200 | 0.8725 | 0.3663 | 0.1264 | 0.0069 | 0.8535 | 0.3289 |
| | CCoT | 0.1605 | 0.0094 | 0.8564 | 0.3421 | 0.1055 | 0.0057 | 0.8535 | 0.3216 |
| | **CaRGo-T** | **0.2219** | **0.0245** | 0.8715 | **0.3726** | **0.1316** | **0.0078** | **0.8569** | **0.3321** |

*(SIU: Satirical Image Understanding on YesBut, MC: Meme Captioning on MemeCap)*

---

### Sarcasm & Satire Detection 성능 비교 (Tables 3 & 4, GPT-4o)

```
1. MMSD 2.0 Multimodal Sarcasm Detection (GPT-4o)
- 0-shot Accuracy: Vanilla 47.42% | CoT 48.07% | CaRGo-T 49.48% (+2.93% relative)
- 0-shot F1-Score: Vanilla 61.05% | CoT 61.69% | CaRGo-T 62.20% (+0.83% relative)
- 2-shot Accuracy: Vanilla 47.81% | CoT 48.32% | CaRGo-T 49.88% (+3.23% relative)
- 6-shot Accuracy: Vanilla 47.85% | CoT 48.61% | CaRGo-T 49.91% (+2.67% relative)

2. YesBut Satire Detection (GPT-4o)
- 0-shot Accuracy: Vanilla 42.60% | CoT 42.70% | CaRGo-T 43.18% (+1.12% relative)
- 2-shot Accuracy: Vanilla 44.05% | CoT 44.39% | CaRGo-T 44.63% (+0.54% relative)
- 6-shot Accuracy: Vanilla 44.91% | CoT 45.38% | CaRGo-T 45.57% (+1.05% relative)
```

---

## 4. 추론 성분 정보이론 및 LLM-as-a-Judge 검증 (Dissection)

### 토큰 분포 KL Divergence (Table 5) & Low Similarity Fraction (Table 6)
- `KL(R_CaRGo-T || R_CoT) = 0.21` > `KL(R_CoT || R_CaRGo-T) = 0.19`
- `KL(R_CaRGo-T || R_CoD) = 0.21` > `KL(R_CoD || R_CaRGo-T) = 0.20`
- `KL(R_CaRGo-T || R_CCoT) = 0.25` = `KL(R_CCoT || R_CaRGo-T) = 0.25`
- `LSF(R_CaRGo-T || R_CoT) = 1.00` > `LSF(R_CoT || R_CaRGo-T) = 0.85`
- `LSF(R_CaRGo-T || R_CoD) = 1.00` > `LSF(R_CoD || R_CaRGo-T) = 0.84`
- `LSF(R_CaRGo-T || R_CCoT) = 0.98` > `LSF(R_CCoT || R_CaRGo-T) = 0.94`

### INFERSCORE (Table 7: GPT-4 Judge)
- CoT: 40.78%
- CoD: 40.68%
- CCoT: 37.64%
- **CaRGo-T: 45.11%** (베이스라인 대비 +4.33%p ~ +7.47%p 향상)

---

## 5. 소거 연구 결과 (Ablation Analysis, Table 8)

| Shots | Variant | ROUGE-L | BLEU | BERTScore | Avg. Score |
|---|---|---|---|---|---|
| 0-shot | WITH DEFN. (가이드라인 텍스트 포함) | 0.2131 | 0.0211 | 0.8718 | 0.3687 |
| | **CaRGo-T** | **0.2219** | **0.0245** | 0.8715 | **0.3726** |
| 2-shot | WITH DEFN. | 0.2406 | 0.0309 | 0.8816 | 0.3844 |
| | UNRECTIFIED (미정제 날 것의 GPT-4o CRG) | 0.2492 | 0.0327 | 0.8495 | 0.3771 |
| | **CaRGo-T** | **0.2534** | **0.0339** | **0.8860** | **0.3911** |
| 5-shot | WITH DEFN. | 0.2478 | 0.0308 | 0.8844 | 0.3877 |
| | UNRECTIFIED | 0.2466 | 0.0310 | 0.8497 | 0.3758 |
| | **CaRGo-T** | **0.2513** | **0.0318** | **0.8872** | **0.3901** |
