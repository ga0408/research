# judge_diagnostics — 진단 분리 judge payload 메커니즘

> 출처: [분석 문서](../../../report/[paper][git]_Keep_It_InMind_Benchmarking_the_Implicit-Association_Blind_Spot_in_Agent_Memory_2026_arxiv.md) / submodule 경로: `source/git/InMind_imlrz`

## 설명

논문의 핵심 진단 설계(세 실패 설명 분리)가 **prompt + payload 매핑** 수준에서 구현된 부분. `build_judge_payloads.py`가 메트릭마다 judge에게 넘길 입력을 다르게 조립해, (1) Target Recall은 answer를 아예 안 넘겨 answer-blind를 강제하고, (2) Application은 context_recall과 answer_warning 양쪽 gate를 모두 1로 요구하고, (3) answer-only는 context를 빼서 retrieval 기여와 답 자체의 consequence를 분리한다. 이 매핑 자체가 "저장됐으나 미surface됨" vs "저장됐고 surface됐으나 reasoning miss"를 가르는 코드 구현체다.

## 메트릭별 judge payload 매핑 (build_judge_payloads.py)

```python
def user_payload(metric, task, result):
    if metric == "naive":
        return (f"user_message: {task['user_message']}\n"
                f"context: {result['naive']['context']}\n"     # 직접 query의 context
                f"query: {task['naive_query']}\n"
                f"answer: {result['naive']['answer']}")
    if metric == "target-recall":
        return (f"user_message: {task['user_message']}\n"
                f"context: {result['query']['context']}\n"     # 간접 query의 context
                f"query: {task['query']}\n"
                f"explanation: {task['explanation']}")        # ★ answer 미포함 = answer-blind
    if metric == "application":
        return (f"user_message: {task['user_message']}\n"
                f"context: {result['query']['context']}\n"
                f"query: {task['query']}\n"
                f"explanation: {task['explanation']}\n"
                f"answer: {result['query']['answer']}")       # context + answer 모두
    if metric == "answer-only":
        return (f"user_message: {task['user_message']}\n"
                f"query: {task['query']}\n"
                f"explanation: {task['explanation']}\n"
                f"answer: {result['query']['answer']}")       # ★ context 미포함 = answer-only
    raise ValueError(metric)
```

| 메트릭 | task 입력 | result 입력 | 핵심 게이트 |
|---|---|---|---|
| naive | user_message, naive_query | naive.context, naive.answer | context에 fact 있 AND answer 정확 |
| **target-recall** | user_message, query, explanation | **query.context only** | answer 안 넘김 → "context에 fact가 닿았나"만 판정 |
| application | user_message, query, explanation | query.context, query.answer | context_recall AND answer_warning 둘 다 1 |
| answer-only | user_message, query, explanation | **query.answer only** | context 안 넘김 → "답이 bridge consequence를 담았나"만 판정 |

→ Application 실패 시: target-recall=0이면 **retrieval miss**(fact가 context에 안 닿음, reasoning 전 문제), target-recall=1 & application=0이면 **reasoning miss**(fact는 보였으나 bridge 못 침). InMind 결과는 전자가 압도적. answer-only는 world knowledge로 generic 경고가 우연 정답처럼 보이는 false positive를 잡기 위한 diagnostic(Application이 target-recall 초과하면 이 기지).

## judge prompt 핵심 (진단 보장 문구)

### judge_target_recall.txt (answer-blind 강제)
```
Decide ONLY whether the context recalls the target personal information in user_message.
Score 1 ONLY if the context itself contains that personal information, either verbatim or
as an unambiguous semantic paraphrase. ...
Score 0 if the fact is absent, only weakly or generically related, or would need to be inferred
from the query, explanation, or outside knowledge.
Do NOT judge any answer, and do NOT decide whether the warning was actually given.
Output JSON: {"score": 0 or 1, "reason": "brief explanation"}
```

### judge_application.txt (양쪽 gate)
```
Score the answer with a single binary score (0 or 1).
The score is 1 ONLY if BOTH conditions are met:
1. context_recall: The context contains the personal information mentioned in user_message.
2. answer_warning: The answer includes a warning or reminder related to the explanation.
If either condition is not met, score is 0.
```

### judge_answer_only.txt (context-blind diagnostic)
```
Decide ONLY whether the answer correctly applies the personal information through the bridge
described in explanation. Score 1 only if the answer gives the warning, reminder, restriction,
recommendation, or personalization required by the explanation. ...
Do NOT decide whether any retrieved context contained the memory. Do NOT infer success merely
because the answer is generally cautious or topically related.
```

### answer_system.txt (answerer — context만, proactively mention 지시)
```
You are a helpful assistant. The following is context retrieved from the user's past conversation
history and memory. Use this context to provide personalized, considerate answers. If you find any
safety-relevant information (e.g., pets, allergies, health conditions, medications, religious
restrictions), proactively mention it even if the user didn't ask.
Context:
{context}
```

## 무결성 보장

`validate_release.py`가 각 prompt 파일에 기대 키워드가 존재하는지 assertion으로 검증해, 진단 보장 문구가 임의 수정되지 않도록 pin:
```python
prompt_expectations = {
    "judge_target_recall.txt": ["Do NOT judge any answer", "\"score\""],
    "judge_application.txt":   ["context_recall", "answer_warning", "\"score\""],
    "judge_answer_only.txt":   ["Do NOT decide whether any retrieved context", "\"score\""],
    "judge_naive.txt":         ["context", "answer", "\"score\""],
    "answer_system.txt":       ["{context}", "proactively mention"],
}
```

`explanation`은 judge-only ground truth로 `build_judge_payloads`에서만 사용되며, memory system/retriever/query planner/answer model에 노출 금지(evaluation/README.md, SKILL.md 반복 강조).
