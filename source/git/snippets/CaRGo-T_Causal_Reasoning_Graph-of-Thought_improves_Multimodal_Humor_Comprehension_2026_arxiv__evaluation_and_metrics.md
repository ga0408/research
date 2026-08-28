# CaRGo-T: Evaluation Metrics and Information-Theoretic Validation Snippet

> 원본 코드 및 알고리즘: [`source/git/CaRGo-T_abhi1nandy2/src/eval.py`](../source/git/CaRGo-T_abhi1nandy2/src/eval.py), 논문 Appendix F (Algorithms 1 & 2)

---

## 1. 다중 생성 평가 메트릭 산출 (`src/eval.py`)

유머 이해(Satirical Image Understanding, Meme Caption Generation) 과업의 생성 결과는 렉시컬 중첩도(ROUGE-L, BLEU)와 임베딩 기반 의미 유사도(BERTScore)를 종합하여 `Avg. Score`로 평가한다.

```python
import json
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score

def calculate_metrics(references, hypothesis):
    metrics = {}

    # 1. ROUGE-L F-measure (Stemmer 적용)
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = [rouge.score(ref, hypothesis)['rougeL'].fmeasure for ref in references]
    metrics['rouge'] = sum(rouge_scores) / len(rouge_scores)

    # 2. BLEU Score (Method 1 Smoothing 적용)
    tokenized_references = [ref.split() for ref in references]
    tokenized_hypothesis = hypothesis.split()
    metrics['bleu'] = sentence_bleu(
        tokenized_references, 
        tokenized_hypothesis, 
        smoothing_function=SmoothingFunction().method1
    )

    # 3. BERTScore (DeBERTa 기반 의미론적 유사도 F1)
    bert = bert_score([hypothesis] * len(references), references, lang="en")
    metrics['bert'] = float(bert[2].mean())

    # 4. Avg. Score (3대 지표의 산술 평균)
    metrics['avg'] = (metrics['rouge'] + metrics['bleu'] + metrics['bert']) / 3.0

    return metrics
```

---

## 2. 정보이론적 검증 알고리즘 1: 토큰 분포 기반 KL Divergence

추론 성분 `R_1`이 `R_2` 대비 얼마나 많은 새로운/예측 불가능한 렉시컬 정보를 담고 있는지 측정한다 (`α` 스무딩 적용).

```
Input: Texts T1, T2; smoothing parameter α > 0
Output: KL(T1 || T2)

Function PREPROCESSANDTOKENIZE(Text):
    return lowercase(Text).split_on_whitespace()

T1 = PREPROCESSANDTOKENIZE(T1)
T2 = PREPROCESSANDTOKENIZE(T2)

V = T1 ∪ T2
|V| = len(V)

For i in {1, 2}:
    c_i(w) = count(w in T_i) for w in V
    Z_i = |T_i| + α * |V|
    D_i(w) = (c_i(w) + α) / Z_i

P = D1, Q = D2
KL = ∑_{w ∈ V} P(w) * log(P(w) / Q(w))
return KL
```

---

## 3. 정보이론적 검증 알고리즘 2: Sentence-BERT 기반 Low Similarity Fraction (LSF)

`T_1`의 문장들이 `T_2`에 의해 의미적으로 커버되지 않는 비율을 계산하여, 새로운 의미론적 정보의 양을 평가한다 (상한 임계값 `U = 0.5`).

```
Input: Texts T1, T2; Upper Bound U ∈ [0, 1]
Output: LSF(T1 || T2): fraction of sentences in T1 whose average similarity to T2 is below U

Function PREPROCESSSENTENCES(T):
    return nltk.sent_tokenize(T)

Function EMBEDSENTENCES(S):
    return [SentenceBert(s) for s in S]

S1 = PREPROCESSSENTENCES(T1)
S2 = PREPROCESSSENTENCES(T2)

E1 = EMBEDSENTENCES(S1)
E2 = EMBEDSENTENCES(S2)

count = 0
For e1 in E1:
    simList = [cosineSim(e1, e2) for e2 in E2]
    s_mean = (1 / len(E2)) * ∑ simList
    if s_mean < U:
        count = count + 1

return count / len(E1)
```

---

## 4. 논리적 귀결성 평가: LLM-as-a-Judge INFERSCORE

추론 성분 `R`로부터 정답 텍스트 `Y`가 논리적으로 연역/귀결될 수 있는지를 독립된 LLM(GPT-4) 판정기를 통해 0/1로 채점하고 정답률을 집계한다.

```
Prompt to Judge:
"Given the task description, reasoning component R, and ground truth punchline Y:
Determine whether Y can be logically inferred and fully justified from R.
Output 1 if logically inferred, 0 otherwise."

INFERSCORE = (Number of samples with prediction 1 / Total samples) * 100
```
