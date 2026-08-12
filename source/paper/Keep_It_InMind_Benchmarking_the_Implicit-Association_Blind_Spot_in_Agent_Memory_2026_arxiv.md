# Keep It InMind: Benchmarking the Implicit-Association Blind Spot in Agent Memory — 핵심 발췌

> 출처: [분석 문서](../../report/[paper]_Keep_It_InMind_Benchmarking_the_Implicit-Association_Blind_Spot_in_Agent_Memory_2026_arxiv.md) / 원본: [arXiv:2607.24368](https://arxiv.org/abs/2607.24368) · [Project](https://keep-it-inmind.github.io/) · [GitHub](https://github.com/imlrz/InMind)
>
> (※ 본 파일은 분석에 인용한 핵심 정의·수식·표·결과를 발췌. 원문 표현을 최대한 보존.)

## Abstract (요지)

Long-term memory 시스템은 사용자 발언을 external store에 저장하고 관련 query 도착 시 검색한다. 이 인터페이스는 "필요한 memory는 그 memory를 필요로 하는 query와 닮아 있을 것"이라는 가정에 기반한다. World knowledge는 이 가정을 깬다 — tree-nut allergy는 알몬드가루 성분을 통해 macaron 요청의 정답을 바꿔야 하지만, 두 텍스트는 retriever가 볼 수 있는 단서를 공유하지 않는다. 이 실패 모드를 **implicit-association blind spot**이라 명명.

InMind: 125-task, expert-verified 벤치마크 (10개 life domain, 113 task는 citable public source 기반). paired control이 섞이는 세 설명을 분리 — (1) 사실이 저장된 적 없음, (2) model에 bridging knowledge 부족, (3) 사실은 저장됐으나 surface되지 않음.

결과: decisive memory가 context에 있으면 backbone이 간접 query의 84.0% 정답; 동일 memory를 검색해야 하면 6개 vector/graph/agentic memory 시스템은 최대 14.4% (같은 사실을 on-demand로는 최대 100% 회상). 8× 차원의 embedding은 모든 시스템의 answer-blind target recall을 올리나 gap은 사실상 유지. memory를 query 전부터 visible하게 유지하는 minimal diagnostic probe가 gap의 대부분을 회복 → 실패를 query-conditioned 인터페이스 자체에 국한, **routing**(어떤 사실이 visible state에 남아야 하는가)을 InMind가 측정하려는 open problem으로 지정.

## Retrieval Hypothesis (정식화)

저장 memory 집합 ℳ, query q. retrieval-based 시스템:

```
ℳ̂ = Retrieve(ℳ; q, θ),    a = LLM(q, ℳ̂)
```

θ는 retriever parameter·index·scoring rule·traversal policy. model은 ℳ̂만 추론하므로 순서가 결정적: **Retrieve는 LLM(world knowledge를 가진 유일 component)이 호출되기 전에 relevant를 결정**해야 한다.

> **Retrieval Hypothesis 1** — memory m이 query q의 답에 필요하면, m은 q만으로 계산되는 relevance score로 회복 가능하다: `m ∈ Retrieve(ℳ; q, θ)`.
>
> 단 θ는 efficient similarity computation에 한정 (전체 memory를 model로 돌리면 sublinear cost 포기=문제의 의미 없음).

## Implicit Association (정의)

memory–query pair (m, q)가 아래 3조건을 만족하면 **implicit association**:
1. **Necessity** — m이 q의 안전/정확/적절 답에 필수.
2. **Semantic distance** — m과 q가 lex/embedding/topical relevance에서 멂.
3. **Knowledge bridge** — m→k→q 인 background knowledge k가 존재 (둘 다에 명시 안 됨).

예: 고양이 소유 memory + "백합이 테이블 장식으로 좋을까?" query. 백합의 고양이 독성(vet 지식)이 bridge. query에 animal/pet/toxic 언급 없음. 약물-음식 상호작용, 교차 알레르기, 식이법, 직업 제한, 가정 위해 등 도메인 전반에 재발.

## 왜 query-conditioned retrieval이 실패하는가

dense retriever는 "cat"과 "lily"를 다른 topic으로 멀리 둠; sparse retriever는 공유 토큰 없음; graph retriever는 bridge가 이미 그래프에 있고 query가 활성화할 때만 도움. **bridge를 적용할 유일 component(model)는 memory와 query를 함께 보는 게 retrieval이 이미 선택한 후**. 어떤 설계의 추가 기계(graph edge / agentic sub-query / hierarchical store)든 결정 memory로 가는 경로는 memory가 보이기 전 만들어진 표현 기반 semantic-similarity link를 한 번은 건넘 — implicit association은 정확히 그 link에 신호를 주지 않는 pair.

→ **falsifiable 예측 (InMind가 검증할 signature)**:
(i) memory가 context에 직접 있으면 간접 query를 잘 답 (bridge knowledge 있음);
(ii) retrieval에서 explicit recall은 천장 근처 유지 (storage 작동);
(iii) indirect application은 붕괴 (interface 실패).
task 난이도/망각/model 약함 어느 설명도 이 패턴을 예측 못 함.

## InMind 벤치마크 구성

### Domain 분포 & source
- domain 가중치: Anthropic의 37,657 personal-guidance 대화 분석 차용 (consumer/other 소 slice까지 보존). persistent assistant가 실제로 consequential advice를 주는 영역에 집중.
- 각 domain마다 공공 검수 가능 collection 구성: FDA 알레르기 가이드, OSHA 규제, USCIS 여행 규칙, CPSC 리콜 등. **3,380 chunk를 비례 샘플링**, 각 chunk는 source trace(URL+text span) 보존 → bridge 감사 가능.

### Task 추출
각 source-grounded task는 source chunk에서 시작. extractor가 한 chunk를 읽고 유효 task 가능 여부 판단; 추출 가능 chunk는 record가 됨:
- (i) personal fact를 진술하는 user message
- (ii) 그 fact를 직접 묻는 **naive query**
- (iii) 정답이 그 fact 때문에 바뀌는 **indirect query** (fact/동의어 명시 금지)
- (iv) source-grounded bridge 설명

user message가 later object를 누설하면 안 됨; indirect query가 fact 명시 금지; chunk 자체가 bridge를 지원해야 함. Gemini 2.5 Flash 메인 extractor + Claude Sonnet 4.6 / Opus 4.8 / DeepSeek V4 Pro로 확장 → 1,000 candidate 생성.

### Filtering (3 content filter — 어느 것도 system 출력 미참조, retention은 system 정답 여부에 영향 안 받음)
1. **Similarity**: BM25 + MiniLM로 모든 memory–query pair 채점; target이 query에 obvious match인 700 candidate 폐기(implicit association 자격 상실). 자기 인코더로 평가 회피 위해 **BGE-small-en-v1.5**(filtering에 안 쓴 인코더)로 InMind 및 LoCoMo/LoCoMo-Plus/LME-s 재임베딩 → InMind만 target memory를 distractor보다 query에 가깝게 두지 않음 (retriever에 신호 없음).
2. **Conflict**: 모든 task가 하나의 배경 대화 trace 공유; persona와 모순된 주입 fact(예: pet 없는 history에서 cat owner) 폐기.
3. **Expert verification**: 인간 전문가가 bridge 사실 정확·fact가 답을 실질 변화·overt retrieval cue 잔존 없음 확인 → retain/revise/reject.

→ 300 survival에서 113 source-grounded + 12 expert-authored = **125-task set (전부 expert-verified, 113은 citable source)**. 많은 task가 safety-relevant by design (놓친 알레르기는 명백히 틀림; 놓친 선호는 논쟁적).

### 평가 프로토콜 (memory가 realistic interference에 생존해야)
- 각 task의 memory turn을 **47-session LongMemEval-s(LME-s) trace 중간에 삽입**, memory 시스템이 정상 동작하며 처리; 주입 fact는 query되기 전 **38 session 추가 일상 상호작용** 생존해야 함.
- 대화 후 2회 test: (1) 직접 naive query, (2) 간접 query.
- 간접 query에 **2개 보완 측정**:
  - **Target recall** — answer-blind 이진 사후 판정: answerer의 실제 context에 target fact가 있는가 (memory/context/query/bridge 제공, 생성 답안 미제공).
  - **Indirect application** — context-aware 답안 이진 판정: 답이 source-grounded bridge를 따르는가.
  - 분리 목적: error가 model이 fact를 보기 전(after retrieval miss)인지 본 후(after reasoning miss)인지 식별.

## 실험 — 실패 위치 특정

### System
- **In-context backbone control**: target memory + query를 직접 context에 → bridge world knowledge 측정.
- **Retrieval-based 6시스템**: A-RAG, xMemory, Mem0, A-Mem, HippoRAG 2, MemoryOS — 각 MiniLM(384-dim)과 text-embedding-3-large(3,072-dim)로 동작. 단발 Naive RAG control(MiniLM/emb3-large/BM25 over raw LME-s chunk). GPT-5-mini answerer + 이진 judge.

### Table 1 (Main Results, %) — 핵심
| System | Naive | Target Recall | Application |
|---|---|---|---|
| Backbone (GPT-5-mini) | – | 100.0 | 84.0 |
| Dense (MiniLM) | 92.0 | 0.8 | 9.6 |
| Dense (emb3-large) | 97.6 | 6.4 | 16.0 |
| BM25 | 53.6 | 3.2 | 9.6 |
| A-RAG (MiniLM / emb3) | 97.6 / 93.6 | 5.6 / 11.2 | 4.8 / 7.2 |
| xMemory (MiniLM / emb3) | 84.8 / 91.2 | 3.2 / 6.4 | 4.8 / 6.4 |
| Mem0 (MiniLM / emb3) | 76.0 / 76.8 | 2.4 / 6.4 | 3.2 / 6.4 |
| A-Mem (MiniLM / emb3) | 99.2 / 100.0 | 2.4 / 12.0 | 6.4 / 9.6 |
| HippoRAG 2 (MiniLM / emb3) | 93.6 / 96.0 | 0.8 / 2.4 | 8.8 / 5.6 |
| MemoryOS (MiniLM / emb3) | 87.2 / 96.8 | 2.4 / 7.2 | 8.0 / 14.4 |

**한 줄**: naive recall 최대 100.0%, indirect application 최대 16.0%. 6개 memory 시스템 자체는 최대 14.4%(Naive RAG control 16.0%보다도 낮음). 시스템 간 분산보다 전체가 84.0% backbone과 떨어진 거리가 큼 → blind spot은 개별 구현이 아닌 **paradigm의 속성**.

### 4.3 Not the Model (in-context가 rule out)
model 능력 부족 설명 배제. fact visible 시 GPT-5-mini 84.0% (105/125). Wilson 95% CI: backbone [76.6, 89.4] vs 최고 retrieval config [10.6, 23.4]. 84.0↔16.0을 가르는 건 능력이 아니라 **접근(access)**. relevance 결정 순간에 world knowledge가 lockout 됨.

### 4.4 Not Storage (naive가 rule out)
저장/생존 실패 설명 배제. naive 거의 완벽(A-Mem 100.0%) → fact write됨, 38 session 생존, on-demand 회상 가능. **target recall이 정확히 위치 특정**: 간접 query 하에 결정 fact가 answerer context에 도달 0.8–5.6%(MiniLM)/2.4–12.0%(emb3-large)에 불과. 저장됐고 model이 쓸 수 있는데 결정적 순간에 context 부재.

> 신뢰 역설: 사용자가 신뢰를 보정하려 테스트하는 상호작용(직접 질문)은 잘 작동하고, memory에 의존하는 상호작용은 거의 chance. 알레르기 회상에 답하고 macaron을 추천하는 agent는 받을 자격 없는 신뢰를 얻음.

### 4.5 Not Representation (8× 차원이 rule out 부분)
MiniLM(384)→emb3-large(3072) 교체. 모든 시스템 target recall 상승(A-Mem 2.4→12.0, A-RAG 5.6→11.2, MemoryOS 2.4→7.2 등). end-to-end application은 덜 깔끔: 6 중 5 향상, HippoRAG 2는 하락(8.8→5.6); 개별 shift는 125 중 몇 task로 ±4–5점 binomial CI 내. **aggregate 방향은 real, per-system 순위는 noise**.

기제: 강한 embedding이 훈련 중 world knowledge 일부를 흡수해 similarity 함수가 bridge 일부 인코딩. 그러나 embedding은 분포적 trace를 남긴 연관만 bridge 가능 — 본 task의 bridge는 약리/법률/종교/발달적, 대부분 co-occurrence로 적힌 적 없음. **8배 용량이 70점 gap 중 몇 점 회복**: world knowledge를 similarity score에 흡수하는 건 잘못된 방향에서 world model 근사, residual이 InMind가 측정하는 것.

## Gap을 닫으려면

### 5.1 Searching Harder로는 불충분
- GAM(General Agentic Memory) 등 lossless 보존+runtime search는 망각을 해결. 그러나 본 시스템은 이미 그 bar 통과(naive accuracy). 신호가 "봐라"를 못 할 때 retriever에 줄 것 없음: query 표현이 결정 fact가 보이기 전 고정. 더 큰 haystack을 더 철저히 뒤져도 바늘은 query와 닮지 않음.
- A-RAG: plan→keyword/semantic search→chunk read→최대 15회 loop → 4.8%/7.2%로 최하위권. 동일 blind 표현에서 나온 query로는 iteration이 search 못 구함.
- memory store는 oracle이 아님: query에 닮은 것 반환. bridge-aware "이 user의 무엇이 백합을 위험하게 만드나?"도 아무 stored fact가 안 닮음. 성공 probe는 "user가 고양이를 키우나?"(stored fact 어휘로). 그러려면 searcher가 unprompted에 lily의 고양이 독성을 회상→고양이를 단독(safe하려면 exact한 enumeration 필요). **bridge를 hypothesize하는 것 = store가 확인 전의 추측** — safety가 신뢰성을 요하는 곳에서 확률적. 그래서 fix가 아닌 open direction.

### 5.2 Always-in-State가 gap 회복 (Table 2)
- **always-in-state memory**: 표현 s_t가 query 도착 전 model에 도달, query-time retriever가 선택 안 함: `a = LLM(q, s_t)`. (injected profile / persistent summary / latent state 등 형식 무관; **persistent visibility**가 정의). model이 s_t와 q를 jointly 추론 → cat–lily bridge를 mismatched query 없이 발견.
- 이 설계는 드물지 않음: profile-style state는 배포 assistant의 표준; MemoryOS가 이미 user profile+knowledge entry를 매 답 context에 둠(hybrid). 부족한 건 측정 — hybrid의 persistent slice가 visible에 남아야 할 fact를 실제 담는지 알려주는 benchmark 없음.
- variable 고립 위해 **가장 bare 버전**: 마크다운 파일 1개(200 line cap), 매 session 후 GPT-5-mini updater가 rewrite, 답 시점 system prompt에 전체 prepend — embedding/vector store/index/ranking/query-time retrieval 전무. 그저 부지런히 관리되는 profile.

> **Table 2 — Always-in-state diagnostic (같은 125 task)**
> | Method | Naive | Indirect | Δ(Indirect−Backbone) |
> |---|---|---|---|
> | Backbone (GPT-5-mini) | – | 84.0 | 0.0 |
> | Best retrieval config | 97.6 | 16.0 | −68.0 |
> | Always-in-state (GPT-5-mini) | 98.4 | 68.8 | −15.2 |

- 68.8% vs 최고 query-time 16.0%, direct recall 98.4% match.
- *진술 신중*: probe는 단일 시스템 controlled ablation 아님(state format/updater도 다름) — 그러나 **sufficiency 증명**: memory가 query 전 visible인 설계 + 그저 유능한 write-time curation만으로 retrieval paradigm이 잃은 것의 대부분 회복. 모든 Table 1 행과 가른 것은 model과 memory 사이 query-time retriever가 없다는 것 한 가지이며, 그 설계 변경이 vector/graph/agentic search 누적 장치를 능가. **Hypothesis 1이 실패하는 것; retriever는 설계대로 작동**.

### 5.3 Routing as the Open Problem
- 두 설계는 complementary necessity: retrieval은 scale에서 lossless 기록 보존, always-visible state는 distant knowledge-connected fact 추론 허용. **hybrid가 결합**하지만 criterion이 없음.
- 모든 hybrid에 **routing decision** 내재: 어떤 memory가 write 시점에 persistent state를 얻나; 어떤 query가 read 시점에 fine-grained 기록 위 비싼 search를 정당화하나.
- 오늘 그 결정은 proxy(recency, frequency, MemoryOS heat-thresholded promotion)에 맡겨짐 — distant bridge를 못 봄. proper criterion은 vertical-specific(의료/재무/coding agent가 같은 forgotten fact를 다르게 가격).
- **InMind는 routing decision 자체를 측정하는 첫 benchmark**(hybrid의 router가 decision-critical fact를 visible state에서 빼면 indirect query 실패).
- routing은 문제를 미룰 뿐 해결 아님: too unimportant해 persistent state 못 얻은 fact도 distant bridge로 later에 decision-critical 될 수 있고, ordinary retrieval fallback 순간 노출. 그 gap을 닫으려면 **다른 종류의 relevance function**(knowledge-conditioned) 필요.

## Related Work (발췌 요점)
- **Memory for agents**: MemGPT(context/external 분리), vector store(Mem0 등), temporal/self-evolving, graph(HippoRAG), heterogeneous(A-Mem/xMemory). 모두 RAG의 query-then-retrieve interface 계승 → "저장하는 것"을 개선, **application이 InMind가 고립하는 gap**.
- **Memory benchmarks**: LongMemEval/LoCoMo(explicit recall/state aggregation/entity multi-hop); ImplicitMemBench(behavioral implicit — procedural/priming/conditioning, short protocol); LoCoMo-Plus(cognitive memory — latent constraint의 cue–trigger disconnect). **InMind = knowledge-mediated application of explicit fact**, bridge는 external/verifiable이고 대화에 안 나옴.
- **Construction provenance 차별점**: 기존 benchmark가 LLM 저작 연관(모델이 자연스럽게 찾는 연관)을 쓰면 construction과 test에 같은 prior 적용 → 모델이 이미 만드는 연관을 편향 샘플링. InMind의 bridge는 공공 문서 기반(source chunk만 지식 허용, extractor 차단) → 연관 분포가 model prior가 아닌 expert knowledge에 anchor.

> **Table 3 — 벤치마크 비교**
> | Benchmark | Memory type | Long ctx | Knowledge bridge | Source grounded | Direct-recall control |
> |---|---|---|---|---|---|
> | LoCoMo / LongMemEval | Factual | ✓ | × | × | – |
> | ImplicitMemBench | Behavioral implicit | × | – | × | × |
> | LoCoMo-Plus | Cognitive | ✓ | × | × | × |
> | InMind | Knowledge-mediated | ✓ | ✓ | ✓ | ✓ |

## Limitations (원문)
- 125 task는 health/wellness/safety에 치우침 — humor/etiquette/long-term goal/institutional policy 미커버. n=125에서 몇 %는 sampling noise (결론은 60–70점 효과에 기반).
- **GPT-5-mini가 answerer兼judge (self-preference risk)** — 절대 점수 부풀림 가능. 그러나 bias가 backbone/retrieval 동일 적용으로 gap 제조 불가, Appendix 14.2 전문가 감사가 각 judge error bound. **독립 judge model이 가장 크 방법론적 gap**.
- 각 bridge가 factual/uncontested 가정 — 실제는 probabilistic/jurisdiction-dependent/disputed 가능.
- InMind 최적화 agent가 over-warn 가능(bridge 적용에 credit, over-eagerness 비용 0 → 음수 control 필요, future work).

## Appendix 요점 (14.1 / 14.2)
- **14.1 Answer-Only Post-Hoc**: 저장된 간접 답안을 retrieved context 없이 GPT-5-mini 재평가(답 자체가 consequence 표현 여부). Backbone 85.6%(107/125), MemoryOS emb3 26.4%, A-Mem emb3 25.6%, retrieval 대부분 18–30% — context-aware application과 대체로 근접(답 자체가 consequence를 거의 안 담).
- **14.2 Human Audit (100 record)**: Target Recall 가장 신뢰 97.0%; context-aware Application 85.0%(오류 전부 false positive); answer-only 91.0%(false positive 잔존). Task 155(A-RAG/MiniLM): 회수 context에 알레르기 无(Target Recall 정확히 0)인데 답이 world knowledge로 generic almond/egg caveat 독립 생성 → answer-only 1은 적절하나 context-aware Application 1은 memory 사용으로 오귀인. 이것이 Application이 Target Recall 초과하는 기제.

## Conclusion (원문 요지)
memory는 query와 닮지 않으면서 essential할 수 있고, relevance는 retriever가 결여한 world knowledge로 생성됨. InMind에서 blind spot은 leading memory 시스템에 84.0↔16.0 차이를 비용(같은 사실 on-demand 최대 100% 회상). 불편한 판독: field가 이미 작동하는 곳(storage/indexing/graph/iterative search — agent가 무엇을 찾을지 아는 후 개선)에 노력을 썼고, 그것이 전체 난점. query-conditioned retrieval은 similarity에 world model 판단을 요구하고 model은 그 후에 참조. **knowledge-conditioned relevance function이 존재하기 전까지 agent memory는 user가 다시 알려줄 수 있는 곳에선 reliable, keep in mind했다고 가정한 곳에선 unreliable**.
