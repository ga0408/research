# Memory Retrieval (메모리 검색) — Low-level Detail

> 출처: [분석 문서](../../../report/[paper][git]_Memora_A_Harmonic_Memory_Representation_Balancing_Abstraction_and_Specificity_2026_ICML.md) / submodule 경로: `source/git/Memora_microsoft`

## 설명

Memora 검색 파이프라인 전체 디테일. QueryGenerator, AgentMemory.query, MemoryFilter, MemoryExpander, 3종 Retriever의 함수 호출 흐름과 LLM 프롬프트 원문 포함.

---

## 1. 전략 선택 — `MemoraClient.advance_query()` (`memora_client.py:145`)

```python
def advance_query(self, context, top_k=5, query_type="prompt", checkpoint_path=None):
    if query_type == "semantic":
        retriever = SemanticRetriever(self._client.cfg, memory_client=self._client)
    elif query_type == "prompt":
        retriever = PromptedPolicyRetriever(self._client.cfg, memory_client=self._client)
    elif query_type == "grpo":
        retriever = LocalPolicyRetriever(self._client.cfg, memory_client=self._client,
                                          checkpoint_path=checkpoint_path)
    return retriever.retrieve(query=context, top_k=top_k)
```

일반 `query()`는 `LocalMemoraClient.query()` → `AgentMemory.query()` 직접 호출 (SemanticRetriever와 동일 경로).

---

## 2. QueryGenerator — LLM Query 확장 (`core/query_generator.py:60`)

`AgentMemory.query()`에서 `enhance_query=True` 시 호출.

### 프롬프트 — `PROMPT_GENERATE_QUERIES`

```text
ROLE
You are the Memory Query Generator. Your job is to transform a context into concise, specific queries for retrieving relevant memory. The context may not explicitly state a task — you must infer what task is implied and what information is required to support it.

INSTRUCTIONS
1. Carefully read the full context.
2. Infer the most likely task or intent implied by the context (e.g., troubleshoot, summarize, explain, retrieve past actions).
3. Decide what information from memory would be necessary to accomplish that task.
4. Generate one or multiple **concise (6–14 words)** queries capturing exactly that need.
5. There are multiple types of memories available such as "factual","procedural" and "episodic" memory.

Here is the context:
{context}
```

### 구조화 응답

```python
class MemoryQuery(BaseModel):
    query: str = Field(description="A short, precise query capturing exactly that needed for retrieval.")

class MemoryQueries(BaseModel):
    queries: list[MemoryQuery]
```

### 키워드 추출 (hybrid search용) — `PROMPT_EXTRACT_KEYWORDS`

```text
ROLE
You are a Keyword Extractor for memory retrieval. Your job is to identify the most important keywords and phrases from a context that would be useful for keyword-based search.

INSTRUCTIONS
1. Carefully read the context.
2. Extract meaningful short phrases **(2-4 words)** that can be used to perform exact-match searches to retrieve relevant information.
3. Keep phrases concise, but avoid single-word names unless absolutely necessary. The phrases do not need to include the user name. Avoid using "and" in the phrases.
4. Return 1-4 keywords/phrases depending on context complexity.

Here is the context:
{context}
```

```python
class Keyword(BaseModel):
    keyword: str

class Keywords(BaseModel):
    keywords: list[Keyword]
```

---

## 3. `AgentMemory.query()` — 검색 엔진 본체 (`core/memory.py:300`)

### 3.1 전체 흐름

```python
def query(self, context, top_k=5, where=None, query_mode=QueryMode.ORIGINAL,
          enhance_query=True, enable_hybrid_search=False, enable_llm_filter=False, ...):

    context = context_to_str(context)

    # (1) Query 확장
    if enhance_query:
        queries = self.query_generator.generate_queries(context)  # LLM → List[str]
    else:
        queries = [context]

    # (2) 모드별 검색
    if query_mode == QueryMode.BOTH:
        primary_results = self._query_result(queries, top_k, where, QueryMode.PRIMARY_ONLY, ...)
        cue_results = self._query_result(queries, self.cfg.memory.cue_top_k, where, QueryMode.CUE_ONLY, ...)
    elif query_mode == QueryMode.PRIMARY_ONLY:
        primary_results = self._query_result(queries, top_k, where, QueryMode.PRIMARY_ONLY, ...)

    # (3) Hybrid Search (옵션)
    if enable_hybrid_search:
        hybrid_results = self._perform_hybrid_search(context, where)

    # (4) RRF 가중 합산
    if enable_hybrid_search and len(result_lists) > 1:
        memory_results = self._merge_results_with_rrf(result_lists, weights)
    elif query_mode == QueryMode.BOTH:
        memory_results = self._merge_results_with_rrf([primary_results, cue_results], [2.0, 1.0])

    # (5) LLM Filter (옵션)
    if enable_llm_filter and memory_results:
        memory_results = self.memory_filter.filter_memory(query=context, memory_results=memory_results)

    return memory_results[:top_k]
```

### 3.2 `_query_result()` — Primary/Cue 검색 + 해결 (`memory.py:111`)

```python
def _query_result(self, queries, top_k, where=None, query_mode=QueryMode.ORIGINAL, ...):
    memory_results = []
    extracted = set()  # value 기준 중복 제거

    # where 조건 설정
    if query_mode == QueryMode.CUE_ONLY:
        where = {"linked_memory": {"$ne": ""}}              # cue entry만
    elif query_mode == QueryMode.PRIMARY_ONLY:
        where = {"$and": [{"linked_memory": {"$eq": ""}},    # primary만
                          {"memory_type": {"$eq": "factual"}}]}  # factual만

    for query in queries:  # 각 query 변형마다
        results = self._store.query(query, top_k, where, include)
        scores = [entry.score for entry in results]
        merged_sorted = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)

        for entry, score in merged_sorted:
            if score < self.QUERY_SCORE_THRESHOLD:   # config 임계값
                break                                 # 정렬됐으므로 이후는 전부 미만

            if entry.is_cue_index():
                # cue hit → linked primary memory fetch
                for primary_index in entry.get_linked_memories():
                    primary_entry = self._store.get(primary_index)
                    if not primary_entry: continue
                    value = primary_entry.get_memory_value(return_history=return_history)
                    if value in extracted: continue           # 중복 제거
                    memory_results.append(primary_entry)
                    extracted.add(value)
            else:
                # primary hit
                value = entry.get_memory_value(return_history=return_history)
                if value in extracted: continue
                memory_results.append(entry)
                extracted.add(value)

    return memory_results[:top_k]
```

**유사도 계산**: ChromaDB distance → `score = 1 - distance` (L2 거리 기반).

### 3.3 `_perform_hybrid_search()` — BM25 / Keyword (`memory.py:185`)

```python
def _perform_hybrid_search(self, context, where):
    hybrid_method = self.cfg.memory.get("hybrid_search_method", "bm25")
    hybrid_top_k = self.cfg.memory.get("hybrid_top_k", 10)

    if hybrid_method == "bm25":
        target_user_id = extract_user_id_from_where(where) or self.user_id
        if target_user_id not in self._store._bm25_indices:
            self._store.build_bm25_index(user_id=target_user_id)  # 최초 1회 빌드
        bm25_threshold = self.cfg.memory.get("bm25_score_threshold", 0.4)
        hybrid_results = self._store.bm25_search(context, hybrid_top_k, where, bm25_threshold)

    elif hybrid_method == "keyword":
        keywords = self.query_generator.extract_keywords(context)  # LLM 호출
        hybrid_results = self._store.keyword_search(keywords, hybrid_top_k, where)
```

**BM25 상세** (`local_memory_store.py:441`):
- 사용자별 `BM25Okapi` 인덱스 빌드: 모든 메모리의 `index + value` 토큰화
- 쿼리 토큰화 → BM25 점수 산출 → 임계값 이상 top-k
- cue entry hit 시 linked primary fetch, episodic memory 제외

**Keyword search 상세** (`local_memory_store.py:223`):
- 긴 구문(3+ 단어)은 부분 구문 매칭 허용
- 짧은 구문(1-2 단어)은 정확 매칭
- 단어 수 기반 점수 → semantic threshold 이하로 스케일

### 3.4 `_merge_results_with_rrf()` — RRF 합산 (`memory.py:231`)

```python
def _merge_results_with_rrf(self, result_lists, weights=None, k=60):
    if weights is None:
        weights = [2.0] * len(result_lists)
    # 기본 가중치: primary(2.0) > cue(1.0) = hybrid(1.0)

    rrf_scores = {}
    all_entries = {}
    for result_list, weight in zip(result_lists, weights):
        for rank, entry in enumerate(result_list, start=1):
            record_id = index_to_id(entry.index)  # sha256(index)
            if record_id not in all_entries:
                all_entries[record_id] = entry
            rrf_scores[record_id] = rrf_scores.get(record_id, 0) + weight * (1.0 / (k + rank))

    # 0-1 정규화
    max_score, min_score = max(rrf_scores.values()), min(rrf_scores.values())
    if max_score > min_score:
        rrf_scores = {rid: (s - min_score)/(max_score - min_score) for rid, s in rrf_scores.items()}

    # score 갱신·정렬
    for rid, s in rrf_scores.items():
        all_entries[rid].score = s
    return sorted(all_entries.values(), key=lambda x: x.score, reverse=True)
```

**RRF 공식**: `RRF_score(d) = Σ weight_i × 1/(k + rank_i)`, k=60

---

## 4. MemoryFilter — LLM 관련성 필터 (`core/memory_filter.py:95`)

`enable_llm_filter=True` 시 검색 후 호출.

### 프롬프트 — `PROMPT_MEMORY_FILTER`

```text
You are an expert Memory Refiner for a retrieval-augmented agent. Your task is to evaluate the relevance of retrieved memories in relation to a user query.

# TASK: 
Given a user query and a list of retrieved memories in the format of [memory_index]: memory_value, rate the relevance of each memory to the query on a scale from 1 to 3.
The scores will be used to filter out irrelevant, unhelpful, or outdated memories.
Then, return a JSON object containing the scores for each memory.

# GUIDELINES:
1. Scoring Criteria:
    - Score 3: The memory is very relevant and directly helps in answering the query.
    - Score 2: The memory might be useful or somewhat relevant to the query. It could provide some context or background information. It might not be directly necessary but still has value.
    - Score 1: The memory is completely unrelated, unhelpful or outdated for answering the query. It does not contribute any useful information to answering the query.

2. Evaluation Considerations:
    - Focus on the relevance of the memory content to the specific query.
    - Ensure that each memory is evaluated independently based on its own content.
    - Be objective and consistent in your scoring.

# OUTPUT FORMAT:
Return a JSON object with a "scores" array. Each entry should have:
- "index": the exact memory index (e.g., "Mike's birthday")
- "score": relevance score from 1 to 3

Example output:
{
    "scores": [
        {"index": "Mike's birthday", "score": 3},
        {"index": "Stacy's favorite color", "score": 1},
        {"index": "Mike's family gathering", "score": 2}
    ]
}

User Query: {query}

Retrieved Memories:
{memories_text}

Evaluate all memories and provide a score for each one.

Output:
```

### 필터 로직

```python
class MemoryFilter:
    def filter_memory(self, query, memory_results):
        memories_text = "\n".join([f"[{entry.index}]: {entry.get_memory_value()}" for entry in memory_results])
        response = self._model_client.invoke(input=PROMPT_MEMORY_FILTER,
                                              prompt_args={"query": query, "memories_text": memories_text},
                                              response_format=MemoryScoreResponse)
        score_dict = {item.index: item.score for item in response.scores}

        # score ≥ 2만 유지, LLM score 내림차순 → search score 내림차순 정렬
        scored_results = [(e, score_dict.get(e.index, 0), e.score) for e in memory_results
                          if score_dict.get(e.index, 0) >= 2]
        scored_results.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return [e for e, _, _ in scored_results]
```

---

## 5. SemanticRetriever (`retriever/semantic_retriever.py:20`)

```python
class SemanticRetriever(BaseMemoryRetriever):
    def __init__(self, cfg, memory_client=None):
        self.memory_client = memory_client
        self.top_k = cfg.memory.get("top_k", 30)
        self.enable_hybrid_search = cfg.memory.get("enable_hybrid_search", False)
        self.enable_llm_filter = cfg.retrieval.get("enable_llm_filter", False)
        if cfg.memory.get("enable_cue_index", False):
            self.query_mode = QueryMode.BOTH
        else:
            self.query_mode = QueryMode.PRIMARY_ONLY

    def retrieve(self, query, top_k=None, ...):
        # config 기본값 적용 후 AgentMemory.query 호출
        memories = self.memory_client.query(
            query, top_k=top_k, enable_hybrid_search=enable_hybrid_search,
            enable_llm_filter=enable_llm_filter, query_mode=query_mode,
            where={"memory_type": {"$eq": "factual"}},  # factual만 검색
        )
        return memories
```

`SemanticRetriever`는 사실상 `AgentMemory.query()`의 thin wrapper.

---

## 6. PromptedPolicyRetriever — Iterative MDP (`retriever/prompted_policy_retriever.py:96`)

### 6.1 `retrieve()` — 전체 루프

```python
def retrieve(self, query, top_k=None, ...):
    self.expander.reset()
    self.last_trace = []

    # Step 0: INIT_RETRIEVE
    memory_entries = self.memory_client.query(query, top_k=top_k, query_mode=self.query_mode, ...)
    frontier = {}
    frontier = self.expander.build_frontier(frontier, memory_entries)

    # Step 1..max_steps
    for step_idx in range(1, self.max_steps + 1):
        decision = self.prompted_policy(
            user_question=query, current_query=current_query,
            memory_entries=memory_entries, frontier=frontier,
            step=step_idx, trace=self.last_trace)
        action = decision.get("action", "STOP")

        if action == "STOP":
            break
        elif action == "EXPAND":
            chosen = self._select_from_frontier(frontier, decision.get("frontier_ids", []))
            memory_entries = dedup_memories(memory_entries + chosen)
            for mem in chosen: frontier.pop(mem.index, None)
            frontier = self.expander.build_frontier(frontier, chosen)  # frontier 재구축
        elif action == "RE_QUERY":
            new_query = decision.get("new_query", current_query)
            current_query = new_query
            new_entries = self.memory_client.query(current_query, top_k=top_k, ...)
            memory_entries = dedup_memories(memory_entries + new_entries)
            frontier = self.expander.build_frontier(frontier, new_entries)
    return memory_entries
```

### 6.2 `prompted_policy()` — LLM 정책 결정

### 프롬프트 — `LLM_POLICY_PROMPT`

```text
SYSTEM:
You are Memora-Control, a decision policy for an iterative memory system. Your goal is to help retrieve relevant memories to answer user questions.

Initial retrieval has ALREADY been performed, yielding a Working Set (W) of memories and a Frontier (F) of expansion candidates.
Your job is to decide how to proceed using ONLY these actions:
- EXPAND: grow memory from expansion candidates in Frontier if they add useful information to answering the user question.
- RE_QUERY: regenerate a better query and retrieve again, if current memories are insufficient to answer the question, or important facts are missing.
- STOP: enough information has been collected in the Working Set (W) to answer the user question.

Definitions:
- Working Set (W): memories already collected
- Frontier (F): expansion candidates reachable from W
- Expansion is cheaper than re-querying the full corpus

Decision rules:
1) Choose STOP if W is sufficient to answer the user question with high confidence.
2) Choose EXPAND if frontier contains high-novelty items likely to fill remaining gaps.
3) Choose RE_QUERY if important facts are missing AND expansion is unlikely to find them.
4) Here are some typical scenarios for RE_QUERY:
    (a) Query refinement: The previous query failed to return relevant results, or the question requires a different semantic angle.
    (b) Relative answers: If W provides a relative answer (a "pointer") rather than a direct value. You need to RE-QUERY for the specific target.
        - *Example:*
            - Query: "Where did Mike go to college?"
            - W: "Mike went to the same college as Sarah"  (relative answer, no specific college - gap identified)
            - Action: RE_QUERY -> `new_query`: "Where did Sarah go to college?" 
5) Prefer EXPAND over RE_QUERY when both are viable.
6) Minimize redundancy and unnecessary steps.

Output STRICT JSON. No extra text.

JSON schema:
{
  'action': 'EXPAND | RE_QUERY | STOP',
  'reason': 'one sentence',
  'confidence': 0.0-1.0,
  'frontier_ids': ['id1','id2'],  // required if EXPAND, pick from frontier
  'new_query': 'string',          // required if RE_QUERY
}

UserQuestion:
{user_question}

CurrentQuery:
{current_query}

Step:
{step}/{max_steps}

WorkingSetSummary:
{W_summary}

FrontierSummary:
{F_summary}

Running History:
{trace}

Constraints:
- Prefer EXPAND over RE_QUERY when possible
- Avoid repeating information already covered in W
- RE_QUERY should be concise and retrieval-optimized
- Avoid RE_QUERY with the same query as before
```

### 6.3 W/F 요약 포맷

```python
def _format_working_set(self, memories):
    # "[1] Alice's new job at Contoso: Alice is starting a new job at Contoso in Seattle next month..."
    # value 150자 트렁케이션

def _format_frontier(self, frontier):
    # "- [Alice's pet plans]: Alice previously considered getting a cat..."
    # value 100자 트렁케이션
```

---

## 7. MemoryExpander — Frontier 구축 (`core/memory_expander.py:16`)

### `build_frontier()` 전체 흐름

```python
def build_frontier(self, frontier, memories):
    working_set = {memory.index for memory in memories}
    cue_to_memory_score = {}  # cue -> best memory score (relaxed용)
    direct_cues = set()

    # Step 1: W 메모리들의 cue indices 수집 (visited 추적)
    for memory in memories:
        if memory.index in self.visited_ids: continue
        self.visited_ids.add(memory.index)
        memory_score = memory.score if memory.score is not None else 1.0
        for cue_index in memory.get_cue_indices():
            if cue_index not in self.visited_ids:
                direct_cues.add(cue_index)
                if cue_index not in cue_to_memory_score or memory_score > cue_to_memory_score[cue_index]:
                    cue_to_memory_score[cue_index] = memory_score

    # Step 2: relaxed frontier (옵션) — 유사 cue 병렬 검색
    all_cues = set(direct_cues)
    if self.enable_relaxed_frontier and direct_cues:
        # score 순 정렬, 상위 max_cues_to_expand(30)개만
        sorted_cues = sorted(cue_to_memory_score.items(), key=lambda x: x[1], reverse=True)
        cues_to_expand = [c for c, _ in sorted_cues[:self.max_cues_to_expand]]
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_cue = {executor.submit(self._find_similar_cues, c): c for c in cues_to_expand}
            for future in as_completed(future_to_cue):
                for similar_cue in future.result():
                    if similar_cue.index not in self.visited_ids:
                        all_cues.add(similar_cue.index)

    # Step 3: 각 cue → linked primary memory를 frontier에 추가
    for cue_id in all_cues:
        if cue_id in self.visited_ids: continue
        self.visited_ids.add(cue_id)
        cue_entry = self.memory_client.get(cue_id)
        if not cue_entry: continue
        for linked_index in cue_entry.get_linked_memories():
            if linked_index in working_set or linked_index in frontier: continue
            linked_entry = self.memory_client.get(linked_index)
            if linked_entry:
                frontier[linked_entry.index] = linked_entry
    return frontier
```

### `_find_similar_cues()` — 유사 cue 검색

```python
def _find_similar_cues(self, cue_index):
    similar_cues = self.memory_client.query(
        cue_index, top_k=self.relaxed_frontier_top_k,   # default 4
        enable_hybrid_search=False,
        query_mode=QueryMode.CUE_ONLY  # cue entry만 검색
    )
    filtered = [c for c in similar_cues
                if c.score >= self.relaxed_frontier_threshold  # default 0.85
                and c.index != cue_index]
    return filtered
```

---

## 8. LocalPolicyRetriever — GRPO Qwen (`retriever/local_policy_retriever.py:112`)

`PromptedPolicyRetriever`와 **동일한 retrieve() 루프**. 정책 결정만 로컬 Qwen 모델:

```python
class LocalPolicyRetriever(BaseMemoryRetriever):
    def __init__(self, cfg, memory_client=None, checkpoint_path=None, ...):
        self.model_name = local_cfg.get("model_name", "Qwen/Qwen2.5-7B-Instruct")
        self.checkpoint_path = checkpoint_path  # None=baseline, path=GRPO fine-tuned
        self.expander = MemoryExpander(memory_client=memory_client, ...)

    def _call_policy(self, user_question, current_query, working_set, frontier, step):
        self._load_model()  # 전역 캐시 (PeftModel + LoRA checkpoint)
        user_message = format_user_message(...)  # policy_utils 공유 포맷
        messages = [{"role": "system", "content": POLICY_SYSTEM_MESSAGE},
                    {"role": "user", "content": user_message}]
        prompt_ids = self.tokenizer.apply_chat_template(messages, tokenize=True,
                                                         add_generation_prompt=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(prompt_ids, max_new_tokens=300, do_sample=False, ...)
        response = self.tokenizer.decode(outputs[0, prompt_ids.shape[1]:], skip_special_tokens=True)
        decision = parse_json_response(response)      # policy_utils 공유
        decision = validate_policy_decision(decision, frontier)
        return decision
```

### 공유 프롬프트 — `POLICY_SYSTEM_MESSAGE` (`retriever/policy_utils.py:90`)

`PromptedPolicyRetriever`의 `LLM_POLICY_PROMPT`와 내용 동일. 차이: `frontier_ids`에 정확한 index string 사용 强调 (`"User's favorite color"` not `"1"`).

### GRPO 학습 (`src/memora/rl/`)

| 파일 | 역할 | Paper 대응 |
|---|---|---|
| `collect_trajectories.py` | trajectory 샘플링 | `τ^(i) = {(st,at)}` |
| `trajectory_scorer.py` | groundedness/redundancy/cost 평가 | `J(τ) = w1·Ground − w2·Redund − w3·Cost` |
| `grpo_trainer.py` | group-relative advantage + LoRA 훈련 | `Ã(i) = J(τ^(i)) − mean(J)`, `L_GR(θ)` |
| `policy_qwen.py` | Qwen 정책 모델 | `πθ(a|s)` |
