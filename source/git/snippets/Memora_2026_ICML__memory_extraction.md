# Memory Extraction (메모리 추출) — Low-level Detail

> 출처: [분석 문서](../../../report/[paper][git]_Memora_A_Harmonic_Memory_Representation_Balancing_Abstraction_and_Specificity_2026_ICML.md) / submodule 경로: `source/git/Memora_microsoft`

## 설명

Memora의 메모리 추출 파이프라인 전체 디테일. 각 단계의 함수 호출 흐름, 데이터 변환, LLM 프롬프트 원문, ChromaDB 저장 구조까지 포함.

---

## 1. 진입점 — `LocalMemoraClient.add()` (`src/memora/core/local_client.py:175`)

```python
def add(self, text=None, type=None, metadata=None, progress_callback=None):
    # 입력을 단일 Segment로 감싼다 (긴 텍스트 분할은 미구현 TODO)
    segments = [Segment(content=text, segment_type="text", metadata=metadata)]
    memory_builder: MemoryBuilder = self._get_memory_builder(type)
    self._last_builder = memory_builder

    memory_entries = []
    for segment in segments:
        merged_metadata = merge_metadata(segment.metadata, metadata)
        memory_entries.extend(
            memory_builder.build(segment.content, metadata=merged_metadata,
                                  progress_callback=progress_callback)
        )
    return memory_entries
```

**Builder 선택** (`_get_memory_builder`): `type` 파라미터로 매핑.
- `"chat"` / `"default"` → `ChatMemoryBuilder`
- `"markdown"` / `"doc"` / `"pdf"` / `"word"` → `DocumentMemoryBuilder`

**파일 입력** (`add_file`): `ProcessorRegistry`가 파일 타입별로 Segment 분할.
- Markdown: `# 헤더` 기준 → `Segment(segment_type="section")`, heading_path 메타데이터 포함
- Plain text: `\n\n` 빈 줄 기준 → `Segment(segment_type="paragraph")`

---

## 2. `MemoryBuilder.build()` — 추출 메인 파이프라인 (`src/memora/builder/memory_builder.py:151`)

### 2.1 입력 정규화 — `normalize_content()` (`utils/misc.py:27`)

다양한 입력 포맷을 통일된 형식으로 변환:

```
입력 str          → {"text": "..."}
입력 List[str]    → {"text": "...\n..."}  (join)
입력 List[Dict]   → {"text": "user: ...\nassistant: ...", "image": [image_url parts]}
                    (conversation format, multimodal 지원)
```

이미지가 있고 `multimodal_support=True`면 `{"text":..., "image":[...]}` dict 반환. 없으면 `{"text":...}` dict 반환.

### 2.2 메타데이터 생성 — `generate_metadata()` (`utils/memory.py:93`)

```python
def generate_metadata(content, metadata):
    if not metadata: metadata = {}
    creation_time = get_current_timestamp()          # "2026-07-07 12:00:00" (ISO UTC)
    metadata["creation_time"] = creation_time        # 배치 추적 (같은 배치 update 방지용)
    # multimodal content에서 image_urls 추출
    if isinstance(content, dict) and "image" in content:
        image_urls = [img["image_url"]["url"] for img in content["image"] if img.get("type")=="image_url"]
        if image_urls: metadata["image_urls"] = image_urls
    return metadata
```

### 2.3 Step 1: Episodic Memory (config: `enable_episodic_memory=True`)

`ChatMemoryBuilder.generate_episodic_memory()` (`chat_memory_builder.py:145`):

**`use_segments_as_episodic=True`** (기본): 원본 segment 텍스트를 그대로 value로.
```
MemoryEntry(
    memory_type="episodic",
    index="[EPISODIC] <segment_topic> (segment <N>)",   # 메타데이터에서 추출
    value=<원본 conversation text>,                       # 압축 없음
    original_text=<원본 text>,
)
```

**`use_segments_as_episodic=False`**: LLM(`PROMPT_EPISODIC_MEMORY`)이 1-3문장 요약. `EpisodicMemoryOutput{episodic_index, episodic_value}` 구조화 응답.

**저장 후 연결**: `agent_memory.add(episodic_entry)` → `metadata["episodic_memory_id"] = episodic_entry.index` → 이후 factual memory들이 이 ID 참조.

### 2.4 Step 2: Factual Memory 추출 — LLM 호출

`ChatMemoryBuilder.generate_memory_entries()` (`chat_memory_builder.py:71`):

#### 프롬프트 — `PROMPT_BUILD_MEMORY` (factual memory 추출)

```text
You are an expert fatual memory extraction assistant. Your goal is to extract factual memories from a conversation segment.

# TASK: 
Read the input conversation carefully, extract ALL factual memories that could be useful for future reference.

Produce each memory as a key-value pair in the following format:

MemIndex: memory index for retrieval
MemValue: memory value with all the details supported directly from the given text.

# GUIDELINES:
1. Content and Scope:
- Use only information explicitly mentioned in the context to create the factual memories.
- Make sure to capture ALL factual information that could be useful for future retrieval. When in doubt, create more factual memories rather than fewer. Capture more details rather than less.
- Do not include greetings, small talk, or filler in the memories.
- Be exhaustive when extracting factual memories from the conversation, do not miss any details, unless they are explicitly irrelevant and unlikely to be useful for future reference (like greetings or small talk).
- Split distinct facts into separate entries.
- Capture all details about people's identities, experiences, past or upcoming events, intentions, hobbies, preferences, states, beliefs, goals, or future plans that may be useful for answering later questions.
- Make sure to include time of events, location, or other contextual details in the MemValue if mentioned.
- If images are included in the conversation, consider the images as a part of the text. DO NOT create a memory entry solely to describe the image. Instead, extract useful facts from the image (e.g., objects, locations) and merge them naturally into relevant memories. For example, if a conversation contains an image showing a birthday cake with candles, include information about that image in the relevant memory about the birthday event, such as "MemIndex: Alice celebrated her birthday\nMemValue: Alice celebrated her birthday with her friends at home, including a birthday cake with candles".
- If the conversation is between a user and an AI assistant, focus on the user's inputs and the overall context rather than the assistant's responses.

2. Format and Style:
- The MemIndex must be a short, human-readable phrase that is self-contained and unambiguous.
- Always include the specific context (e.g., domain, or entity) from the source text in the MemIndex to avoid vague terms. For example, instead of "Vacation", use "Alice's Japan Vacation". Instead of "Mike's plans", use "Mike's summer plans to visit Europe".
- Write MemValue as one or two full factual sentences, capturing all relevant details. 
    - Ensure wording is neutral and factual.
    - Use the original wordings from the conversation when possible.
    - Replace pronouns with specific names or entities to ensure clarity.
    - Handling times and dates in the MemValue: When dates and times are mentioned in the conversation, replace relative times (e.g., "yesterday", "next week", "last year") with absolute dates based on the timestamp of the conversation. For example, if the conversation timestamp is "16 June, 2023" and it mentions something happened "last year", convert it to "2022" in the MemValue.

Timestamp of conversation: {timestamp}

Input Conversation:
{content}

Output:
```

#### 구조화 응답 — `MemoryOutputs`

```python
class MemoryOutput(BaseModel):
    memory_type: str = Field(description="Type of memory: 'Factual' or 'Procedural'.")
    index: str = Field(description="short, specific phrase that captures the fact clearly")
    value: str = Field(description="A concise but complete factual statement")

class MemoryOutputs(BaseModel):
    entries: list[MemoryOutput]
```

#### LLM 호출 후 변환 — `convert_memory_output()` (`utils/memory.py:167`)

```python
def convert_memory_output(memories, metadata, enable_cue_index):
    memory_entries = []
    episodic_memory_id = metadata.get("episodic_memory_id", None)
    for memory_output in memories.entries:
        cue_indices = ""
        if enable_cue_index and hasattr(memory_output, "cue_indices") and memory_output.cue_indices:
            cue_indices = "||".join(memory_output.cue_indices)
        episodic_memory_ids = [episodic_memory_id] if episodic_memory_id else []
        entry = MemoryEntry(
            memory_type=memory_output.memory_type,
            index=memory_output.index,
            value=memory_output.value,
            creation_time=metadata["creation_time"],
            timestamp=metadata.get("timestamp", ""),
            cue_indices=cue_indices,
            episodic_memory_ids=episodic_memory_ids,
        )
        memory_entries.append(entry)
    return memory_entries
```

`ChatMemoryBuilder`는 모든 entry의 `memory_type`을 `"factual"`로 강제 세팅.

### 2.5 Step 2-후: Cue Anchor 배치 생성

`enable_cue_index=True` → `CueIndexGenerator.generate_cue_indices_batch()` (`cue_index_generator.py:175`):

**단일 LLM 호출**로 모든 메모리의 cue index를 한번에 생성.

#### 프롬프트 — `PROMPT_CUE_GENERATION`

```text
You are a memory-indexing assistant optimized for knowledge retrieval. Your goal is to create "Cue Indices" that serve as semantic anchors for specific memories.

# TASK
For each memory provided, generate 1-3 short, meaningful CUE INDICES that can later help recall or reason about that memory. Provide the cue indices as a list of strings for each memory.

# GUIDELINES
1. **Definition**: A cue index is a concise phrase (2-4 words) that anchors a specific topic to a memory. It takes the following structure: [Main Entity] + [Key Aspect].
    - The **Main Entity** is the primary person, domain, or object involved in the memory (the "Who" or "What").
    - The **Key Aspect** specifies the event, preference, action, state, or object associated with the entity.
    Examples of Main Entity + Key Aspect patterns:
        - [Person] + [Event/Activity] → "Jane hiking trip", "Mike vacation"
        - [Person] + [Hobby/Preference] → "Michael Jazz music", "Sophie vegan diet"
        - [Person] + [Condition/State] → "Emma career change", "Liam health problems"
        - [Person] + [Object/Relation] → "Alice research paper", "David guitar"
        - [Domain] + [Attribute/Artifact] → "Project Orion timeline", "Product X features"

2. **Specificity**: Avoid generic single words like "summer", "happiness", or "project meeting". Every cue index must be contextually anchored to the main entity. Use "Mike mental health problems" instead of "Mike feelings."
3. **Atomicity**: Each cue index must represent a single, indivisible aspect. Do not overload a cue with timestamps, specific numbers, or multiple descriptors. Use "Mike birthday party" instead of "Mike birthday party 2023".
4. **Distinct Facets**: Each index must target a completely different dimension of the memory. Avoid generating cue indices that are similar to each other for the same memory.
5. **Uniqueness**: Do not repeat the primary memory index as a cue index.
6. **Purpose**: Cue indices could help with recall and reasoning by providing additional semantic keys beyond the primary index.

# EXAMPLES
Primary Index: "Jane's hiking trip to Appalachian Trail"
Memory Value: "Last summer, Jane went on a week-long hiking trip along the Appalachian Trail. She enjoyed the scenic views and challenging trails."
Cue indices: ["Jane hiking","Appalachian Trail views","Jane summer trip"]

# MEMORIES TO PROCESS
{memories}
```

#### 구조화 응답 — `BatchCueIndices`

```python
class MemoryCueIndices(BaseModel):
    memory_index: str
    cue_indices: List[str]

class BatchCueIndices(BaseModel):
    results: List[MemoryCueIndices]
```

#### 결과 부착

```python
cue_indices_map = generator.generate_cue_indices_batch(memories_batch)
for entry in memory_entries:
    cue_indices = cue_indices_map.get(entry.index, [])
    entry.cue_indices = "||".join(cue_indices) if cue_indices else ""
```

### 2.6 Step 3: `upsert_memory_entry()` — 각 entry별 중복·갱신 (`memory_builder.py:365`)

```
각 entry마다:
├─ (1) 동일 index 검사: agent_memory.get(entry.index)
│       존재 → build_stats["duplicate"] += 1, return (skip)
│
├─ (2) 유사 메모리 탐색: _query_update_candidates(entry)
│       where = {"$and": [
│           {"creation_time": {"$ne": entry.creation_time}},   # 같은 배치 제외
│           {"linked_memory": {"$eq": ""}},                    # cue entry 제외 (primary만)
│           {"memory_type": {"$eq": "factual"}},               # factual만
│       ]}
│       query_results = agent_memory.query(entry.index, top_k=5, where=where,
│                                          query_mode=PRIMARY_ONLY,
│                                          enhance_query=False, return_history=False)
│       update_candidates = [c for c in query_results if c.score >= UPDATE_SCORE_THRESHOLD]
│
├─ (3a) candidates 없음 → agent_memory.add(entry)  (신규)
│        build_stats["new"] += 1
│
└─ (3b) candidates 있음 → _decide_memory_update(entry, candidates)
         LLM 호출 (PROMPT_MEMORY_UPDATE_DECISION) → MemoryUpdateDecision
         should_update=True  → update_memory(entry, decision)
         should_update=False → agent_memory.add(entry)  (신규)
```

#### 프롬프트 — `PROMPT_MEMORY_UPDATE_DECISION`

```text
You are a memory management assistant. Given a new memory entry and similar existing entries, determine whether to update an existing entry or add a new one.

NEW MEMORY ENTRY:
Index: {new_index}
Value: {new_value}

EXISTING SIMILAR ENTRIES:
{candidates_info}

INSTRUCTIONS:
1. Analyze if the new entry should update any existing entry based on semantic similarity and content overlap
2. If update is needed, determine which candidate is best to update
3. Generate the updated value that combines relevant information from both entries
4. Decide if the memory index should be updated to better reflect the combined information
```

#### 구조화 응답 — `MemoryUpdateDecision` (`utils/memory.py:12`)

```python
class MemoryUpdateDecision(BaseModel):
    should_update: bool
    best_candidate_index: Optional[int]     # candidates 리스트 중 index
    updated_value: Optional[str]            # 병합된 value
    updated_index: Optional[str]            # 갱신된 primary index (변경 가능)
    updated_cues: List[str] = []            # 갱신된 cue indices
```

#### 갱신 실행 — `update_memory()` (`memory_builder.py:494`)

```python
def update_memory(self, entry, update_decision):
    best_candidate = update_decision["best_candidate"]
    updated_value = update_decision["updated_value"]
    updated_index = update_decision["updated_index"]

    # updated_index 충돌 검사
    existing_entry = self.agent_memory.get(updated_index)
    if existing_entry and updated_index != best_candidate.index:
        updated_index = f"{updated_index}. (Added on {get_current_timestamp()}]"

    # 기존 메모리 삭제
    self.agent_memory.delete(best_candidate.index)

    # cue 재생성 (config 활성화 시)
    updated_cue_indices = []
    if self.cfg.memory.enable_cue_index:
        updated_cue_indices = self.cue_index_generator.generate_cue_indices(
            memory_value=updated_value, primary_index=updated_index)

    # history 누적 (이전 버전 추적)
    history = generate_history(entry, best_candidate)
    # [{"index":기존, "value":기존, "creation_time":..., "timestamp":...},
    #  {"index":신규, "value":신규, ...}]

    # image_urls, episodic_memory_ids 병합 (양쪽에서 합집합, 중복 제거)
    updated_image_urls = list(set((best_candidate.image_urls or []) + (entry.image_urls or [])))
    updated_episodic_memory_ids = list(set((best_candidate.episodic_memory_ids or []) + (entry.episodic_memory_ids or [])))

    # 신규 entry 생성·저장
    new_memory_entry = MemoryEntry(
        memory_type=best_candidate.memory_type,
        index=updated_index,
        value=updated_value,
        creation_time=entry.creation_time,
        timestamp=entry.timestamp,
        cue_indices="||".join(updated_cue_indices),
        history=history,
        image_urls=updated_image_urls,
        episodic_memory_ids=updated_episodic_memory_ids,
    )
    self.agent_memory.add(new_memory_entry)
```

---

## 3. `AgentMemory.add()` — ChromaDB 저장 (`core/memory.py:583`)

```python
def add(self, entry: MemoryEntry):
    assert entry.is_primary_index(), "Only primary memory entries can be added directly."
    # 중복 index 검사
    exist_entry = self._store.get(entry.index)
    if exist_entry is not None:
        if entry.memory_type == "episodic":
            # episodic은 sequential number 추가 ("...", "(2)", "(3)")
            original_index = entry.index
            counter = 2
            while self._store.get(f"{original_index} ({counter})") is not None:
                counter += 1
            entry.index = f"{original_index} ({counter})"
        else:
            raise AssertionError(f"Memory entry {entry.index} already exists.")

    # primary memory 저장 (index를 ChromaDB document로 embedding)
    self._store.upsert(index=entry.index, value=entry.value, metadata=entry.get_metadata())

    # cue index entries 저장 (value="", linked_memory로 primary 참조)
    for cue_index in entry.get_cue_indices():
        cue_entry = self._store.get(cue_index)
        # cue가 이미 primary index면 skip (충돌 방지)
        if (cue_entry and cue_entry.is_primary_index()) or cue_index == entry.index:
            entry.delete_cue_index(cue_index)
            continue
        # cue가 이미 존재하면 linked_memory 결합 (다대다)
        linked_memory = entry.index
        if cue_entry and cue_entry.is_cue_index():
            linked_memory = combine_list(linked_memory, cue_entry.linked_memory)  # "||" 결합
        self._store.upsert(index=cue_index, value="", metadata={"linked_memory": linked_memory})
```

### `LocalMemoryStore.upsert()` (`core/local_memory_store.py:119`)

```python
def upsert(self, index, value="", metadata=None):
    with self._lock:  # user_id별 RLock
        rid = index_to_id(index)  # hashlib.sha256(index).hexdigest()
        meta = {"index": index, "value": value, **metadata}
        # embedding cache 확인 (LRU, max 300)
        cached_embedding = self._get_cached_embedding(index)
        if cached_embedding is not None:
            self.db_client.upsert(collection, ids=[rid], documents=[index],
                                  metadatas=[meta], embeddings=[cached_embedding])
        else:
            self.db_client.upsert(collection, ids=[rid], documents=[index], metadatas=[meta])
            # 새 embedding은 cache에 저장
```

**핵심**: ChromaDB의 `documents` 필드에 `index`(primary abstraction 또는 cue text)를 넣어 embedding. `value`는 metadata에만 저장 (인덱싱 안 됨).

### MemoryEntry 메타데이터 구조 (`get_metadata()`)

```python
{
    "index": self.index,
    "history": json.dumps(self.history),          # list를 JSON string으로
    "timestamp": self.timestamp,
    "creation_time": self.creation_time,
    "linked_memory": self.linked_memory,          # "" (primary) 또는 "idx1||idx2" (cue)
    "cue_indices": self.cue_indices,              # "cue1||cue2" (primary만)
    "image_urls": json.dumps(self.image_urls),
    "memory_type": self.memory_type,              # "factual"/"episodic"/"procedural"
    "episodic_memory_ids": json.dumps(self.episodic_memory_ids),
    "original_text": self.original_text,          # episodic만
}
```

---

## 4. 데이터 구조 — `MemoryEntry` (`core/memory_entry.py:28`)

```python
class MemoryEntry(BaseModel):
    value: str                              # 메모리 본문 (비인덱싱)
    original_text: Optional[str] = ""        # episodic용 원본
    index: Optional[str] = ""               # primary abstraction (인덱싱)
    history: Optional[List[Dict]] = []      # 갱신 이력
    memory_type: Optional[str] = ""          # factual/procedural/episodic
    episodic_memory_ids: Optional[List[str]] = []  # 연결된 episodic
    score: Optional[float] = 0.0            # 검색 유사도
    timestamp: Optional[str] = ""           # 이벤트 발생 시각
    query: Optional[str] = ""               # 검색 query
    creation_time: Optional[str] = ""       # 저장 시각
    linked_memory: Optional[str] = ""       # "" (primary) 또는 "idx1||idx2" (cue)
    cue_indices: Optional[str] = ""         # "cue1||cue2" (primary만)
    image_urls: Optional[List[str]] = []

    def is_cue_index(self) -> bool:     return self.linked_memory != ""
    def is_primary_index(self) -> bool: return not self.linked_memory
    def get_cue_indices(self) -> List[str]:
        return [c.strip() for c in self.cue_indices.split("||") if c.strip()] if self.cue_indices else []
    def get_linked_memories(self) -> List[str]:
        return [l.strip() for l in self.linked_memory.split("||") if l.strip()] if self.linked_memory else []
    def get_memory_value(self, return_history=False, use_original_text=False) -> str:
        if use_original_text and self.original_text: return self.original_text
        if not self.history or not return_history: return self.value
        return "\n".join([f"[{h['timestamp']}] {h['value']}" for h in self.history])
```

**3요소 분리 (harmonic representation)**:
- `value` (비인덱싱): ChromaDB metadata에만 저장, embedding 안 됨
- `index` (인덱싱): ChromaDB document로 embedding → 검색 대상
- `cue_indices` (인덱싱): 별도 cue entry로 저장, `linked_memory`로 primary 참조 (다대다)
