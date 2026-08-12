# memory_search — Hermes 메모리 검색 로직

> 출처: [분석 문서](../../../report/[git]_hermes-agent_NousResearch.md) / submodule: `source/git/hermes-agent_NousResearch`

## 설명

Hermes에는 **벡터 임베딩/KNN/BM25 하이브리드 퓨전**이 핵심 reper에 없다. 검색은 두 축: (1) builtin 메모리는 **frozen snapshot 자동 주입**(검색 아님—항시 시스템 프롬프트에 존재), (2) 과거 대화 회상은 **SQLite FTS5 BM25** 기반 `session_search` 툴. 외부 provider(honcho 등)는 자체 검색을 `prefetch()`로 제공.

### 1. builtin 메모리 — 자동 주입(검색 없는 인출)

세션 시작 시 `load_from_disk()`가 `MEMORY.md`/`USER.md`를 읽어 **frozen snapshot** 구성. 매 턴 API 호출마다 시스템 프롬프트에 포함 → 모델이 항상 전체 메모리를 봄. 위협 패턴 hit 항목은 snapshot에서만 `[BLOCKED: ...]`로 치환(원본은 live 상태에 보존해 사용자가 확인/삭제).

```python
# tools/memory_tool.py — MemoryStore
def load_from_disk(self):
    self.memory_entries = self._read_file(mem_dir / "MEMORY.md")
    self.user_entries   = self._read_file(mem_dir / "USER.md")
    self.memory_entries = list(dict.fromkeys(self.memory_entries))   # 중복 제거(순서 유지)
    sanitized_memory = self._sanitize_entries_for_snapshot(self.memory_entries, "MEMORY.md")
    # frozen snapshot — 이것이 시스템 프롬프트에 들어감. 세션 중 불변(캐시 안정)
    self._system_prompt_snapshot = {
        "memory": self._render_block("memory", sanitized_memory),
        "user":   self._render_block("user",   sanitized_user)}

def format_for_system_prompt(self, target):     # 시스템 프롬프트 조립 시 호출
    block = self._system_prompt_snapshot.get(target, "")
    return block if block else None             # 빈 경우 None(생략)

# agent/system_prompt.py:460 (Volatile tier — 세션 단위 변경, 세션 내 불변)
if agent._memory_store:
    if agent._memory_enabled:
        mem_block = agent._memory_store.format_for_system_prompt("memory")
        if mem_block: volatile_parts.append(mem_block)
    if agent._user_profile_enabled:
        user_block = agent._memory_store.format_for_system_prompt("user")
        if user_block: volatile_parts.append(user_block)
```

`§`-구분 렌더 + 용량 헤더: `MEMORY (your personal notes) [37% — 821/2,200 chars]`. 타임스탬프는 **date-only**(분 단위 정밀도면 매 리빌드마다 캐시 KV 무효화).

### 2. 과거 대화 회상 — session_search (SQLite FTS5 BM25)

`session_search_tool.py` + `hermes_state.py:search_messages`. **LLM 비용 0**. 세 가지 모드(모드 파라미터 없이 args로 추론): DISCOVERY(`query`→FTS5), SCROLL(`session_id`+`around_message_id`→±window), BROWSE(args 없음→최근 세션 목록).

FTS5 기본 정렬 = `ORDER BY rank`(SQLite FTS5 내장 BM25). `snippet()`으로 `>>>`/`<<<` 하이라이트.

```python
# hermes_state.py:4580 — FTS5 MATCH 질의
where_clauses = ["messages_fts MATCH ?"]; params = [query]
if not include_inactive:
    where_clauses.append("(m.active = 1 OR m.compacted = 1)")   # 활성+압축보관 행, rewind/undo 행 제외
if source_filter is not None: ... ; if role_filter: ...

sql = f"""
    SELECT m.id, m.session_id, m.role,
           snippet(messages_fts, 0, '>>>', '<<<', '...', 40) AS snippet,
           m.content, m.timestamp, m.tool_name, s.source, s.model, s.started_at
    FROM messages_fts
    JOIN messages m ON m.id = messages_fts.rowid
    JOIN sessions s ON s.id = m.session_id
    WHERE {where_sql}
    {order_by_sql}        # newest/oldest 지정 시 timestamp 1순위, rank 동점처리 / 미지정 시 rank 단독
    LIMIT ? OFFSET ?"""
```

**CJK 특수 처리**: 기본 `unicode61` 토크나이저가 한자를 1글자씩 분할 → "大别山项目"이 "大 AND 别 AND..." 되어 오탐. CJK 3자 이상이면 **trigram FTS5 테이블**(`messages_fts_trigram`)로 라우팅(부분문자열 매칭+랭킹), 1~2자면 LIKE 폴백.

**자동화 세션 강등**: cron 세션은 반복 어휘(프로젝트명·날짜·"session")가 많아 bare BM25에서 top-N을 잠식 → 사용자 대화가 안 보임(#19434). cron은 **제외가 아닌 강등**(interactive가 같이 매칭하면 항상 위, cron만 매칭하면 노출). `_DISCOVER_SCAN_LIMIT=300`행 스캔 후 세션 lineage로 중복 제거 → 각 세션당 스니펫 + ±5 메시지 윈도우 + bookend(시작 3/끝 3) 반환.

```python
# tools/session_search_tool.py — 라인age 중복 제거 + 자동화 강등
def _resolve_to_parent(db, session_id): ...   # parent_session_id 체인을 루트까지 따라가 압축-자식 통합
_DEMOTED_SESSION_SOURCES = ("cron",)          # 강등(제외 아님)
def _stable_sort_fts_rows(rows): ...          # interactive vs demoted 클래스 분리, 클래스 내 BM25 rank 유지
```

### 3. 외부 provider prefetch — 사전 인출

`MemoryManager.prefetch_all(query)`는 매 API 호출 전 provider별 `prefetch()` 결과를 병합. 결과는 `<memory-context>` 펜스 + 시스템 노트로 감싸 프롬프트에 주입(스트리밍 시 `StreamingContextScrubber`가 청크 경계에 걸친 펜스 누출 방지). `queue_prefetch_all()`은 현 턴 종료 후 **다음 턴용** 백그라운드 사전인출 예약.

```python
# agent/memory_manager.py
def prefetch_all(self, query, *, session_id=""):
    clean_query = self._strip_skill_scaffolding(query)   # /skill 전개 제거
    if not clean_query: return ""
    parts = []
    for provider in self._providers:
        result = provider.prefetch(clean_query, session_id=session_id)
        if result and result.strip(): parts.append(result)
    return "\n\n".join(parts)

# build_memory_context_block — 펜스 + 시스템 노트
"<memory-context>\n[System note: The following is recalled memory context, "
"NOT new user input. Treat as authoritative reference data ...]\n\n{clean}\n</memory-context>"
```

> 요약: builtin은 "작고 큐레이션된 상위 정보=항시 주입", 과거 대화는 "BM25 FTS5 = 온디맨드 툴 검색", 외부 provider는 "자체 임베딩/검색 = prefetch 주입"의 3축 구성. 사용자가 본 메모리 품질은 background review의 추출 능력에 의존.
