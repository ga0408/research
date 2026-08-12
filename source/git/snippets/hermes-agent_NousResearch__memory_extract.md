# memory_extract — Hermes 메모리 추출(쓰기) 로직

> 출처: [분석 문서](../../../report/[git]_hermes-agent_NousResearch.md) / submodule: `source/git/hermes-agent_NousResearch`

## 설명

Hermes의 메모리는 **3계층**으로 분리되어 있고, 각각 추출(쓰기) 경로가 다르다.

1. **builtin curated memory** — `MEMORY.md`(에이전트 개인 메모) / `USER.md`(사용자 프로파일). 파일 기반, `§` 구분자, 문자수 제한(2200/1375). 에이전트가 `memory` 툴로 직접 add/replace/remove.
2. **external memory provider** — Honcho/Mem0/Hindsight 등(한 번에 1개만). `MemoryManager.sync_all()`이 매 턴 끝에 백그라운드로 (user, assistant) 쌍을 넘김 → provider가 자체 임베딩/저장.
3. **session DB (SQLite FTS5)** — `hermes_state.py`가 매 메시지를 `messages` 테이블 + `messages_fts` 가상테이블에 트리거로 자동 색인. 별도 추출 없이 전체 대화가 곧 검색 대상.

핵심 설계 원칙: **프롬프트 캐시 불변성**. builtin 메모리는 세션 시작 시 **frozen snapshot**을 시스템 프롬프트에 주입하고, 세션 중 쓰기는 디스크에만 반영(시스템 프롬프트는 건드리지 않음). 스냅샷은 다음 세션 시작 시 갱신.

### builtin MemoryStore — add / replace / remove / batch

단일 `memory` 툴이 `action` 파라미터로 분기. `replace`/`remove`는 부분 문자열 매칭(전체 텍스트/ID 아님). `operations` 배열은 **최종 문자수 예산에 대해서만** 원자적 all-or-nothing 적용 → 한 번에 공간 확보+추가 가능.

```python
# tools/memory_tool.py — MemoryStore
def add(self, target: str, content: str) -> Dict[str, Any]:
    content = content.strip()
    # 주입/유출 패턴 스캔 (threat_patterns, strict scope)
    scan_error = _scan_memory_content(content)
    if scan_error:
        return {"success": False, "error": scan_error}
    with self._file_lock(self._path_for(target)):   # 별도 .lock 파일 + fcntl/msvcrt
        self._reload_target(target, skip_drift=True)  # 디스크 최신 상태 재읽기(다른 세션 쓰기 반영)
        entries = self._entries_for(target)
        limit = self._char_limit(target)
        if content in entries:
            return self._success_response(target, "Entry already exists (no duplicate added).")
        new_total = len(ENTRY_DELIMITER.join(entries + [content]))
        if new_total > limit:                        # 용량 초과 → consolidation 유도(재시도 안내)
            return self._consolidation_failure({...,"current_entries": entries})
        entries.append(content)
        self._set_entries(target, entries)
        self.save_to_disk(target)                    # atomic temp+rename (os.replace)
    return self._success_response(target, "Entry added.")

ENTRY_DELIMITER = "\n§\n"
# memory_char_limit=2200, user_char_limit=1375 (config: memory.*_char_limit)
```

```python
# batch — 최종 상태에 대해서만 예산 검사, 중간 오버플로 무시
def apply_batch(self, target, operations):
    # 모든 op 위협 스캔 → 하나라도 걸리면 전체 거부
    with self._file_lock(...):
        bak = self._reload_target(target)            # 외부 drift 감지(라운드트립 불일치/단일 항목 초과)
        if bak: return _drift_error(...)              # patch tool/쉘 append/수동 편집 내용 보존
        working = list(self._entries_for(target))
        for i, op in enumerate(operations):           # add/replace/remove 각각 부분문자열 매칭
            ...                                       # 다중 매칭 시 거부(더 구체적으로 하라)
        new_total = len(ENTRY_DELIMITER.join(working))
        if new_total > limit: return self._consolidation_failure({...,"current_entries":...})
        self._set_entries(target, working); self.save_to_disk(target)  # 전체 커밋
    return self._success_response(target, f"Applied {len(operations)} operation(s).")
```

### 턴당 실패 캡 + consolidation 루프 방지

용량 초과/매칭 실패가 한 턴에 `_MAX_CONSOLIDATION_FAILURES_PER_TURN=3`회 반복되면, 재시도 안내를 버리고 **terminal "save skipped"** 반환 → 메모리 사이드이펙트가 턴의 응답을 블록하지 않게 함(issue #42405).

```python
_MAX_CONSOLIDATION_FAILURES_PER_TURN = 3
def _consolidation_failure(self, response):
    self._consolidation_failures += 1
    if self._consolidation_failures <= 3:
        return response                         # 모델에게 자가수정+재시도 안내
    return {"success": False, "done": True,
            "error": "Stop retrying memory calls — leave memory unchanged..."}  # 터미널
```

### builtin ↔ 외부 provider 동기화 (두 경로, 단방향)

builtin `MEMORY.md`와 외부 provider는 **동기화가 아닌 단방향 통지** 관계. 역방향(외부→builtin)은 없다. 두 백엔드는 저장 대상 자체가 다르므로(MEMORY.md=큐레이션 핵심 사실 vs 외부=임베딩 기반 전체 대화), **목적이 다른 두 스트림**이 병렬로 흐른다.

```
경로 A — 사실 미러링 (MEMORY.md 쓰기를 외부에 통지)
  에이전트 memory 툴(add/replace/remove/batch)
    → MemoryStore.* → MEMORY.md/USER.md 디스크 쓰기 (성공+커밋 결과)
    → notify_memory_tool_write → on_memory_write(action,target,content,metadata)
       · builtin 자신은 SKIP(쓰기 근원)
       · 외부 provider 각자의 on_memory_write() 호출 → provider 자율 반영(미구현 시 no-op)

경로 B — 대화 인제스트 (매 턴 전체 대화를 외부에 전달, MEMORY.md와 무관)
  run_conversation 턴 종료
    → MemoryManager.sync_all(user, assistant, messages=전체대화)
       · 백그라운드 단일 워커(직렬, 턴 순서 보장)
       · provider.sync_turn() → provider가 자체 임베딩/저장
```

**경로 A — 미러링 통지** (`notify_memory_tool_write`). builtin `memory` 툴 호출 후 에이전트 루프가 이 메서드로 raw 결과+args를 넘기면, manager가 "통지할지·무엇을·어떤 메타데이터로"를 단일 지점에서 결정. 게이트 3단계:

```python
# agent/memory_manager.py
_MIRRORED_MEMORY_ACTIONS = {"add", "replace", "remove"}

@staticmethod
def _memory_tool_result_succeeded(result):              # 게이트 1: 실제 커밋된 쓰기만
    # JSON 파싱 → dict → success is True AND staged is not True
    # (승인 대기/staged 또는 실패/조회형 결과는 "did not land" → 외부에 안 알림)
    return result.get("success") is True and result.get("staged") is not True

def notify_memory_tool_write(self, tool_result, tool_args, *, build_metadata=None):
    if not self._memory_tool_result_succeeded(tool_result): return
    # batch면 operations 각 op로 확장, 단일이면 한 op로 래핑
    for op in raw_operations:
        if op.get("action") not in self._MIRRORED_MEMORY_ACTIONS: continue   # 게이트 2: 변이 액션만
        metadata = dict(build_metadata() or {})                                # 게이트 3: 프로바이넌스
        if old_text: metadata["old_text"] = str(old_text)
        self.on_memory_write(action, target, str(op.get("content") or ""), metadata=metadata)

def on_memory_write(self, action, target, content, metadata=None):
    for provider in self._providers:
        if provider.name == "builtin": continue            # 쓰기 근원은 SKIP
        provider.on_memory_write(action, target, content, metadata=...)   # provider 자율 구현(default no-op)
```

> `MemoryProvider.on_memory_write`는 ABC에서 optional hook(기본 no-op). provider가 override해야 자기 저장소에 반영 → "복사"가 아니라 "통지". Honcho 등은 사용자 선호 사실을 dialectic 모델에 통합하는 용도로 구현.

**경로 B — 대화 인제스트** (`sync_all`). 매 턴 종료 후 `(user, assistant)` 쌍 + 전체 `messages`를 provider에 전달. MEMORY.md와 무관 — provider가 전체 대화를 자체 임베딩/저장. 백그라운드 단일 워커(`DaemonThreadPoolExecutor max_workers=1`)로 직렬화 → 턴 N이 턴 N+1보다 먼저 랜딩(순서 보장). 인라인 실행 시 느린 provider가 턴을 막았던 문제(Hindsight 298s 블록) 회피.

```python
# agent/memory_manager.py
_SYNC_DRAIN_TIMEOUT_S = 5.0
def sync_all(self, user_content, assistant_content, *, session_id="", messages=None):
    clean_user_content = self._strip_skill_scaffolding(user_content)  # /skill 전개 프롬프트 제거
    if not clean_user_content: return
    def _run():
        for provider in self._providers:
            if messages is not None and self._provider_sync_accepts_messages(provider):
                provider.sync_turn(user_content, assistant_content, session_id=session_id, messages=messages)
            else:
                provider.sync_turn(user_content, assistant_content, session_id=session_id)
    self._submit_background(_run)   # 단일 워커, FIFO 직렬
```

> 두 경로는 용도 보완: A가 "핵심 사실을 외부에 보강", B가 "대화 흐름을 외부에 색인". 같은 데이터를 중복 저장하는 게 아니라, builtin의 큐레이션 결과만 외부에 통지해 외부 검색 정확도를 높이는 관계.

### 세션 종료 추출 + 압축 전 추출

`on_session_end(messages)`(실제 세션 경계—CLI /reset, 게이트웨이 만료)와 `on_pre_compress(messages)`(컨텍스트 압축 직전) 훅에서 provider가 요약/사실 추출(LLM 호출, 수초) 수행. `/new` 회전은 `commit_session_boundary_async()`로 end→switch를 **단일 직렬 태스크**로 묶어 extract가 switch보다 먼저 실행되도록 보장(#16454).

### 프로바이넌스 태깅

모든 쓰기에 `write_origin`(`assistant_tool`/`background_review`)·`execution_context`(`foreground`/`background_review`)·`session_id`·`parent_session_id`·`platform` 메타데이터 부여 (`build_memory_write_metadata`). 외부 provider가 포그라운드 쓰기와 자기개선 쓰기를 구분 가능.
