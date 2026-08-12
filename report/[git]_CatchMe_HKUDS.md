> [git] https://github.com/HKUDS/CatchMe.git

# CatchMe: Personal Digital Activity Memory

## Overview

CatchMe는 사용자의 모든 디지털 활동을 6개 recorder가 백그라운드에서 캡처하여, 룰 기반으로 5계층 activity tree(Day → Session → App → Location → Action)를 구성하고, LLM이 tree를 top-down 탐색하며 reasoning으로 관련 활동을 검색하는 시스템이다. 벡터 DB, 임베딩, 청킹 없이 PageIndex에서 영감을 받은 tree-based retrieval을 사용한다.

## 핵심 방법론 요약

### 인덱싱 (캡처 → 트리 빌드 → 요약)

**캡처**: 6개 recorder가 이벤트를 실시간으로 SQLite에 저장하고, 동시에 Organizer에게 boundary 이벤트를 전달
- window switch 또는 idle(5분+)이 발생하면 트리 빌드 트리거
- 트리 빌드는 LLM 없이 순수 룰 기반: window span(3초 이상) → session 분할(idle gap 5분) → app 그룹핑 → location(URL/file/title) 그룹핑 → action(interaction 3초 gap 클러스터링)
- 증분 확장: 매번 전체 재빌드가 아닌 마지막 session만 재빌드, 기존 summary는 node_id 매칭으로 보존

**요약**: 닫힌 노드(마지막 sibling이 아닌 노드)만 4계층으로 bottom-up 비동기 요약
- L0: mouse cluster → vision LLM (스크린샷 + 좌표로 마우스 행동 요약)
- L1: action → text LLM (키보드+마우스 요약+클립보드 타임라인)
- L2: location/app → text LLM (자식 action 요약 취합)
- L3: session → text LLM (자식 app 요약 취합)
- 자식 요약 완료시 부모로 cascade (우선순위 큐 + ThreadPoolExecutor)

### 검색 (tree-based reasoning retrieval)

**PageIndex에서 영감을 받은 tree traversal 검색.** 3단계 LLM 호출 반복:

1. **시간 pre-filtering**: LLM이 query에서 날짜/시간 추출 → 해당 날짜 tree만 로드 (선택적)
2. **Select**: LLM이 현재 레벨의 ToC(title + summary)를 읽고 관련 노드 선택
3. **Evaluate**: 선택된 노드의 상세 내용(summary + evidence + children)을 읽고 useful 정보 추출
4. **Deeper**: action 노드까지 내려가면 raw 이벤트(키보드 텍스트, 마우스 클러스터 요약, 스크린샷, 파일 내용, URL fetch)로 확장하여 검사
5. **Answer**: 충분한 정보가 모이면 collected context로 최종 답변 생성

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      CAPTURE (실시간 이벤트 수집)                     │
│                                                                     │
│  6 Recorders (daemon threads)                                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐   │
│  │ Window   │ │ Keyboard │ │ Mouse    │ │Clipboard │ │ Idle   │   │
│  │ (focus)  │ │ (keys)   │ │(click+ss)│ │ (copy)   │ │(5min+) │   │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └───┬────┘   │
│       │            │            │            │            │        │
│       └──────┬─────┴──────┬─────┘            │            │        │
│              │            │                  │            │        │
│              ▼            ▼                  │            │        │
│        emit(data, blob)  ←── Engine._make_emitter(kind)   │        │
│              │                                               │        │
│       ┌──────┴───────────────────────────────────┐        │        │
│       │  Queue[Event]                             │        │        │
│       └──────┬──────────────────┬────────────────┘        │        │
│              │                  │                         │        │
│              ▼                  ▼                         │        │
│     ┌─────────────┐    ┌─────────────────┐               │        │
│     │ _write_loop │    │ Organizer       │               │        │
│     │ batch write │    │ .on_event()     │               │        │
│     │ → SQLite    │    │  window switch? │               │        │
│     │  + FTS5     │    │  idle/locked?   │               │        │
│     └─────────────┘    │  → _pending.set │               │        │
│                          └───────┬─────────┘               │        │
│                                  ▼                          │        │
│                          ┌──────────────────┐             │        │
│                          │ _process()       │             │        │
│                          │ (3초 debounce)   │             │        │
│                          └───────┬──────────┘             │        │
│                                  │                          │        │
│ ┌────────────────────────────────┼────────────────────────┼────────┘
│ │              INDEX (트리 빌드)  │                        │
│ │                                ▼                        │
│ │  ┌─────────────────────────────────────────────────┐   │
│ │  │  build_tree() / extend_tree()  (순수 룰 기반)    │   │
│ │  │                                                  │   │
│ │  │  window events → WindowSpan (3초+ 필터, 병합)    │   │
│ │  │       │                                          │   │
│ │  │       ▼                                          │   │
│ │  │  Day ─┬─ Session (idle gap 5분 / 시간 gap 5분)   │   │
│ │  │       │   └─ App (app 이름 그룹핑)               │   │
│ │  │       │       └─ Location (URL / file / title)   │   │
│ │  │       │           └─ Action (3초 gap 클러스터)   │   │
│ │  │       │               title: "typing: hello"    │   │
│ │  │       │                      "click × 3"         │   │
│ │  │       │                      "copy-paste"        │   │
│ │  │       │                      "scroll"            │   │
│ │  │  extend: 마지막 session만 재빌드                 │   │
│ │  │         기존 summary → node_id 매칭 보존         │   │
│ │  └──────────────────────┬──────────────────────────┘   │
│ │                         ▼                               │
│ │  ┌──────────────────────────────────────────────────┐  │
│ │  │  _enqueue_closed_nodes()                          │  │
│ │  │  닫힌 노드(마지막 sibling 제외) → SummaryQueue     │  │
│ │  └──────────────────────┬───────────────────────────┘  │
│ │                         ▼                               │
│ │  ┌──────────────────────────────────────────────────┐  │
│ │  │  SummaryQueue (PriorityQueue + ThreadPool)        │  │
│ │  │                                                   │  │
│ │  │  L0  mouse cluster → vision LLM (스크린샷+좌표)   │  │
│ │  │  L1  action       → text LLM (타임라인 요약)      │  │
│ │  │  L2  location/app → text LLM (자식 취합)          │  │
│ │  │  L3  session      → text LLM (자식 취합)          │  │
│ │  │                                                   │  │
│ │  │  자식 완료 → 부모 cascade (bottom-up)             │  │
│ │  └──────────────────────────────────────────────────┘  │
│ │                         │                               │
│ │                  save_tree() → JSON                     │
│ └─────────────────────────┼───────────────────────────────┘
│                           │
└───────────────────────────┼───────────────────────────────┘
                            │
┌───────────────────────────┼───────────────────────────────┐
│                     RETRIEVE (검색)                        │
│                           │                                │
│                   _load_all_trees()                        │
│                   모든 날짜 JSON 로드                       │
│                           │                                │
│                           ▼                                │
│  ┌────────────────────────────────────────────────────┐   │
│  │  1. Time Pre-filter (LLM)                          │   │
│  │     query → 날짜/시간 추출 → 해당 tree만 선별       │   │
│  └──────────────────────┬─────────────────────────────┘   │
│                         ▼                                  │
│  ┌────────────────────────────────────────────────────┐   │
│  │  2. Main Loop (max 15 iterations)                  │   │
│  │                                                    │   │
│  │  ┌─ Select (LLM) ──────────────────────────────┐   │   │
│  │  │  현재 frontier의 ToC(title+summary)를 LLM에  │   │   │
│  │  │  보내고 관련 노드 선택 (max 7개)             │   │   │
│  │  └──────────────────┬──────────────────────────┘   │   │
│  │                     ▼                               │   │
│  │  ┌─ Evaluate (LLM) ────────────────────────────┐   │   │
│  │  │  선택된 노드의 summary+evidence+children 읽기│   │   │
│  │  │  → useful 추출 → collected에 추가            │   │   │
│  │  └──────────────────┬──────────────────────────┘   │   │
│  │                     │                               │   │
│  │           ┌─────────┼──────────┐                   │   │
│  │           ▼         ▼          ▼                    │   │
│  │       answer    deeper    siblings                  │   │
│  │       (종료)   (자식으로)  (같은 레벨)               │   │
│  │                    │                                │   │
│  │  deeper:                                           │   │
│  │    action → raw_keyboard / raw_mouse (스크린샷)    │   │
│  │    location → children + file/URL 직접 읽기        │   │
│  │    raw_mouse → vision LLM (스크린샷 검사)          │   │
│  │    raw_file → read_file_content                    │   │
│  │    raw_url → fetch_url_content                     │   │
│  └────────────────────────────────────────────────────┘   │
│                         ▼                                  │
│  ┌────────────────────────────────────────────────────┐   │
│  │  3. Answer (LLM)                                   │   │
│  │     collected context로 최종 답변 생성              │   │
│  └────────────────────────────────────────────────────┘   │
│                                                            │
│  ※ 벡터 검색 없음. LLM reasoning = 검색 엔진              │
└────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      STORAGE (저장)                          │
│                                                             │
│  SQLite (events_raw 테이블)                                 │
│    ├── 단일 테이블: ts, kind, data(JSON), blob, processed   │
│    ├── FTS5 가상 테이블 (data 컬럼 full-text search)        │
│    ├── WAL 모드 (동시 읽기/쓰기)                             │
│    └── trigger: insert/delete시 FTS5 자동 동기화            │
│                                                             │
│  Tree JSON ({date}_time.json)                               │
│    ├── 트리 전체 구조 + 각 노드의 summary/evidence           │
│    ├── atomic write (.tmp → os.replace)                     │
│    └── save_interval(5초)마다 주기적 저장                    │
│                                                             │
│  Blobs (스크린샷 이미지)                                     │
│    └── ~/data/blobs/ 하위 디렉토리                           │
│                                                             │
│  summary_updates.jsonl                                      │
│    └── 요약 완료 알림 (web SSE용)                           │
│                                                             │
│  llm_usage.json                                             │
│    └── LLM 토큰 사용량 추적 (prompt/completion)             │
└─────────────────────────────────────────────────────────────┘
```

## 메모리 추출 및 저장 과정 상세 분석

### 전체 흐름

```
┌─────────────────────────────────────────────────────────────────────────┐
│                 MEMORY EXTRACTION & STORAGE PIPELINE                     │
│                                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐   │
│  │ Window   │  │ Keyboard │  │ Mouse    │  │Clipboard │  │ Idle   │   │
│  │ Recorder │  │ Recorder │  │Recorder  │  │Recorder  │  │Recorder│   │
│  │ (focus)  │  │ (keys)   │  │(click+ss)│  │ (copy)   │  │(5min+) │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └───┬────┘   │
│       │             │             │             │             │        │
│       └─────────────┴──────┬──────┴─────────────┘             │        │
│                            │                                  │        │
│                   emit(data, blob)                            │        │
│                   (Engine._make_emitter)                      │        │
│                            │                                  │        │
│              ┌─────────────┴─────────────┐                   │        │
│              ▼                           ▼                    │        │
│      ┌─────────────┐           ┌──────────────────┐          │        │
│      │  Queue      │           │  Organizer       │          │        │
│      │  [Event]    │           │  .on_event()     │          │        │
│      └──────┬──────┘           │                  │          │        │
│             │                  │  window switch?  │          │        │
│      ┌──────▼──────┐           │  idle/locked?    │          │        │
│      │ _write_loop │           │  → _pending.set()│          │        │
│      │ batch write │           └────────┬─────────┘          │        │
│      └──────┬──────┘                    │                    │        │
│             │                           ▼                    │        │
│      ┌──────▼──────┐           ┌──────────────────┐          │        │
│      │   SQLite    │           │  _process()      │          │        │
│      │ events_raw  │           │  (3초 debounce)  │          │        │
│      │ + FTS5 index│           └────────┬─────────┘          │        │
│      └─────────────┘                    │                    │        │
│                                ┌───────▼────────┐            │        │
│                                │ build_tree()   │            │        │
│                                │ / extend_tree()│            │        │
│                                │ (룰 기반, LLM X)│           │        │
│                                └───────┬────────┘            │        │
│                                        │                      │        │
│                          ┌─────────────▼──────────────┐      │        │
│                          │ _enqueue_closed_nodes()    │      │        │
│                          │ 닫힌 노드만 SummaryQueue로  │      │        │
│                          └─────────────┬──────────────┘      │        │
│                                        │                      │        │
│                          ┌─────────────▼──────────────┐      │        │
│                          │   SummaryQueue             │      │        │
│                          │   (PriorityQueue+ThreadPool)│      │        │
│                          │                            │      │        │
│                          │ L0: mouse → vision LLM     │      │        │
│                          │ L1: action → text LLM      │      │        │
│                          │ L2: loc/app → text LLM     │      │        │
│                          │ L3: session → text LLM     │      │        │
│                          │ 자식 완료 → 부모 cascade   │      │        │
│                          └─────────────┬──────────────┘      │        │
│                                        │                      │        │
│                          ┌─────────────▼──────────────┐      │        │
│                          │ save_tree()                │      │        │
│                          │ {date}_time.json           │      │        │
│                          │ (atomic write)             │      │        │
│                          └────────────────────────────┘      │        │
│                                                             │        │
└─────────────────────────────────────────────────────────────┼────────┘
```

### Step 1: 이벤트 캡처 (6개 Recorder)

6개 recorder가 각각 daemon thread로 동작하며, 이벤트 발생시 `emit(data, blob)` 콜백을 호출한다:

- **WindowRecorder**: 윈도우 포커스 변경시 `emit({app, title, url, filepath})`. URL/filepath는 활성 탭/파일에서 추출
- **KeyboardRecorder**: 키 입력시 `emit({type, key})`. type은 `text`/`special`/`shortcut`. IME 조합중에는 zero-width space(`\u200b`) 삽입
- **MouseRecorder**: 마우스 클릭/스크롤시 `emit({action, x, y, button, display}, blob=스크린샷경로)`. `scroll_start`/`scroll_end`로 스크롤 세션 추적. `detail` 필드로 크롭 이미지 경로
- **ClipboardRecorder**: 클립보드 변경시 `emit({type, content})`. type은 `text`/`image`
- **IdleRecorder**: idle/locked 상태 변화시 `emit({status, start, end})`. status는 `idle`/`active`/`locked`
- **NotificationRecorder** (macOS only): 알림 수신시 `emit({app, title, body})`

### Step 2: 이벤트 저장 (Queue → SQLite)

`Engine._make_emitter()` (`engine.py:78`)가 각 recorder의 emit 콜백을 생성한다. 이벤트는 두 경로로 동시 분기된다:

```python
def emit(data: dict, blob: str = "") -> None:
    if self._paused:
        return
    event = Event(timestamp=time.time(), kind=kind, data=data, blob=blob)
    self._queue.put(event)           # 경로1: SQLite 저장용
    self._organizer.on_event(event)  # 경로2: 트리 빌드 트리거
```

**경로1 — batch write**: `_write_loop()` (`engine.py:93`)가 `batch_size`개 이벤트 또는 `batch_timeout`초를 기준으로 batch를 모아 `Store.insert_raw()`로 SQLite에 일괄 삽입한다. 이는 쓰기 빈도를 낮춰 I/O 부담을 줄인다.

**SQLite 스키마** (`store.py:21`): 단일 `events_raw` 테이블에 모든 종류의 이벤트를 저장한다:

```sql
CREATE TABLE events_raw (
    id        INTEGER PRIMARY KEY,
    ts        REAL NOT NULL,        -- Unix timestamp
    kind      TEXT NOT NULL,        -- window/keyboard/mouse/clipboard/idle/notification
    data      TEXT NOT NULL,        -- JSON (이벤트 상세 데이터)
    blob      TEXT DEFAULT '',      -- 스크린샷 파일 경로 등
    processed INTEGER DEFAULT 0     -- 처리 여부 (현재 미사용)
);
```

**FTS5 full-text index**: `data` 컬럼에 FTS5 가상 테이블을 구성한다. trigger가 insert/delete시 자동 동기화한다:

```sql
CREATE VIRTUAL TABLE events_raw_fts USING fts5(data, content=events_raw, content_rowid=id);
CREATE TRIGGER events_raw_ai AFTER INSERT ON events_raw BEGIN
    INSERT INTO events_raw_fts(rowid, data) VALUES (new.id, new.data);
END;
```

- **WAL 모드** (`PRAGMA journal_mode=WAL`): 동시 읽기/쓰기 지원. `catchme awake` (쓰기 프로세스)와 `catchme web` (읽기 프로세스)가 동시 실행 가능
- **thread-local connection**: `_conn` 프로퍼티가 스레드별 독립 SQLite connection을 지연 생성한다. batch write 스레드와 다른 스레드가 충돌하지 않음

**FTS5 키워드 검색** (`store.py:111`): `Store.search(text, kind, since, until)`가 full-text search로 빠른 키워드 검색을 제공한다. tree retrieval과 별개로 작동하며, `CatchMe.search("meeting notes")` 형태로 CLI/API에서 사용 가능하다.

### Step 3: Boundary 감지 → 트리 빌드 트리거

`Organizer.on_event()` (`organizer.py:51`)는 non-blocking으로 호출되며, boundary 이벤트를 감지한다:

```python
def on_event(self, event: Event) -> None:
    if event.kind == "window":
        key = (event.data.get("app"), event.data.get("title"))
        if key != self._last_window_key:   # 창이 바뀌면
            self._last_window_key = key
            self._pending.set()            # _process() 트리거
    elif event.kind == "idle":
        if event.data.get("status") in ("idle", "locked"):
            self._pending.set()
```

`Organizer.run()` (`organizer.py:63`)이 daemon thread에서 `_pending.wait(timeout=300초)`로 대기하다가, trigger가 발생하면 `_process()`를 호출한다. `_FALLBACK_INTERVAL`(5분) 동안 이벤트가 없어도 폴링으로 처리한다.

### Step 4: 룰 기반 트리 구성 (LLM 없음)

`_process()` (`organizer.py:83`)가 호출하는 `build_tree()` / `extend_tree()`는 순수 룰 기반이다.

**최초 빌드** — `build_tree()` (`tree.py:80`):

1. **이벤트 조회**: Store에서 6종 이벤트(window, keyboard, mouse, clipboard, idle)를 시간 범위로 조회 (limit 50,000)
2. **WindowSpan 생성** — `build_window_spans()` (`filter.py:57`):
   - 3초 미만 체류(`window_min_dwell`) 필터. 단 마지막 이벤트(현재 활성 창)는 무조건 포함
   - 동일 (app, title)의 연속 span은 병합 (예: A(60s) → B(1s) → A(2s) → B의 두 A가 병합)
   - 5분(`session_gap`) 이상 span은 캡핑 (idle/sleep 구분용)
   - 필터된 brief window는 부모 span의 `briefs` 리스트에 첨부
3. **Session 분할** — `_split_sessions()` (`tree.py:591`):
   - idle 이벤트에서 5분 이상 idle/locked 구간 추출
   - span 사이의 시간 gap이 5분 이상이거나 idle 구간에 포함되면 session 경계
4. **App 그룹핑** — `_build_app_location_children()` (`tree.py:448`):
   - session 내에서 같은 app 이름의 span을 OrderedDict로 그룹핑 (순서 유지)
5. **Location 그룹핑**:
   - 같은 app 내에서 `url or filepath or title`을 location 키로 그룹핑
   - location 노드의 title은 60자로 truncate, full 경로는 `context.full_location`에 저장
6. **Action 클러스터링** — `_collect_actions()` (`tree.py:412`):
   - location 범위 내의 keyboard + mouse + clipboard 이벤트를 시간순 병합
   - `cluster_events(merged, action_gap=3초)`로 3초 gap 클러스터링. 단, scroll 세션이 열려있으면 gap과 무관하게 같은 클러스터 유지
   - 각 클러스터(Action)의 title은 `_derive_action_title()`이 이벤트 종류로 자동 생성:
     - clipboard 있음 → `"copy-paste"`
     - keyboard text → `"typing: hello world"` (IME pinyin 중간 단계는 `_strip_ime_pinyin()`으로 제거)
     - scroll만 → `"scroll"`
     - click → `"click × 3"`
     - shortcut → `"shortcut: Ctrl+S"`
     - 기타 → `"interaction × 5"`
   - 각 Action의 `context`에 text, shortcuts, mouse_actions(좌표+스크린샷경로), clipboard preview 저장

**노드 ID 생성 규칙** (stable ID):
- Day: `d20260702`
- Session: `d20260702_s1751500000` (unix timestamp)
- App: `d20260702_s1751500000_cursor` (`_sanitize_app` — 소문자화, 특수문자→`_`, 24자 truncate)
- Location: `d20260702_s1751500000_cursor_a1b2c3d4` (`_hash_loc` — SHA1 8자)
- Action: `d20260702_s1751500000_cursor_a1b2c3d4_t1751500123` (unix timestamp)

### Step 5: 증분 트리 확장

`extend_tree()` (`tree.py:138`)는 마지막 session만 재빌드하여 전체 재빌드를 피한다:

1. 마지막 이벤트 timestamp 이후의 새 window event를 조회하여 `new_spans` 생성
2. 마지막 session의 `end`와 `new_spans[0].start` 사이의 gap 계산
3. **새 session 경계 감지** (gap ≥ 5분 또는 idle 구간):
   - 기존 마지막 session은 자동으로 닫힘
   - 새 session들을 `children.append()`로 추가
4. **경계 없음 (같은 session 연장)**:
   - 마지막 session의 시작 시점부터 전체 재빌드 (새 이벤트 포함)
   - `_apply_merge()`가 node_id 매칭으로 기존 summary/evidence/mouse_summaries를 새 노드로 복사
   - `children[-1]`을 새 session dict로 교체

### Step 6: 닫힌 노드 Enqueue

`_enqueue_closed_nodes()` (`organizer.py:145`)가 트리를 순회하며 닫힌 노드를 SummaryQueue에 enqueue한다:

- **닫힌 노드 = 마지막 sibling이 아닌 노드**: `is_last_sibling = (i == len(children) - 1)`. 마지막 sibling은 사용자가 현재 활동중인 노드이므로 아직 완성되지 않았다고 간주
- `node_id`를 `SummaryQueue._closed_set`에 추가
- 부모 노드를 `_parent_map`에 등록 (cascade용)
- summary가 없는 닫힌 노드만 `enqueue(node, level)` 호출. level은 `KIND_TO_LEVEL`: action=1, location=2, app=2, session=3

### Step 7: 비동기 계층적 요약

`SummaryQueue` (`summary_queue.py:52`)가 PriorityQueue + ThreadPoolExecutor로 4계층 요약을 비동기 처리한다:

```
                    SummaryQueue 처리 흐름

  enqueue(node, level)
         │
         ▼
  ┌─────────────────┐
  │ PriorityQueue   │  우선순위: (level, enqueue_time, node_id)
  │ (_QueueItem)    │  L1(action) > L2(location/app) > L3(session)
  └───────┬─────────┘
          │
  ┌───────▼─────────┐
  │ _dispatch_loop  │  daemon thread, timeout=1초 poll
  │ → pool.submit   │
  └───────┬─────────┘
          │
  ┌───────▼─────────────────────────────────────────┐
  │ _process_item                                   │
  │                                                 │
  │  1. _ready(node) 체크                           │
  │     - action: 항상 ready                        │
  │     - location/app/session: 자식 중 1개 이상     │
  │       summary 있어야 ready                      │
  │     - ready 안됨 → 1초 대기 후 재enqueue        │
  │       (최대 10회 retry)                          │
  │                                                 │
  │  2. summarize_node(node, llm) → LLM 호출        │
  │     ┌─────────────────────────────────────┐     │
  │     │ L0: mouse cluster → vision LLM      │     │
  │     │   스크린샷(base64) + 좌표 + action   │     │
  │     │   → summary + evidence              │     │
  │     │   max_images_per_cluster=5          │     │
  │     │   초과시 등거리 샘플링               │     │
  │     ├─────────────────────────────────────┤     │
  │     │ L1: action → text LLM               │     │
  │     │   _build_action_timeline:           │     │
  │     │   keyboard text + shortcuts +       │     │
  │     │   clipboard preview +               │     │
  │     │   mouse_summaries(L0 결과)          │     │
  │     │   → 시간순 타임라인 → summary        │     │
  │     ├─────────────────────────────────────┤     │
  │     │ L2: location/app → text LLM         │     │
  │     │   자식 action들의 summary +         │     │
  │     │   시간/이벤트수/dwell 취합           │     │
  │     │   → summary                         │     │
  │     ├─────────────────────────────────────┤     │
  │     │ L3: session → text LLM              │     │
  │     │   자식 app들의 summary +            │     │
  │     │   사용 앱 목록/dwell 취합            │     │
  │     │   → summary                         │     │
  │     └─────────────────────────────────────┘     │
  │                                                 │
  │  3. _parse_structured_summary(raw)              │
  │     LLM 출력에서 ## Summary / ## Evidence 분리  │
  │     → node["summary"], node["evidence"]          │
  │                                                 │
  │  4. _write_notification(node)                   │
  │     data/summary_updates.jsonl에 append          │
  │     (web SSE 스트리밍용)                         │
  │                                                 │
  │  5. _maybe_save()                               │
  │     save_interval(5초)마다 tree JSON 저장        │
  │                                                 │
  │  6. _cascade_parent(nid)                        │
  │     부모가 closed_set에 있고 summary 없으면      │
  │     → 부모 enqueue (bottom-up 전파)             │
  └─────────────────────────────────────────────────┘
```

**L0 Mouse Cluster Vision 요약** (`_summarize_mouse_clusters`, `summarize.py:368`):
- action 노드의 `context.mouse_actions`를 `mouse_cluster_gap`(3초)로 재클러스터링 (`_sub_cluster_mouse`)
- 각 클러스터에서 스크린샷이 있는 이벤트만 추출
- `max_images_per_cluster`(5) 초과시 등거리 샘플링
- 각 (action text + screenshot) 쌍을 interleaved multi-modal message로 vision LLM 호출
- 결과를 `mouse_summaries` 리스트에 `{start, end, summary, evidence}` 형태로 저장

**L1 Action 요약** (`_summarize_action`, `summarize.py:494`):
- `_build_action_timeline()`이 keyboard text, shortcuts, clipboard preview, mouse_summaries(L0 결과)를 시간순으로 interleave
- L0 결과가 없으면 mouse event 수만 표시
- header에 app, location, 시간, 이벤트 수 포함
- system prompt + timeline을 text LLM에 전달

**L2/L3는 자식 summary를 취합** — location/app은 자식 action summary를, session은 자식 app summary를 header와 함께 LLM에 전달. 자식에 summary가 하나도 없으면 `_ready()`가 false를 반환하여 retry 대기.

**Budget 제어** (`_CallBudget`, `llm.py:70`): `llm.max_calls` 설정으로 프로세스 전체 LLM 호출 수를 제한. 초과시 `LLMBudgetExhausted` 예외로 요약 중단.

### Step 8: 트리 영속화

`save_tree()` (`tree.py:310`)가 트리를 JSON에 atomic write한다:

```python
def save_tree(result: dict) -> str | None:
    tree = result.get("tree")
    mode = result.get("mode", "time")
    date = tree.get("title", "unknown")  # "2026-07-02"
    path = os.path.join(tree_dir, f"{date}_{mode}.json")  # ~/data/trees/2026-07-02_time.json
    tmp = path + ".tmp"
    meta = {"saved_at": time.time(), "mode": mode, "date": date, "tree": tree}
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())           # 디스크 동기화
    os.replace(tmp, path)              # 원자적 교체
    return path
```

- **저장 시점**: `_process()` 종료시 + `SummaryQueue._maybe_save()`가 `save_interval_sec`(5초)마다 호출 + `stop()`시 강제 저장
- **복원**: `load_tree(date, mode)`가 다음 실행시 디스크에서 트리를 복원하여 증분 확장의 기준점으로 사용

### 저장 매체 요약

| 매체 | 위치 | 용도 |
|------|------|------|
| SQLite (`events_raw` + FTS5) | `~/data/catchme.db` | 모든 raw 이벤트 (영구) |
| Tree JSON | `~/data/trees/{date}_time.json` | 트리 구조 + summary + evidence (날짜별) |
| Blobs | `~/data/blobs/` | 스크린샷 이미지 (full + detail crop) |
| summary_updates.jsonl | `~/data/summary_updates.jsonl` | 요약 완료 알림 (web SSE용, 최대 500라인) |
| llm_usage.json | `~/data/llm_usage.json` | LLM 토큰 사용량 (prompt/completion별 기록) |

## 메모리 검색 과정 상세 분석

### 전체 흐름

```
┌─────────────────────────────────────────────────────────────────────────┐
│                 MEMORY RETRIEVAL PIPELINE                                │
│                                                                         │
│  User Query: "What was I working on yesterday morning?"                 │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────┐                       │
│  │  Step 1: Load All Trees                     │                       │
│  │  _load_all_trees() → 모든 {date}_time.json  │                       │
│  │  디스크에서 로드, 날짜순 정렬                 │                       │
│  └──────────────────────┬──────────────────────┘                       │
│                         ▼                                               │
│  ┌─────────────────────────────────────────────┐                       │
│  │  Step 2: Time Pre-filter (LLM #1)           │                       │
│  │  _resolve_time_range(query)                  │                       │
│  │  LLM이 query에서 날짜/시간 추출              │                       │
│  │  → dates=["2026-07-01"], start_hour=9,      │                       │
│  │    end_hour=12                               │                       │
│  └──────────────────────┬──────────────────────┘                       │
│                         │                                               │
│            ┌────────────┴────────────┐                                  │
│            ▼                         ▼                                  │
│     dates 있음               dates 없음                                 │
│     trees 필터링             모든 trees 사용                            │
│            │                         │                                  │
│            └────────────┬────────────┘                                  │
│                         ▼                                               │
│  ┌─────────────────────────────────────────────┐                       │
│  │  Step 3: Build node_idx                     │                       │
│  │  모든 트리를 순회하며 node_id → node 맵 구축  │                       │
│  └──────────────────────┬──────────────────────┘                       │
│                         ▼                                               │
│  ┌─────────────────────────────────────────────┐                       │
│  │  Step 4: Day Selection                      │                       │
│  │  dates 있음 → 자동 선택 (LLM 호출 X)        │                       │
│  │  dates 없음 → LLM이 Day ToC 읽고 선택 (#2)  │                       │
│  └──────────────────────┬──────────────────────┘                       │
│                         ▼                                               │
│  ┌─────────────────────────────────────────────┐                       │
│  │  Step 5: Session Expansion                  │                       │
│  │  hours 있음 → _sessions_in_range() 필터링   │                       │
│  │  hours 없음 → 모든 session 자식 사용         │                       │
│  │  → frontier 구성                             │                       │
│  └──────────────────────┬──────────────────────┘                       │
│                         ▼                                               │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │  Step 6: Main Loop (max_iterations=15)                      │       │
│  │                                                             │       │
│  │  ┌─── Iteration ───────────────────────────────────────┐    │       │
│  │  │                                                     │    │       │
│  │  │  6a. SELECT (LLM #n+2)                              │    │       │
│  │  │      frontier 중 explored 아닌 노드들의 ToC 생성     │    │       │
│  │  │      ToC = "- [node_id] (kind) title\n  summary"   │    │       │
│  │  │      LLM에 query + collected + ToC 전달             │    │       │
│  │  │      → selected node_ids (max 7개)                  │    │       │
│  │  │      → action: "sufficient" | "deeper" | ...        │    │       │
│  │  │                                                     │    │       │
│  │  │      yield {type: "browse", level, candidates,      │    │       │
│  │  │             selected, reasoning}                    │    │       │
│  │  │                                                     │    │       │
│  │  │  6b. EVALUATE (LLM #n+3)                            │    │       │
│  │  │      선택된 노드의 details 생성                     │    │       │
│  │  │      details = summary + evidence + children ToC    │    │       │
│  │  │      LLM에 query + collected + details 전달         │    │       │
│  │  │      → useful: [{node_id, extract}]                │    │       │
│  │  │      → action: "answer" | "deeper" | "siblings"    │    │       │
│  │  │                                                     │    │       │
│  │  │      useful 항목을 collected에 append               │    │       │
│  │  │      yield {type: "read", nodes, collected_count,  │    │       │
│  │  │             action, reasoning}                      │    │       │
│  │  │                                                     │    │       │
│  │  │  6c. ACTION ROUTING                                │    │       │
│  │  │      ┌─────────────┬─────────────┬────────────┐    │    │       │
│  │  │      ▼             ▼             ▼             │    │    │       │
│  │  │   "answer"     "deeper"     "siblings"        │    │    │       │
│  │  │   → break      → frontier     → pass           │    │    │       │
│  │  │                  교체           (다음 iter)     │    │    │       │
│  │  └─────────────────────────────────────────────────────┘    │       │
│  └─────────────────────────────────────────────────────────────┘       │
│                         │                                               │
│            ┌────────────┴────────────────────────┐                      │
│            ▼                                     ▼                      │
│     action == "deeper"                     action == "answer"           │
│     frontier 교체                           → Step 8                    │
│            │                                                           │
│            ▼                                                           │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │  Step 7: Raw Event Expansion & Inspection                   │       │
│  │                                                             │       │
│  │  선택된 노드의 kind에 따라 분기:                             │       │
│  │                                                             │       │
│  │  ┌─ action ──────────────────────────────────────────┐      │       │
│  │  │ _expand_action_context(node)                      │      │       │
│  │  │   context.text → raw_keyboard 가상 노드            │      │       │
│  │  │   context.mouse_summaries → raw_mouse 가상 노드    │      │       │
│  │  │     (각 클러스터의 스크린샷 경로 포함)              │      │       │
│  │  │ 가상 노드를 node_idx에 등록 → frontier에 추가      │      │       │
│  │  └────────────────────────────────────────────────────┘      │       │
│  │                                                             │       │
│  │  ┌─ raw_keyboard ────────────────────────────────────┐      │       │
│  │  │ 키보드 텍스트를 그대로 collected에 추가             │      │       │
│  │  │ LLM 호출 없음                                      │      │       │
│  │  └────────────────────────────────────────────────────┘      │       │
│  │                                                             │       │
│  │  ┌─ raw_mouse ───────────────────────────────────────┐      │       │
│  │  │ _screenshots에서 첫 스크린샷 로드                  │      │       │
│  │  │ LLM.complete_with_vision(prompt, image) (#n+4)    │      │       │
│  │  │ → {useful, extract, reasoning}                    │      │       │
│  │  │ useful이면 collected에 추가                        │      │       │
│  │  │ yield {type: "inspect", image_url, useful, ...}   │      │       │
│  │  │ detail crop 있으면 _inspect_detail() 추가 검사     │      │       │
│  │  └────────────────────────────────────────────────────┘      │       │
│  │                                                             │       │
│  │  ┌─ raw_file ────────────────────────────────────────┐      │       │
│  │  │ read_file_content(filepath, max_chars=8000)       │      │       │
│  │  │ → 파일 내용을 collected에 추가                     │      │       │
│  │  │ LLM 호출 없음                                      │      │       │
│  │  └────────────────────────────────────────────────────┘      │       │
│  │                                                             │       │
│  │  ┌─ raw_url ─────────────────────────────────────────┐      │       │
│  │  │ fetch_url_content(url, max_chars=8000)            │      │       │
│  │  │ → 웹 페이지 내용을 collected에 추가                │      │       │
│  │  │ LLM 호출 없음                                      │      │       │
│  │  └────────────────────────────────────────────────────┘      │       │
│  │                                                             │       │
│  │  ┌─ location ────────────────────────────────────────┐      │       │
│  │  │ children(actions)을 frontier에 추가                │      │       │
│  │  │ + full_location이 파일/URL이면 직접 _inspect_raw   │      │       │
│  │  └────────────────────────────────────────────────────┘      │       │
│  │                                                             │       │
│  │  ┌─ session/app ─────────────────────────────────────┐      │       │
│  │  │ children을 frontier에 추가 (다음 레벨로)           │      │       │
│  │  └────────────────────────────────────────────────────┘      │       │
│  └─────────────────────────────────────────────────────────────┘       │
│                         │                                               │
│                         ▼                                               │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │  Step 8: Answer Generation (LLM #final)                     │       │
│  │  collected = [{node_id, extract}, ...]                     │       │
│  │  prompt = retrieve_answer 템플릿 + query + collected       │       │
│  │  LLM.complete() → 최종 답변                                │       │
│  │  sources = 수집된 node_id 집합                              │       │
│  │  yield {type: "answer", content, sources}                  │       │
│  └─────────────────────────────────────────────────────────────┘       │
│                                                                         │
│  ※ 전체 LLM 호출: 1(time) + 1(day, optional) + N×2(select+evaluate)   │
│    + M(vision) + 1(answer) ≈ 5~20회 (질문에 따라)                       │
│  ※ 벡터 검색 없음. LLM reasoning = 검색 엔진                           │
└─────────────────────────────────────────────────────────────────────────┘
```

### Step 1: 트리 로드

`_load_all_trees()` (`retrieve.py:80`)가 `~/data/trees/`에서 모든 `*_time.json` 파일을 날짜순으로 로드한다. 각 파일은 `{saved_at, mode, date, tree}` 구조를 가진다.

### Step 2: 시간 Pre-filtering (LLM)

`_resolve_time_range()` (`retrieve.py:208`)가 LLM을 호출하여 query에서 시간 정보를 추출한다:

```
입력: "What was I working on yesterday morning?"
현재: "2026-07-02 14:30 Thursday"
LLM 출력: {has_time: true, dates: ["2026-07-01"], start_hour: 9, end_hour: 12}
```

- `dates`가 있으면 `_filter_trees_by_dates()`로 해당 날짜 tree만 필터링 (LLM Day selection 단계 생략)
- `start_hour`/`end_hour`가 있으면 Session 확장 단계에서 `_sessions_in_range()`로 시간 필터링
- 자정 넘김(예: 22-06)은 두 개 서브윈도우 `[22,24)` + `[0,6)`으로 분할 처리
- 시간 정보가 없으면 모든 tree를 사용

### Step 3: node_idx 구축

`_node_index()` (`retrieve.py:96`)가 모든 트리를 순회하며 `node_id → node` 맵을 구축한다. 이후 Select 단계에서 LLM이 반환한 node_id로 실제 노드를 찾을 때 사용한다.

### Step 4: Day Selection

- **dates 있음**: LLM 호출 없이 해당 Day 노드를 자동 선택. `explored`에 추가
- **dates 없음**: LLM이 Day 노드들의 ToC(`_format_toc`)를 읽고 관련 Day 선택:
  - ToC 형식: `"- [d20260701] (day) 2026-07-01\n  Session summaries..."`
  - LLM이 `{reasoning, selected: [node_id, ...], action: "..."}` JSON 반환
  - `action == "sufficient"`이면 바로 Answer 단계로 이동

### Step 5: Session Expansion

선택된 Day 노드의 자식(session)들을 frontier로 구성한다:
- `has_hours`이면 `_sessions_in_range(day, start_hour, end_hour)`로 시간 필터링
- 필터링 결과가 없으면 fallback으로 모든 session 사용

### Step 6: Main Loop (Select → Evaluate 반복)

`max_iterations`(15) 회 반복하며 트리를 top-down으로 순회한다:

**6a. Select** (`retrieve.py:569`):
- frontier 중 `explored`에 없는 노드들의 ToC를 생성
- ToC에는 각 노드의 `node_id`, `kind`, `title`, `summary`(또는 자식 summary 취합) 포함
- `_truncate_prompt()`로 전체 prompt를 `max_prompt_chars`(42,000자)로 제한
- LLM 반환: `{reasoning, selected: [node_id, ...], action: "sufficient"|"deeper"|...}`
- `action == "sufficient"`이고 이전 action이 `"deeper"`가 아니면 종료
- 이전 action이 `"deeper"`인데 selected가 비어있으면 첫 `max_nodes`개를 auto-pick

**6b. Evaluate** (`retrieve.py:616`):
- 선택된 노드의 상세 내용(`_format_details`) 생성:
  - `summary` + `evidence` + children overview (최대 8개 자식의 title + summary)
- LLM 반환: `{reasoning, useful: [{node_id, extract}], action: "answer"|"deeper"|"siblings"}`
- `useful` 항목을 `collected` 리스트에 append
- `action == "answer"`이면 종료 → Step 8

**6c. Action Routing**:
- `"answer"`: 종료
- `"deeper"`: `useful` 노드의 자식을 frontier로 교체 → Step 7
- `"siblings"`: frontier 유지, 다음 iteration 계속

### Step 7: Raw Event Expansion & Inspection

`action == "deeper"`일 때, 선택된 노드의 `kind`에 따라 frontier를 재구성한다:

- **action**: `_expand_action_context()`가 action 노드를 raw 이벤트로 확장
  - `context.text` → `raw_keyboard` 가상 노드 (node_id: `{nid}::kb`)
  - `context.mouse_summaries` → `raw_mouse` 가상 노드들 (node_id: `{nid}::mouse_{i}`)
  - 각 raw_mouse 노드에 `_screenshots` (스크린샷 파일 경로) 포함
  - 가상 노드를 `node_idx`에 등록 후 frontier에 추가
  
- **raw_keyboard**: 키보드 텍스트를 `collected`에 직접 추가. LLM 호출 없음
- **raw_mouse**: `_inspect_raw_node()`가 vision LLM으로 스크린샷 검사
  - `LLM.complete_with_vision(prompt, image_paths=[full_path], detail="auto")`
  - `detail` crop 이미지가 있으면 `_inspect_detail()`로 추가 검사 (detail="high")
- **raw_file**: `read_file_content(filepath)`로 파일 내용을 `collected`에 추가. LLM 호출 없음
- **raw_url**: `fetch_url_content(url)`로 웹 페이지 내용을 `collected`에 추가. LLM 호출 없음
- **location**: children(actions)을 frontier에 추가 + `full_location`이 파일/URL이면 직접 `_inspect_raw_node` 호출
- **session/app**: children을 frontier에 추가 (다음 레벨로)

### Step 8: Answer Generation

`_generate_answer()` (`retrieve.py:956`)가 collected context로 최종 답변을 생성한다:
- `collected`가 비어있으면 "I couldn't find relevant information..." 반환
- prompt = `retrieve_answer` 템플릿 + query + collected extracts
- LLM 호출 → 최종 텍스트 답변
- `sources` = collected에서 추출한 고유 node_id 집합
- `yield {type: "answer", content, sources}`

### 검색 step 타입

| step type | 설명 | yield 시점 |
|-----------|------|-----------|
| `time_filter` | 시간 추출 결과 | Step 2 |
| `browse` | LLM이 ToC를 보고 노드 선택 | Step 4, 6a |
| `read` | LLM이 선택된 노드의 상세를 평가 | Step 6b, 7 (raw_keyboard/file/url) |
| `inspect` | vision LLM이 스크린샷 검사 | Step 7 (raw_mouse) |
| `answer` | 최종 답변 | Step 8 |
| `error` | 오류 | 예외 상황 |

### 검색 특징

- **No vector, no embedding**: 트리 구조 + LLM reasoning만으로 검색
- **추적 가능**: 각 step이 yield되어 검색 경로가 투명. web UI에서 step별 시각화 가능
- **Cross-day reasoning**: 여러 날짜의 tree를 동시에 로드하여 교차 날짜 질의 가능
- **Vision 검사**: 스크린샷을 vision LLM으로 검사하여 시각적 정보 추출
- **파일/URL 직접 읽기**: location 노드의 파일 경로나 URL을 직접 읽어 내용 제공
- **Prompt 제한**: 모든 prompt가 `max_prompt_chars`(42,000자)로 truncate되어 컨텍스트 창 초과 방지

## Key Components

- **`CatchMe`** (`__init__.py:23`) — 최상위 API. Engine + Store + recorders 초기화, start/stop/query/search
- **`Engine`** (`engine.py:19`) — recorder → Queue → SQLite batch write + Organizer 동시 실행
- **`Organizer`** (`organizer.py:25`) — boundary 이벤트 감지 → incremental tree 빌드 → closed 노드 enqueue
- **`build_tree()` / `extend_tree()`** (`tree.py:80/138`) — 룰 기반 5계층 트리 구성 + 증분 확장
- **`SummaryQueue`** (`summary_queue.py:52`) — 우선순위 큐 + ThreadPool로 4계층 비동기 요약
- **`retrieve()`** (`retrieve.py:435`) — tree traversal reasoning 검색 (Select → Evaluate → Deeper → Answer)
- **`_inspect_raw_node()`** (`retrieve.py:717`) — raw event 검사 (키보드/파일/URL/스크린샷 vision)
- **`Store`** (`store.py:46`) — SQLite 단일 테이블 + FTS5, thread-local connection
- **`LLM`** (`services/llm.py:224`) — OpenAI 호환 클라이언트 (sync/async/vision/streaming), call budget + token tracker
- **`build_window_spans()`** (`filter.py:57`) — window 이벤트 → span (3초 필터, 병합, briefs 포함)
- **`cluster_events()`** (`filter.py:157`) — 시간 gap + scroll session semantic 기반 클러스터링

## Analysis

### 장점

- **벡터 DB 불필요**: PageIndex와 동일한 철학. 임베딩, 청킹, 벡터 스토어 없음
- **실시간 캡처**: 이벤트 드리븐 방식으로 타이머 없이 즉시 반응. window switch / idle이 트리거
- **룰 기반 트리**: 인덱싱에 LLM이 필요 없음. 요약만 LLM 사용. 비용 효율적
- **증분 확장**: 전체 재빌드가 아닌 마지막 session만 업데이트. 기존 summary 보존
- **닫힌 노드만 요약**: 현재 활성 노드는 요약하지 않아 LLM 호출 최소화
- **Vision 검색**: 스크린샷을 vision LLM으로 검사하여 시각적 정보까지 검색 가능
- **로컬 우선**: 모든 데이터 로컬 저장, Ollama/vLLM으로 완전 오프라인 가능
- **FTS5 키워드 검색**: tree retrieval 외에 SQLite FTS5로 빠른 키워드 검색도 제공

### 단점

- **플랫폼 의존**: macOS(windows keyboard_macos) / Windows(keyboard_windows) 각각 구현 필요
- **LLM 요약 비용**: 2시간 사용시 ~6M input / ~0.7M output 토큰. 장시간 사용시 비용 누적
- **스크린샷 저장 용량**: 2시간 사용시 ~200MB 디스크. 장기 사용시 용량 부담
- **검색 속도**: tree traversal이 여러 LLM 호출을 필요로 함 (5-20s per query)
- **단일 사용자**: multi-device 동기화는 roadmap (미구현)

### PageIndex와의 비교

- **인덱싱**: PageIndex는 LLM으로 TOC 탐지 + 트리 생성. CatchMe는 룰 기반(_LAYER 구성에 LLM 불필요), 요약만 LLM
- **검색**: 둘 다 tree-based reasoning retrieval. CatchMe는 raw event(키보드/스크린샷/파일/URL)까지 확장 검사
- **저장**: PageIndex는 JSON + lazy-load. CatchMe는 SQLite + FTS5 + JSON tree + blob 파일
- **대상**: PageIndex는 문서(PDF/MD). CatchMe는 실시간 사용자 활동

## References

- [CatchMe GitHub](https://github.com/HKUDS/CatchMe)
- [CatchMe Blog](https://hkuds.github.io/CatchMe/)
- [PageIndex](https://github.com/VectifyAI/PageIndex) — tree-structured retrieval 영감 출처
- [HKUDS Ecosystem](https://github.com/HKUDS) — NanoBot, CLI-Anything, ClawWork, ClawTeam
