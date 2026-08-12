# Engine + Organizer: 캡처 → 트리 빌드 오케스트레이션

> 출처: [CatchMe_HKUDS.md](../../../report/[git]_CatchMe_HKUDS.md) / `source/git/CatchMe_HKUDS/catchme/engine.py`, `organizer.py`

## 설명

Engine은 6개 recorder의 이벤트를 Queue로 받아 SQLite에 batch write하면서, 동시에 Organizer daemon에게 boundary event를 알린다. Organizer는 window switch / idle 이벤트를 트리거로 activity tree를 incremental 빌드하고, 닫힌 노드를 SummaryQueue에 enqueue한다.

## 코드

```python
# engine.py:47-91  start — recorder → queue → SQLite + organizer 동시 시작
def start(self) -> None:
    self._stop.clear()
    self._writer = threading.Thread(target=self._write_loop, daemon=True)
    self._writer.start()
    self._organizer_thread = threading.Thread(target=self._organizer.run, daemon=True)
    self._organizer_thread.start()
    for rec in self._recorders:
        emitter = self._make_emitter(rec.kind)
        rec.start(emitter)  # 각 recorder는 emit() 콜백을 받음

def _make_emitter(self, kind: str):
    def emit(data: dict, blob: str = "") -> None:
        if self._paused:
            return
        event = Event(timestamp=time.time(), kind=kind, data=data, blob=blob)
        self._queue.put(event)           # SQLite write용
        self._organizer.on_event(event)  # 트리 빌드 트리거
    return emit
```

```python
# engine.py:93-110  _write_loop — batch write (batch_size / batch_timeout)
def _write_loop(self) -> None:
    while not self._stop.is_set():
        batch: list[Event] = []
        deadline = time.monotonic() + self._config.batch_timeout
        while len(batch) < self._config.batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                batch.append(self._queue.get(timeout=remaining))
            except Empty:
                break
        if batch:
            self._store.insert_raw(batch)
```

```python
# organizer.py:51-59  on_event — boundary 감지 (window switch / idle)
def on_event(self, event: Event) -> None:
    if event.kind == "window":
        key = (event.data.get("app"), event.data.get("title"))
        if key != self._last_window_key:  # 창이 바뀌면
            self._last_window_key = key
            self._pending.set()           # 트리 빌드 트리거
    elif event.kind == "idle":
        if event.data.get("status") in ("idle", "locked"):
            self._pending.set()
```

```python
# organizer.py:83-141  _process — incremental tree 빌드 + closed 노드 enqueue
def _process(self) -> None:
    now = time.time()
    if now - self._last_build_time < self._debounce_sec:  # 3초 debounce
        return
    today = datetime.now().strftime("%Y-%m-%d")

    if self._tree_cache is None:
        # 최초: 오늘 전체 트리 빌드
        cached = load_tree(today, "time")
        if cached and cached.get("tree"):
            self._tree_cache = cached  # 디스크에서 복원
        else:
            result = build_tree(self._store, since=d0, until=until, mode="time")
            self._tree_cache = result
    else:
        # 증분: 마지막 이벤트 이후 새 이벤트만 extend
        modified = extend_tree(self._tree_cache, self._store,
                               since=self._last_event_ts - 1.0, until=until)
        if not modified:
            return

    tree = self._tree_cache.get("tree")
    if tree:
        self._enqueue_closed_nodes(tree)  # 닫힌 노드 → SummaryQueue
        self._save_tree()                 # JSON에 snapshot 저장

# organizer.py:145-181  _walk_enqueue — 마지막 sibling이 아닌(=닫힌) 노드만 enqueue
def _walk_enqueue(self, node, is_last_sibling, parent, kind_to_level):
    children = node.get("children", [])
    for i, ch in enumerate(children):
        child_is_last = i == len(children) - 1
        self._walk_enqueue(ch, child_is_last, parent=node, kind_to_level=kind_to_level)

    kind = node.get("kind", "")
    if kind not in kind_to_level:
        return
    nid = node.get("node_id", "")
    if parent:
        self._queue.register_parent(nid, parent)
    is_closed = not is_last_sibling  # 마지막 sibling = 현재 활성 노드 → 닫히지 않음
    if is_closed:
        self._queue.mark_closed(nid)
    if not is_closed or node.get("summary"):
        return
    level = kind_to_level[kind]  # action=1, location/app=2, session=3
    self._queue.enqueue(node, level, parent=parent, is_closed=True)
```
