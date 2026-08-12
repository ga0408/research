# retrieve.py + summary_queue.py: 트리 검색 + 비동기 요약

> 출처: [CatchMe_HKUDS.md](../../../report/[git]_CatchMe_HKUDS.md) / `source/git/CatchMe_HKUDS/catchme/pipelines/retrieve.py`, `summary_queue.py`

## 설명

CatchMe의 검색은 PageIndex에서 영감을 받은 tree-based reasoning retrieval이다. LLM이 Day → Session → App → Location → Action → raw event(키보드/마우스/스크린샷)까지 top-down으로 트리를 순회하며, 각 레벨에서 ToC를 읽고 관련 노드를 선택하고, 선택한 노드의 상세 내용을 읽고, 충분한 정보가 모이면 답변을 생성한다. 요약은 4계층(L0~L3)으로 bottom-up 비동기 처리된다.

## 코드

```python
# retrieve.py:435-711  retrieve — 메인 검색 generator (tree traversal)

def retrieve(query: str) -> Iterator[dict]:
    llm = LLM()
    trees = _load_all_trees()  # 모든 날짜의 tree JSON 로드

    # 1. 시간 기반 pre-filtering (LLM이 query에서 날짜/시간 추출)
    time_range = _resolve_time_range(query, llm)
    if has_dates:
        trees = _filter_trees_by_dates(trees, time_range["dates"])

    # 2. node_id → node lookup 인덱스 구축
    node_idx = {}
    for t in trees:
        _node_index(t["tree"], node_idx)

    # 3. Day → Session 확장
    day_nodes = [t["tree"] for t in trees]
    frontier = []
    for day in selected_nodes:
        if has_hours:
            frontier.extend(_sessions_in_range(day, start_hour, end_hour))
        else:
            frontier.extend(day.get("children", []))

    # 4. 메인 루프: Select → Evaluate → Deeper/Sufficient 반복
    for _iteration in range(max_iters):  # 기본 15회
        available = [n for n in frontier if n.get("node_id") not in explored]
        if not available:
            break

        # Step A: LLM Select — ToC 보고 관련 노드 선택
        toc_text = _format_toc(available, explored)
        prompt = _get_prompt("retrieve_select").format(
            query=query, collected=_format_collected(collected),
            prev_action=prev_action, toc=toc_text, max_nodes=max_nodes)
        sel_result = _llm_json(llm, prompt, temperature=0.3)

        if sel_result.get("action") == "sufficient":
            break  # 충분한 정보 수집됨

        sel_nodes = [node_idx[nid] for nid in sel_result.get("selected", []) if nid in node_idx]
        explored.update(sel_result.get("selected", []))

        # Step B: LLM Evaluate — 선택된 노드의 상세 내용 읽기
        details_text = _format_details(sel_nodes)  # summary + evidence + children overview
        prompt = _get_prompt("retrieve_evaluate").format(
            query=query, collected=_format_collected(collected),
            prev_action=prev_action, details=details_text)
        eval_result = _llm_json(llm, prompt, temperature=0.3)

        # useful로 판단된 노드의 extract를 collected에 추가
        for u in eval_result.get("useful", []):
            if u.get("node_id") and u.get("extract"):
                collected.append(u)

        action = eval_result.get("action", "answer")
        if action == "answer":
            break
        elif action == "deeper":
            # 선택된 노드의 자식으로 frontier 교체
            frontier = []
            for n in sel_nodes:
                kind = n.get("kind", "")
                if kind == "action":
                    # action → raw event 확장 (키보드 텍스트, 마우스 클러스터, 스크린샷)
                    expanded = _expand_action_context(n)
                    for vn in expanded:
                        node_idx[vn["node_id"]] = vn
                    frontier.extend(expanded)
                elif kind in ("raw_keyboard", "raw_mouse", "raw_file", "raw_url"):
                    yield from _inspect_raw_node(llm, query, n, collected)
                elif kind == "location":
                    # location → children(actions) + 파일/URL 직접 읽기
                    frontier.extend(n.get("children", []))
                    # 파일이나 URL이 있으면 직접 _inspect_raw_node로 읽기
                else:
                    frontier.extend(n.get("children", []))

    # 5. 최종 답변 생성
    yield from _generate_answer(llm, query, collected)


# retrieve.py:307-347  _expand_action_context — action 노드를 raw 이벤트로 확장
def _expand_action_context(node):
    ctx = node.get("context", {})
    nid = node.get("node_id", "")
    items = []
    # 키보드 입력 → raw_keyboard 가상 노드
    if ctx.get("text"):
        items.append({"node_id": f"{nid}::kb", "kind": "raw_keyboard",
                      "summary": f"Raw keystroke stream: {text[:500]}"})
    # 마우스 클러스터 → raw_mouse 가상 노드 (스크린샷 포함)
    for i, ms in enumerate(ctx.get("mouse_summaries", [])):
        screenshots = _find_screenshots_for_cluster(ms["start"], ms["end"], mouse_actions)
        items.append({"node_id": f"{nid}::mouse_{i}", "kind": "raw_mouse",
                      "summary": ms.get("summary", ""), "_screenshots": screenshots})
    return items


# retrieve.py:717-896  _inspect_raw_node — raw 이벤트 검사 (키보드/파일/URL/스크린샷)
def _inspect_raw_node(llm, query, node, collected, max_file_chars=8000):
    kind = node.get("kind", "")
    if kind == "raw_keyboard":
        # 키보드 텍스트를 그대로 collected에 추가
        collected.append({"node_id": nid, "extract": f"Keyboard input: {raw_text}"})
    elif kind == "raw_file":
        # 파일 내용 읽기 (read_file_content)
        content, file_type = read_file_content(filepath, max_chars=max_file_chars)
        collected.append({"node_id": nid, "extract": f"File content:\n{content}"})
    elif kind == "raw_url":
        # URL fetch (fetch_url_content)
        content = fetch_url_content(url, max_chars=max_file_chars)
        collected.append({"node_id": nid, "extract": f"Web page content:\n{content}"})
    elif kind == "raw_mouse":
        # 스크린샷을 vision LLM으로 검사
        screenshots = node.get("_screenshots", [])
        full_path = _resolve_blob(screenshots[0]["full"])
        raw = llm.complete_with_vision(prompt=prompt, image_paths=[full_path], detail="auto")
        vresult = json.loads(raw)
        if vresult.get("useful") and vresult.get("extract"):
            collected.append({"node_id": nid, "extract": vresult["extract"]})
        # detail crop이 있으면 추가 검사 (_inspect_detail)


# retrieve.py:956-981  _generate_answer — collected context로 최종 답변
def _generate_answer(llm, query, collected, temperature=0.7, max_tokens=None):
    if not collected:
        yield {"type": "answer", "content": "I couldn't find relevant information..."}
        return
    prompt = _get_prompt("retrieve_answer").format(
        query=query, collected=_format_collected(collected))
    answer = llm.complete([{"role": "user", "content": prompt}],
                          temperature=temperature, max_tokens=max_tokens)
    sources = list({c["node_id"] for c in collected})
    yield {"type": "answer", "content": answer, "sources": sources}
```

```python
# summary_queue.py:52-198  SummaryQueue — 우선순위 기반 비동기 요약

class SummaryQueue:
    """PriorityQueue + ThreadPoolExecutor로 LLM 요약을 비동기 처리.
    level이 낮을수록 우선 (action=1 > location/app=2 > session=3).
    자식 요약 완료시 부모로 cascade."""

    def __init__(self, max_workers=2, save_fn=None):
        self._q = PriorityQueue()
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._dispatcher = threading.Thread(target=self._dispatch_loop, daemon=True)
        self._parent_map = {}   # child_nid → parent_node
        self._closed_set = set()  # 닫힌 노드 ID 집합

    def enqueue(self, node, level, parent=None, is_closed=True):
        nid = node.get("node_id", "")
        if nid in self._in_flight:
            return
        if not self._needs_summary(node):  # 이미 summary 있음
            return
        self._in_flight.add(nid)
        if parent:
            self._parent_map[nid] = parent
        if is_closed:
            self._closed_set.add(nid)
        self._q.put(_QueueItem(level, node))  # (level, enqueue_time, node_id) 우선순위

    def _process_item(self, item):
        node, nid = item.node, item.node_id
        # 자식 요약이 완료될 때까지 대기 (최대 10회 retry)
        if not self._ready(node):
            if item.retry < 10:
                time.sleep(1.0)
                self._in_flight.discard(nid)
                item.retry += 1
                self.enqueue(node, item.level, parent=self._parent_map.get(nid))
            return

        produced = summarize_node(node, self._llm)  # LLM 호출
        if produced:
            self._write_notification(node)  # data/summary_updates.jsonl에 기록
            self._maybe_save()              # 주기적 tree 저장
            self._cascade_parent(nid)       # 부모 노드 요약 cascade

    def _cascade_parent(self, child_nid):
        parent = self._parent_map.get(child_nid)
        if not parent:
            return
        pid = parent.get("node_id", "")
        if pid not in self._closed_set or parent.get("summary"):
            return
        # 부모도 닫힌 노드이고 summary가 없으면 enqueue
        level = KIND_TO_LEVEL.get(parent.get("kind", ""), 3)
        self.enqueue(parent, level, parent=self._parent_map.get(pid))
```
