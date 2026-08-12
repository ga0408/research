# tree.py: 룰 기반 계층적 트리 구성 + 증분 확장

> 출처: [CatchMe_HKUDS.md](../../../report/[git]_CatchMe_HKUDS.md) / `source/git/CatchMe_HKUDS/catchme/pipelines/tree.py`

## 설명

CatchMe의 트리 빌드는 LLM 없이 순수 룰 기반으로 동작한다. window span → session 분할(window idle gap 5분) → app 그룹핑 → location(URL/file/title) 그룹핑 → action(interaction cluster 3초 gap)의 5계층을 구성한다. extend_tree는 마지막 session만 재빌드하여 증분 업데이트한다.

## 코드

```python
# tree.py:80-135  build_tree — 룰 기반 5계층 트리 구성 진입점
def build_tree(store, since=None, until=None, mode="time", cfg=None):
    merged_cfg = {**_DEFAULT_TREE, **(load_filter_config()), **(cfg or {})}
    # 6종 이벤트 조회
    windows = store.query_raw(kind="window", since=since, until=until, limit=50000)
    all_kb = store.query_raw(kind="keyboard", ...)
    all_mouse = store.query_raw(kind="mouse", ...)
    all_clip = store.query_raw(kind="clipboard", ...)
    all_idle = store.query_raw(kind="idle", ...)

    # window 이벤트 → span (min_dwell 3초 필터 + 동일창 병합)
    spans = build_window_spans(windows, merged_cfg["window_min_dwell"],
                               max_span_dwell=merged_cfg["session_gap"])
    interaction = (sorted(all_kb,...), sorted(all_mouse,...), sorted(all_clip,...))

    if mode == "app":
        root = _build_by_app(day_id, day_date, day_start, day_end, spans, interaction, action_gap)
    else:
        root = _build_by_time(day_id, day_date, day_start, day_end, spans, all_idle,
                              merged_cfg["session_gap"], interaction, action_gap)
    return {"tree": root.to_dict(), "mode": mode}


# tree.py:523-562  _build_by_time — Day → Session → App → Location → Action
def _build_by_time(day_id, day_date, day_start, day_end, spans, idle_events,
                   session_gap, interaction, action_gap):
    day_node = ActivityNode(node_id=day_id, kind="day", title=day_date,
                            start=day_start, end=day_end)
    # idle gap(5분) 또는 시간 gap으로 session 분할
    session_groups = _split_sessions(spans, idle_events, session_gap)
    for sess_spans in session_groups:
        sid = f"{day_id}_s{int(sess_spans[0].start)}"
        session_node = ActivityNode(node_id=sid, kind="session",
                                    title=f"{_fmt_hm(s_start)} – {_fmt_hm(s_end)}", ...)
        # session → App → Location → Action 자식 구성
        session_node.children = _build_app_location_children(sid, sess_spans, interaction, action_gap)
        day_node.children.append(session_node)
    return day_node


# tree.py:448-517  _build_app_location_children — App → Location → Action
def _build_app_location_children(id_prefix, spans, interaction, action_gap):
    by_app = OrderedDict()
    for s in spans:
        by_app.setdefault(s.app, []).append(s)  # app 이름으로 그룹핑

    app_nodes = []
    for app_name, app_spans in by_app.items():
        aid = f"{id_prefix}_{_sanitize_app(app_name)}"
        app_node = ActivityNode(node_id=aid, kind="app", title=app_name, ...)

        by_loc = OrderedDict()
        for s in app_spans:
            loc_key = s.url or s.filepath or s.title  # location 키 결정
            by_loc.setdefault(loc_key, []).append(s)

        for loc_key, loc_spans in by_loc.items():
            lid = f"{aid}_l{_hash_loc(loc_key)}"
            loc_node = ActivityNode(node_id=lid, kind="location", title=loc_display, ...)
            # location → Action (keyboard+mouse+clipboard를 3초 gap으로 클러스터링)
            loc_node.children = _collect_actions(loc_spans, lid, interaction, action_gap, ...)
            app_node.children.append(loc_node)
        app_nodes.append(app_node)
    return app_nodes


# tree.py:412-442  _collect_actions — interaction 클러스터링 → Action 노드
def _collect_actions(spans, parent_id, interaction, action_gap, app="", location=""):
    kb_sorted, mouse_sorted, clip_sorted = interaction
    all_kb = [e for e in kb_sorted if span.start <= e.timestamp < span.end]  # ...
    merged = sorted(all_kb + all_mouse + all_clip, key=lambda e: e.timestamp)
    clusters = cluster_events(merged, action_gap)  # 3초 gap으로 클러스터 분할

    return [ActivityNode(node_id=f"{parent_id}_t{int(c.start)}", kind="action",
                         title=_derive_action_title(c.events),  # "typing: hello", "click × 3" 등
                         start=c.start, end=c.end,
                         context=_action_context(c.events, app=app, location=location))
            for c in clusters]


# tree.py:591-620  _split_sessions — idle gap + 시간 gap으로 session 분할
def _split_sessions(spans, idle_events, gap):
    idle_breaks = []
    for ev in idle_events:
        d = ev.data
        if d.get("status") in ("idle", "locked"):
            s, e = d.get("start", ev.timestamp), d.get("end", ev.timestamp)
            if e - s >= gap:  # idle이 5분 이상이면 session 경계
                idle_breaks.append((s, e))
    sessions = [[spans[0]]]
    for i in range(1, len(spans)):
        span_gap = spans[i].start - spans[i-1].end
        in_idle = any(s <= spans[i-1].end and e >= spans[i].start for s, e in idle_breaks)
        if span_gap >= gap or in_idle:
            sessions.append([])  # 새 session
        sessions[-1].append(spans[i])
    return sessions


# tree.py:138-283  extend_tree — 증분 확장 (마지막 session만 재빌드)
def extend_tree(existing, store, since, until=None, cfg=None):
    tree = existing.get("tree")
    # 새 window span 조회
    new_spans = build_window_spans(windows, ...)
    if not new_spans:
        return False
    if mode == "time":
        return _extend_time_tree(tree, new_spans, store, since, until, merged_cfg, interaction, action_gap)

def _extend_time_tree(tree, new_spans, store, since, until, cfg, interaction, action_gap):
    children = tree.get("children", [])
    last_session = children[-1]
    last_end = last_session.get("end", 0)
    gap_to_new = new_spans[0].start - last_end

    if gap_to_new >= session_gap or in_idle:
        # 새 session — 기존 session 닫고 새 session 생성
        for sess_spans in session_groups:
            sess_dict = ActivityNode(...)
            sess_dict.children = _build_app_location_children(...)
            children.append(sess_dict.to_dict())
    else:
        # 마지막 session 연장 — 전체 session 시간 범위 재빌드
        # 기존 summary를 node_id 매칭으로 보존 (_apply_merge)
        new_sess = ActivityNode(...)
        new_sess.children = _build_app_location_children(...)
        old_index = {}
        _index_tree(last_session, old_index)
        _apply_merge(new_dict, old_index)  # 기존 summary 복사
        children[-1] = new_dict
    return True


# tree.py:626-649  _derive_action_title — 이벤트 종류로 action 제목 자동 생성
def _derive_action_title(events):
    kinds = {e.kind for e in events}
    if "clipboard" in kinds: return "copy-paste"
    if "text" in kb_types:
        text = _strip_ime_pinyin(kb_events)  # IME pinyin 중간 단계 제거
        return f"typing: {text[:40]}" if text.strip() else f"typing ({len(kb_events)})"
    if mouse_actions.issubset({"scroll","scroll_start","scroll_end"}): return "scroll"
    if mouse_actions & {"click","double_click"}: return f"click × {len(mouse_events)}"
    if "shortcut" in kb_types: return f"shortcut: {keys[:40]}"
    return f"interaction × {len(events)}"
```
