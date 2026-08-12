# timeline_injection — 중간 주입 타임라인 생성

> 출처: [분석 문서](../../../report/[paper][git]_Keep_It_InMind_Benchmarking_the_Implicit-Association_Blind_Spot_in_Agent_Memory_2026_arxiv.md) / submodule 경로: `source/git/InMind_imlrz`

## 설명

논문의 "38 further sessions of ordinary interaction before it is ever queried"라는 long-horizon 주입 프로토콜을 코드로 보증하는 핵심 모듈. `build_timeline.py`가 고정 47-session LME-s 배경 trace의 **9번째 session(인덱스 8) 끝**에 task의 target user/assistant pair를 append하고, `manifest.json`의 주입 상수가 이 위치를 pin한다. 모든 task가 동일한 배경·동일 주입점을 공유하며, target 뒤 38 session의 일상 대화가 이어져 memory가 realistic interference에 생존해야 함을 코드 차원에서 강제한다.

`validate_release.py`는 manifest의 모든 상수(turn/session count, injection index, prefix/through-injection user-turn count, SHA-256)를 assertion으로 검증해 아티팩트 변조를 차단한다.

## manifest.json (주입 상수 — 평가 reproducibility의 핵심)

```json
{
  "name": "InMind fixed LME-s background trace",
  "file": "lme_s_background.jsonl",
  "sha256": "aef51e8a4e606a3ac5ceafdc64d875a7f1f41a051dafd68dce45f9c90d626da1",
  "turns": 486,
  "sessions": 47,
  "user_turns": 240,
  "assistant_turns": 246,
  "injection_session_index": 8,
  "injection_session_number": 9,
  "phase_a_sessions": 8,
  "prefix_user_turns": 35,
  "injection_user_turn_locator": 40,
  "background_user_turns_before_target": 41,
  "post_injection_sessions": 38,
  "source": {
    "project": "LongMemEval",
    "repository": "https://github.com/xiaowu0162/LongMemEval",
    "paper": "https://arxiv.org/abs/2410.10813",
    "dataset": "longmemeval_s_cleaned.json",
    "question_id": "0bc8ad93",
    "upstream_commit": "9e0b455f4ef0e2ab8f2e582289761153549043fc",
    "license": "MIT"
  }
}
```

주입 상수의 의미:
- `injection_session_index: 8` → 9번째 session에 주입(0-indexed). `phase_a_sessions: 8`로 앞단 8 session은 공통 prefix.
- `injection_user_turn_locator: 40` → 논문 configuration명 `inject-turn 40`: 40번째 background user turn을 포함하는 session 선택.
- `background_user_turns_before_target: 41` → session 경계를 보존해 9번째 session 전체 처리하므로 target 앞 41 user turn(잘린 40이 아님). README가 "41 background user turns—not a truncated 40-turn prefix"로 강조.
- `post_injection_sessions: 38` → target 뒤 38 session. 논문 "38 further sessions"과 정확 대응.
- `source` → LongMemEval(`question_id: 0bc8ad93`)에 upstream attribution + MIT license.

## build_timeline.py 핵심 (sessions 그룹화 + target append)

```python
def group_sessions(turns):
    # turn id "sid-turn-N" 에서 session id 추출, 연속성 검증(같은 session이 비연속이면 에러)
    grouped: OrderedDict[str, list] = OrderedDict()
    closed: set[str] = set(); previous = None
    for turn in turns:
        sid = session_id(str(turn.get("id","")))
        if sid != previous:
            if sid in closed:
                raise ValueError(f"Session {sid} is not contiguous in the background file")
            if previous is not None:
                closed.add(previous)
            previous = sid
        copied = dict(turn); copied["source"] = "lme_s_background"
        grouped.setdefault(sid, []).append(copied)
    return [{"session_id": sid, "turns": rows} for sid, rows in grouped.items()]

def build_timeline(task, background_turns, manifest):
    sessions = group_sessions(background_turns)
    injection_index = int(manifest["injection_session_index"])   # 8
    if not 0 <= injection_index < len(sessions):
        raise ValueError(...)
    task_id = int(task["task_id"])
    target_turns = [
        {"id": f"inmind-task-{task_id}-turn-0", "role": "user",
         "content": task["user_message"], "source": "inmind_target"},
        {"id": f"inmind-task-{task_id}-turn-1", "role": "assistant",
         "content": task["assistant_message"], "source": "inmind_target"},
    ]
    sessions[injection_index]["turns"].extend(target_turns)        # 9번째 session 끝 append
    return {
        "task_id": task_id,
        "protocol": {
            "name": "inmind-middle-injection-v1",
            "background_sessions": len(sessions),
            "injection_session_index": injection_index,
            "injection_session_number": injection_index + 1,
            "background_user_turns_before_target": int(manifest["background_user_turns_before_target"]),
            "post_injection_sessions": len(sessions) - injection_index - 1,
            "target_position": "end_of_injection_session",
            "query_state_policy": "naive and indirect queries read the same frozen post-history state",
        },
        "sessions": sessions,
        "queries": {"naive_query": task["naive_query"], "query": task["query"]},
    }
```

출력 timeline 구조: `sessions`(47개, 9번째 끝에 `source:"inmind_target"` 2턴) + `queries` 객체(test query는 timeline **외부**에 분리 — memory에 ingest 금지). `query_state_policy`가 naive/indirect가 동일 frozen state를 읽도록 명시.

## 주의 (protocol non-negotiable)

SKILL.md / README가 금지하는 행위:
- target을 모든 background turn 뒤에 주입(tail injection)
- target을 별도 later session으로 이동
- `entity_1/entity_2/relation/explanation/provenance`를 system under test에 노출(judge-only)
- naive query의 retrieval/answer이 indirect용 state를 변경
- 서로 다른 task 간 target-dependent state 재사용

phase A(처음 8 session)는 immutable prefix bank로 사전 구축해 task마다 copy 허용 — 매 task 47 session을 처음부터 replay하는 비용 절감.
