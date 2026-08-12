> [git] https://github.com/openclaw/openclaw.git

# OpenClaw

> 분석 대상: `openclaw` (fork 체크아웃 기준). TypeScript ESM, pnpm 모노레포. 버전 `2026.6.11`. 약 21,800 파일(`src/` 9,322 + `extensions/` 6,976 + `apps/` + `packages/`).

## Overview

OpenClaw는 사용자 기기에서 직접 구동되는 **로컬 우선 개인 AI 어시스턴트**. 단일 **Gateway 데몬**(`node ... gateway run`)이 세션·채널·도구·이벤트의 제어 평면이며, 20+ 메시징 채널(WhatsApp/Telegram/Slack/Discord/Signal/iMessage/Teams/Matrix/webchat…) 인바운드를 하나의 embedded agent 루프로 수렴시킨다. 핵심은 plugin-agnostic한 `src/` + 번들 플러그인 `extensions/*` 구조이며, 메모리는 `extensions/memory-core`(builtin SQLite 엔진)가 단일 슬롯으로 담당한다.

## Architecture — 전체 동작 Flow

```
   Channel plugin (extensions/<id>)  |  WS chat.send
   Telegram/WhatsApp/Slack/.../webchat
                 │ inbound MsgContext
                 ▼
   Gateway server (src/gateway/server.impl.ts:524 startGatewayServer)
     WS JSON-RPC chat.send (server-methods/chat.ts:3585)
     Channel manager (server-channels.ts:852)  →  plugin monitors
                 │
                 ▼
   dispatchInboundMessage (src/auto-reply/dispatch.ts:528)
        → dispatchReplyFromConfig → getReplyFromConfig (get-reply.ts:240)
        → runPreparedReply (get-reply-run.ts:511)
                 │
                 ▼
   agentCommand (src/agents/agent-command.ts:2595 / 내부 :864)
     prepareAgentCommandExecution(:598): 세션키/에이전트/워크스페이스/모델/인증/스킬 스냅샷 해석
       멀티에이전트 라우팅(agent-scope.ts, session-key.ts: agent:<id>:<key>)
       세션 메타: state/.../sessions.json / 트랜스크립트: .../sessions/<id>/transcript.jsonl (legacy 파일; AGENTS.md 정책상 SQLite가 정규 저장소, sessions.json은 마이그레이션 부채)
                 │
                 ▼
   runAgentAttempt (attempt-execution.ts:481) → runEmbeddedAgent (embedded-agent-runner/run.ts:622)
     레인 큐 입장(command-queue.ts) + OUTER while(true) 재시도/페일오버/컴팩션 루프(run.ts:1957)
                 │  매 이터레이션:
                 ▼
   runEmbeddedAttempt (run/attempt.ts:850)
     [1] 시스템 프롬프트 조립: buildAttemptSystemPrompt (attempt-system-prompt.ts:39) → buildAgentSystemPrompt (system-prompt.ts:680)
     [2] 도구 구성: createOpenClawCodingTools (agent-tools.ts:384) + 정책/스키마 필터
     [3] 컨텍스트 엔진 조립/부트스트랩(context-engine/*, compact.ts) — 임계시 컴팩션 후 재시도
     [4] AgentSession + subscribe (embedded-agent-subscribe.ts:162) — 스트림 → 채널 가시 블록 투영
                 │
                 ▼
   session.prompt → Agent.prompt (agent-session.ts:1145, agent-core/agent.ts:377)
     → runAgentLoop/runLoop (packages/agent-core/src/agent-loop.ts:267)   ◄── 프레임워크 무관 핵심 루프
        ├── streamAssistantResponse (agent-loop.ts:447): provider streamFn(model,{systemPrompt,messages,tools})
        │     events: text_*/thinking_*/toolcall_*/done → AssistantMessage
        ├── stopReason=toolUse → executeToolCalls (agent-loop.ts:549)
        │     resolve tool → beforeToolCall 훅(승인/정책/루프검출) → tool.execute → afterToolCall 훅 → ToolResultMessage → context.push
        └── stopReason != toolUse && steering/follow-up 없으면 → agent_end
                 │
                 ▼
   EmbeddedAgentRunResult → deliverAgentCommandResult (command/delivery.runtime.ts)
     resolveAgentDeliveryPlan → ReplyDispatcher.deliver → 채널 아웃바운드 → 사용자
```

**핵심 루프(`packages/agent-core/src/agent-loop.ts`)**는 프레임워크 무관: 외부 `while(true)`는 steering/follow-up 메시지 펌프, 내부 `while(hasMoreToolCalls||pendingMessages)`가 매 턴 `streamFn`(provider) 호출 → 도구 호출 파싱 → `executeToolCalls`(병렬/순차) → 결과를 `ToolResultMessage`로 컨텍스트에 push → 재스트림. `stopReason`이 `toolUse`가 아니면 내부 루프 종료. 도구는 `result.terminate=true`로 배치 종료 가능(`shouldTerminateToolBatch`).

**백그라운드**(모두 동일 agent 런타임 재사용, 별도 LLM 루프 없음):
- **Cron**: `CronService`(`src/cron/service.ts`)가 게이트웨이 시작시 `start()`(중단 복구+밀린 잡+`armTimer`). due 잡 → `runCronIsolatedAgentTurn`(`cron/isolated-agent/run.ts:1488`)가 `trigger:"cron"`으로 embedded 루프 구동. 저장소 SQLite `cron_jobs` 테이블.
- **Heartbeat**: `runHeartbeatOnce`(`src/infra/heartbeat-runner.ts`)가 동일 dispatch 경유.
- **Memory dreaming**: `memory-core` 플러그인이 관리 cron(`0 3 * * *`)으로 시스템 이벤트 토큰을 세션에 주입 → `before_agent_reply` 훅이 스윕 실행.

> ACP 경로(외부 Codex harness 등)는 `agentCommandInternal`(`agent-command.ts:1084`)에서 분기해 `acpManager.runTurn`로 가며 embedded 루프를 쓰지 않지만 동일 라이프사이클 이벤트 방출.

## Key Components

### 1) System Prompt 구성

워크스페이스 주입 파일 8종을 `loadWorkspaceBootstrapFiles`(`src/agents/workspace.ts:1084`)로 로드. 세션 타입별 필터: subagent→`{AGENTS,TOOLS}`, cron→`{AGENTS,TOOLS,SOUL,IDENTITY,USER}`, main→전체. 결정론적 순서 `CONTEXT_FILE_ORDER`(`system-prompt.ts:69`: agents=10→soul=20→identity=30→user=40→tools=50→bootstrap=60→memory=70). HEARTBEAT.md만 캐시 경계 아래 동적 영역.

**마스터 어셈블러**: `buildAgentSystemPrompt`(`src/agents/system-prompt.ts:680`). 호출 계층: `buildAttemptSystemPrompt`(attempt-system-prompt.ts:39) → `buildEmbeddedSystemPrompt`(embedded-agent-runner/system-prompt.ts:22) → `buildConfiguredAgentSystemPrompt`(system-prompt-config.ts:58) → `buildAgentSystemPrompt`.

**정적 prefix(캐시 가능, `cacheStablePromptPrefix`가 sha256 키로 LRU 64)** — 대략 순서:
1. identity 라인 → 2. `## Tooling`(도구 이름+요약 목록, `coreToolSummaries` `:757`+`toolOrder` `:796` 정렬; "TOOLS.md는 사용 가이드이지 가용성 아님") → 3-4. 도구 워크플로 힌트/서브에이전트 위임/툴 콜 스타일/실행 바이어스 → 5. provider stable prefix → 6. `## Safety`/`## OpenClaw Control` → 7. `## Skills`(`<available_skills>` 카탈로그, 하단 참조) → 8. `## Memory Recall`(memory-core `buildPromptSection`가 `memory_search`/`memory_get` 가용시 인출 가이드) → 9. 모델 별칭/시간대/워크스페이스/문서/샌드박스 → 10. `# Project Context`(stable context 파일: AGENTS/SOUL/IDENTITY/USER/TOOLS/BOOTSTRAP/MEMORY.md) → **`SYSTEM_PROMPT_CACHE_BOUNDARY` 마커**(`:1266`).

**동적 suffix(캐시 불가)**: `# Dynamic Project Context`(HEARTBEAT.md) → exec 승인/authorized senders/webchat/messaging/voice/extra context/reactions/heartbeats → `## Runtime` 모델 identity 라인.

**결정론/프롬프트 캐시**: AGENTS.md 정책("Prompt cache: deterministic ordering for maps/sets/registries")을 `.toSorted()` 패턴+stable/dynamic 분리로 시행. 모델 identity는 boundary 뒤에 강제 배치(`appendModelIdentitySystemPrompt` `:646`)해 정적 prefix를 바이트 동일로 유지.

**툴 설명**: 시스템 프롬프트엔 이름+한줄 요약만. 실제 JSON 스키마는 LLM API `tools` 필드로 별도 전송(agent-core harness `AgentContext.tools`). `TOOLS.md`는 사용자 편집 환경 메모(카메라명/SSH 호스트)지 도구 문서가 아님.

### 2) Tools & Skills

**도구 카탈로그** — `createOpenClawCodingTools`(`src/agents/agent-tools.ts:384`)가 코어/쉘/채널/OpenClaw/Tool Search 도구를 조립 후 정책 파이프라인(`applyToolPolicyPipeline` `:1093`: profile→global→agent→group→sender→**sandbox**→subagent)으로 필터:
- 기본: `read/write/edit`(`sessions/tools/*`), `exec`/`process`(lazy, `agent-tools.ts:183/213`; 레거시 `bash`는 `exec`로 대체되어 스킵), `apply_patch`, `grep/find/ls`.
- OpenClaw(`tools/*`, `openclaw-tools.ts` 조립): `message`, `web_search`, `web_fetch`, `image`, `image_generate/video_generate/music_generate`, `pdf`, `tts`, `nodes`, **`cron`**(`cron-tool.ts:883`), `gateway`, `sessions_list/history/send/spawn/yield`, `subagents`, `agents_list`, `goal_*`, `transcripts`, `skill_workshop`, `heartbeat_response`, `crestodian`(ring-0).
- 플러그인: `memory_search`/`memory_get`(memory-core `tools.ts:416/735`), `wiki_*`(memory-wiki), `browser`, `canvas`, `tavily_*`, `feishu_*` 등(`api.registerTool`).
- 샌드박스 정책(README: non-main 기본 allow `bash/process/read/write/edit/sessions_*`; deny `browser/canvas/nodes/cron/gateway`): 구조적 생략(`openclaw-tools.ts:430-465`) + 정책 파이프라인 sandbox 단계로 이중 게이트.

**디스패치/훅 라이프사이클**: `agent-tool-definition-adapter.ts:359 toToolDefinitions`가 각 `execute`를 훅으로 래핑 → `runBeforeToolCallHook`(`agent-tools.before-tool-call.ts:1387`: 툴 루프 검출→스킬워크숍 승인→신뢰 정책→플러그인 `before_tool_call` 훅→승인/진단) → `tool.execute` → 결과 정규화 → `afterToolCall`(세션 `agent.afterToolCall` `agent-session.ts:533`가 이벤트 방출 + 플러그인 `after_tool_call` 훅).

**Skills** — `skills/<name>/SKILL.md`(frontmatter `name/description` 필수 + `metadata.openclaw.{requires,install,os}` + 호출 정책 필드) + 선택적 `scripts/`, `references/`, `assets/`. 발견 우선순위(workspace.ts:1229): workspace > .agents/skills(project>personal) > managed(`~/.openclaw/skills`) > bundled > extra. 이름 localeCompare 정렬, 예산 `maxSkillsInPrompt=150`/`maxSkillsPromptChars=18_000`(초과시 compact 포맷→이진탐색 접두사 절단).

**활성화**(전용 `skill` 도구 **없음**): (a) 모델이 시스템 프롬프트의 `<available_skills>` 카탈로그를 보고 `read` 도구로 `SKILL.md` 본문을 로드(버전 변경시 재읽기) — 이것이 주 경로. `read`가 스킬 파일을 가리키면 스킬 사용 텔레메트리 발화(before-tool-call 훅). (b) 사용자 슬래시 명령 `/skill <name>` 또는 `/<name>`(`discovery/chat-command-invocation.ts`); frontmatter `command-dispatch: tool` + `command-tool` 선언시 해당 도구로 직접 디스패치(`command-specs.ts:140`).

**Tools vs Skills**: 도구=JSON 스키마 프로그램 콜러블(항상 모델에 전송), 스킬=마크다운 지시문(카탈로그만 항상 가시, 본문은 on-demand `read`). `skill_workshop` 도구는 스킬 **생명주기/제안**(create/revise/apply) 전용이지 본문 주입용 아님. 배포는 ClawHub(`clawhub.ai`) 레지스트리.

### 3) Cron

- **저장소**: 공유 SQLite `state/openclaw.sqlite`, `cron_jobs` 테이블(`src/cron/store.ts`). `CronSchedule` 판별 유니언: `at`/`every`/`cron`/`on-exit`(`types.ts:9`). `computeNextRunAtMs`(`schedule.ts:55`, Croner + LRU 512 + tz). `on-exit`은 `undefined` 리턴(시간기반 타이머가 안 침).
- **스케줄러 틱**: `CronService.start`(`service/ops.ts:228`)가 중단된 `runningAtMs` 잡을 failed("interrupted by gateway restart") 처리 → `runMissedJobs` → `armTimer`(`service/timer.ts:1098`, `setTimeout` 60s clamp/2s floor). `onTimer`(`:1180`)가 잡 예약(`runningAtMs` 스탬프+저장) → 동시 워커(기본 1) → `executeJobCoreWithTimeout` → `applyJobResult`(종료상태/백오프/다음실행/delete-after-run).
- **실행 분기**(`executeJobCore` `timer.ts:1908`): main 세션→시스템 이벤트+heartbeat; detached→`payload.kind`가 `command`면 `runCronCommandJob`(`command-runner.ts`), `agentTurn`이면 `runCronIsolatedAgentTurn`. isolated agent는 `prepareCronRunContext`로 ephemeral 세션(`cron:<jobId>`) + 별도 모델/인증/스킬/타임아웃으로 embedded 루프 구동 후 `dispatchCronDelivery`.
- **전달(delivery)**: `resolveCronDeliveryPlan`(`delivery-plan.ts:61`)이 mode 정규화(announce 기본). 모델이 런 중 이미 `message` 도구로 타겟에 보냈으면 그걸 신뢰(source-delivery), 아니면 `deliverViaDirect`가 멱등키(`cron-direct-delivery:v1:...`)로 `sendDurableMessageBatch` 전송(재시도 5/10/20s, 3h 지연시 스킵). 실패 알림은 `sendFailureNotificationAnnounce`(30s 타임아웃). `heartbeat-policy.ts:15`가 heartbeat-ack 전용 출력은 가시 전달 억제.
- **에이전트 도구**: `cron` 도구(`cron-tool.ts:879`, action `status/list/get/add/update/remove/run/runs/wake`). **제약**: `payload.kind="command"`와 `schedule.kind="on-exit"`은 에이전트 도구에서 거부(`assertNoCronShellExecution`) — CLI/Gateway API 전용. **스코프**: cron 트리거로 시작한 세션에선 `cronSelfRemoveOnlyJobId`(`agent-tools.ts:554`) 설정 → 자기 잡 정리(`status/list/get/remove/runs`만, 타 잡 변이 차단). agentTurn 잡의 `toolsAllow`는 생성자의 유효 도구 allowlist로 cap.
- **cron-on-exit 설계**(`DESIGN-cron-on-exit.md`): "이 명령/프로세스가 종료되면 깨워라"를 에이전트가 직접 arm 못 하는 문제 해결. CLI 백엔드가 턴 종료시 프로세스 트리를 SIGTERM→SIGKILL 하므로 에이전트가 백그라운드 띤 프로세스도 같이 죽음. 해결: `on-exit` 스케줄 종류 + **gateway supervisor 소유 watcher**(`cron-exit-watchers.ts:64`)가 턴 라이프사이클과 무관하게 `supervisor.spawn`(`bash -lc`/`cmd.exe /d /s /c`)로 감시(24h 안전 타임아웃). 완료시 **잡을 disabled로 먼저 저장(영속) 후 발화**(fail-closed → 재시작시 이중 발화 방지). 이후 기존 cron run→시스템이벤트→전달 파이프라인 재사용.
- **게이트웨이 연결**: `createLazyGatewayCronState`(`server-cron-lazy.ts`)가 지연 로드, `startGatewayRuntimeServices`가 시작, config 리로드시 stop→rebuild→reconcile watcher, shutdown시 `cron.stop()`+`cancelAll()`.

### 4) Memory 관리 (상세)

> 핵심: OpenClaw 메모리는 **파일(Markdown)은 영구 저장, 검색은 에이전트가 필요할 때 도구로 가져오는** 구조. 미리 전부 프롬프트에 넣지 않아 token 폭발을 막는다. 상세 코드 → [memory_search](../source/git/snippets/openclaw_openclaw__memory_search.md) / [memory_extract](../source/git/snippets/openclaw_openclaw__memory_extract.md) / [bootstrap](../source/git/snippets/openclaw_openclaw__bootstrap.md) / [dreaming](../source/git/snippets/openclaw_openclaw__dreaming.md) 스니펫

#### 전체 동작 — 한눈에 보기

```
┌──────────────── 워크스페이스 파일 (영구 저장, 사용자/에이전트가 직접 편집) ────────────────┐
│                                                                                         │
│  MEMORY.md ─────── 장기 기억 (증류된 팩층, 메인 세션마다 프롬프트에 주입)                  │
│  memory/YYYY-MM-DD.md ─── 일일 노트 (원시 기록, 첫 턴에 이틀치만 주입·검색은 전체)        │
│  BOOTSTRAP.md ──── 최초 1회 identity 의식용 → 완료 후 삭제                                │
│  DREAMS.md ─────── 꿈 일기 (dreaming 출력, 프롬프트엔 안 들어감, 사람이 리뷰)              │
│  memory/.dreams/ ─── dreaming 머신 상태 (SQLite KV)                                      │
└──────┬───────────────────────────────────────┬───────────────────────────┬──────────────┘
       │                                         │                           │
  쓰기(저장)                                읽기(검색)                    통합(dreaming)
       │                                         │                           │
       ▼                                         ▼                           ▼
┌──────────────────┐              ┌──────────────────────┐      ┌─────────────────────┐
│ 파일 변경 감지    │              │ memory_search 도구    │      │ cron (매일 03:00)   │
│ (fs.watch/세션/   │              │ 호출 (모델이 필요할때)│      │                     │
│  타이머)          │              │  ↓                    │      │ light→REM→deep 3단계│
│  ↓                │              │ 벡터 KNN + 키워드 BM25│      │ deep만 MEMORY.md에  │
│ 청크(400토큰) →   │              │  → 하이브리드 퓨전    │      │  승격(append)      │
│ 임베딩 →          │              │  → 시간감쇠 → 선택     │      │ 그림자 트라이얼은    │
│ SQLite 3테이블    │              │  → tool-result 반환   │      │  report-only       │
│ (원자적)          │              │                       │      │                     │
└──────────────────┘              └──────────────────────┘      └─────────────────────┘
```

**기억 흐름 (사람처럼)**: 매일 일일 노트에 기록(daily) → dreaming이 밤에 정리해 중요한 건 장기 기억(MEMORY.md)으로 승격 → 세션 시작엔 장기기억+최근 이틀 일일노트만 프롬프트에 주입 → 더 깊은 과거는 필요할 때 검색 도구로 인출.

#### 메모리 계층 — 무엇이 어떤 용도인가

| 파일 | 비유 | 언제 프롬프트에 들어오나 | 언제 검색 인덱스에 | 자동 삭제 |
| --- | --- | --- | --- | --- |
| `MEMORY.md` | 장기 기억 (curated wisdom) | **매 세션 시작** (main만, 잘려서) | 항상 | 없음 (예산 초과시 오래된 승격섹션만 드랍) |
| `memory/YYYY-MM-DD.md` | 일일 노트 (raw diary) | **첫 턴에 이틀치만** (2,800자 한도) | 항상 (전체, 시간감쇠 적용) | **없음** (사람/에이전트가 수동 정리) |
| 세션 트랜스크립트 | 대화 원본 (raw) | 안 들어옴 | **옵인 시만** (`experimental.sessionMemory`) | 없음 (QMD만 retention 옵션) |
| `BOOTSTRAP.md` | 출생증명서 | **최초 1회만** → 완료 후 삭제 | - | 완료시 삭제 |
| `DREAMS.md` | 꿈 일기 | **안 들어옴** | 안 됨 | 없음 |

#### 쓰기(저장) — 메모리가 어떻게 기록되나

**저장 전용 도구는 없다.** 메모리는 3가지 방식으로 워크스페이스 파일에 쓰여진다:

| 누가 | 무엇을 | 언제 | 어떻게 |
| --- | --- | --- | --- |
| **① 에이전트 자발적** | daily 노트 / MEMORY.md | "remember this" 요청, heartbeat 정리 | 범용 `write`/`edit` 도구로 파일 직접 편집 (AGENTS.md가 "이렇게 기록하라"고 지시) |
| **② compaction 자동** | daily 노트 (APPEND) | 컨텍스트 압축 직전 (토큰 임계 도달) | LLM 턴이 조용히 돌아가 daily에 저장. MEMORY.md/DREAMS.md는 읽기전용 강제 |
| **③ dreaming 자동** | MEMORY.md (승격) | cron 매일 03:00 | deep 단계가 점수화된 후보를 MEMORY.md에 append |

파일이 바뀌면 → 파일 감시(`fs.watch`, 1.5s 디바운스) 또는 세션 트랜스크립트 업데이트(5s 디바운스)가 **자동으로 인덱스 재색인 트리거**. 에이전트는 "저장했다"的意识 없이 `write`만 하면 시스템이 뒤에서 색인.

**인덱스 재색인 과정**: 마크다운 → 400토큰 청크(80 오버랩) → 임베딩 provider 배치(재시도/분할/캐시 내장) → SQLite 3 테이블(본문/벡터/키워드)에 원자적 기록. provider나 설정이 바뀌면 그림자 DB에서 전체 재구축 후 원자적 교체(읽기 차단 없음). 상세 → [snippet](../source/git/snippets/openclaw_openclaw__memory_extract.md)

#### 읽기(검색) — 필요할 때 어떻게 찾나

**검색은 모델이 `memory_search` 도구를 호출할 때만 일어난다.** 자동 인출 없음. system prompt의 `## Memory Recall` 섹션이 "이전 작업 언급 전에 먼저 검색하라"고 가이드만 제공.

```
모델이 memory_search("발표 슬라이드") 호출
        ↓
① 쿼리를 임베딩 (provider, 실패시 키워드 전용으로 전환)
        ↓
② 두 경로 병렬 검색
   ├─ 벡터 KNN: 의미가 비슷한 청크 찾기 (sqlite-vec, cosine 거리)
   └─ 키워드 BM25: 단어가 겹치는 청크 찾기 (SQLite FTS5)
        ↓
③ 하이브리드 퓨전: 벡터 0.7 + 키워드 0.3 가중 합
        ↓
④ 시간 감쇠: daily 노트는 30일 half-life로 가중치 감소 (한달 전=50%)
   (MEMORY.md는 evergreen — 감쇠 없음)
        ↓
⑤ (옵션) MMR 다양성 재순위 — 비슷한 결과 중복 제거 (기본 OFF)
        ↓
⑥ 점수 상위 6개 선택 → tool-result로 모델에게 반환
```

**핵심 특징**:
- **fail-closed**: 설정된 provider가 죽으면 빈 결과 대신 에러 반환 (조용히 키워드만 쓰지 않음)
- **오래된 인덱스 안 씀**: provider/설정이 바뀌어 인덱스가 무효면 재구축까지 `[]` 반환
- **검색 범위는 기간 무관**: 이틀이든 1년이든 전체 인덱스에서 찾음 (시간감쇠만 적용)
- `corpus` 파라미터로 `memory`(기본)/`sessions`(옵인 시)/`wiki`/`all` 선택
- 상세 → [snippet](../source/git/snippets/openclaw_openclaw__memory_search.md)

#### Bootstrap — 최초 실행 identity 의식

**최초 1회만, 완료 후 삭제된다.** 매 세션마다 쓰는 게 아님.

```
최초 실행 (워크스페이스 비어있음)
  ↓ runtime이 AGENTS/SOUL/TOOLS/IDENTITY/USER/HEARTBEAT/BOOTSTRAP.md 시드
  ↓
자유형 대화로 이름·성격·바이브·이모지 결정
  ↓ IDENTITY/USER/SOUL.md에 기록
  ↓
의식 완료 → BOOTSTRAP.md 삭제 (다시 안 함, 재시드도 방어)
  ↓ 이후 세션: IDENTITY/SOUL/USER.md가 identity 역할 (매 세션 주입)
```

- 주의: "bootstrap files"라는 용어는 이 의식용 **BOOTSTRAP.md 파일**(1회 삭제)과, 매 세션 주입되는 **8개 워크스페이스 파일**(AGENTS/SOUL/.../MEMORY.md)을 **구분**해야 함
- BOOTSTRAP.md 자체는 시스템 프롬프트가 아니라 **사용자 프롬프트**로 주입 → `read` 도구를 안 쓰는 모델도 수행
- 상세 → [snippet](../source/git/snippets/openclaw_openclaw__bootstrap.md)

#### 인출 주입 — 무엇이 언제 프롬프트에 들어오나

**토큰 폭발을 막기 위해 각 경로마다 글자 수 한도가 있다.** "영속적"은 파일이 영구 존재한다는 뜻이지, 전체가 매 턴 들어간다는 뜻이 아니다:

| 무엇이 | 언제 | 어떻게 | 한도 |
| --- | --- | --- | --- |
| `MEMORY.md` | 매 세션 시작 (main만) | system prompt `# Project Context`에 자동 | 파일당 20K자 / 전체 60K자에서 잘림 |
| daily 이틀치 | `/new`·`/reset` 첫 턴만 | `[Untrusted daily memory]` 블록으로 preload | 파일당 1,200자 / 전체 2,800자 |
| `memory_search` 결과 | 모델이 도구 호출 시 | tool-result 메시지 | compaction이 통제 |
| `DREAMS.md` | **안 들어옴** | - | - |
| 나머지 daily/세션 | **안 들어옴** | 검색 인덱스에만 → on-demand | - |

**이틀치 daily preload 동작** (`/new`·`/reset` 첫 턴):
```
오늘 07-09에 /new 실행
  ↓ runtime이 memory/2026-07-09.md + 07-08.md 선택 (이틀치, tz 보정)
  ↓ 각 파일을 1,200자로 잘라 아래 블록으로 첫 프롬프트에 삽입:

[Untrusted daily memory: memory/2026-07-09.md]
BEGIN_QUOTED_NOTES
```text
- 사용자: 다음주 발표 슬라이드 정리 요청
- cron: 매일 09:00 뉴스 요약
- 약속: 금요일 14:00 김팀장 1:1
```
END_QUOTED_NOTES

[Untrusted daily memory: memory/2026-07-08.md] ...
```
- "Untrusted"인 이유: daily에 DM 입력 유래 내용이 섞일 수 있어 prompt-injection 격리 포맷
- 3일치 이전 파일은 주입 안 됨 → 필요시 `memory_search`/`read`로 가져옴
- 보통 턴엔 이 블록 자체가 없음

#### Dreaming — 밤에 기억을 정리하는 통합

**기본 비활성.** cron(매일 03:00)이 트리거해 최근 단기 기억에서 중요한 걸 찾아 장기 기억으로 승격. 사람의 수면 사이클 비유.

```
cron 03:00 → isolated agent 세션에서 시스템 이벤트 주입
  ↓
① Light (최근 2일): daily 노트 + 세션 대화를 단기 recall 저장소에 적재
   → DREAMS.md "## Light Sleep" + 1인칭 일기 작성        [MEMORY.md 안 씀]

② REM (최근 7일): light가 모은 항목에서 패턴/반복 주제 반영
   → DREAMS.md "## REM Sleep" + 일기                      [MEMORY.md 안 씀]

③ Deep: 단기 recall 후보들을 6개 신호로 점수화
   → 임계값 통과한 후보만 MEMORY.md에 "## Promoted From Short-Term Memory" append
   → DREAMS.md "## Deep Sleep" 보고서 + 일기
```

**Deep 승격 점수 (6 신호 가중합 + 3단계 게이트)**:

| 신호 | 비유 | 가중치 |
| --- | --- | --- |
| relevance | 평균 검색 품질 (자주 잘 찾혔나) | 0.30 |
| frequency | 얼마나 자주 회상됐나 | 0.24 |
| diversity | 여러 질문/여러 날에 걸쳐 나타났나 | 0.15 |
| recency | 최근인가 (14일 half-life 감쇠) | 0.15 |
| consolidation | 여러 날에 걸쳐 반복됐나 | 0.10 |
| conceptual | 컨셉 풍부성 (태그 다양성) | 0.06 |

게이트: 점수≥임계 ∩ 회상횟수≥임계 ∩ 다양성≥임계 (모두 통과해야 승격). 승격시 `MEMORY.md`가 10KB를 넘지 않게 가장 오래된 승격 섹션부터 드랍(예산 보호). 일기 자체는 승격 대상에서 제외(자기 참조 방지).

**그림자 트라이얼**: 승격 전 시험 실행(helpful/neutral/harmful 평결). **report-only** — MEMORY.md에 직접 안 씀. helpful면 순위 약간 올림.

**DREAMS.md 용도** (프롬프트엔 안 들어옴): ① dreaming이 단계별 블록/일기 작성 ② dreaming이 과거 일기 읽어 반복 방지 ③ 사람이 리뷰 ④ doctor CLI로 역방fill/리셋. 상세 → [snippet](../source/git/snippets/openclaw_openclaw__dreaming.md)

#### Memory 도구

| 도구 | 용도 | 핵심 |
| --- | --- | --- |
| `memory_search` | **검색** | `query`/`corpus`(memory·sessions·wiki·all). 모델 호출 시만. timeout 15s, 실패시 60s 쿨다운. unavailable→`disabled=true` 안내 |
| `memory_get` | 정확 발췌 | `path`/`from`/`lines`로 특정 라인 읽기 (잘림/continuation 정보 포함) |
| `read`/`write`/`edit` | **저장**(범용) | **메모리 전용 저장 도구 없음** — 에이전트는 범용 파일 도구로 직접 편집. 쓰면 자동으로 인덱스 재색인 |

**데이터 기한 요약**:
- daily 파일: **자동 삭제 없음** (영구). 시간감쇠(30일)로 검색 가중치만 감소. dreaming lookback(2/7/30일) 밖은 후보에서 빠질 뿐 삭제 아님
- 세션 트랜스크립트: 옵인 시 전체 검색 대상. `retentionDays`는 QMD 전용
- 핵심: 기간 제한은 **dreaming lookback에만**, 검색은 기간 무관(전체)


## Analysis

**장점**:
- 프레임워크 무관 agent-core 루프(`packages/agent-core`)로 provider/harness 교체 용이(ACP vs embedded). 스트리밍/도구 호출/컴팩션이 한 루프에 정리.
- 프롬프트 캐시 의식 설계(stable/dynamic 분리 + 결정론 정렬 + sha256 캐시 키)로 비용 절감.
- 메모리 builtin 엔진이 외부 의존 없이 SQLite(vec0+FTS5)로 하이브리드 검색 + 시간감쇠 + MMR + 임베딩 캐시 + 리인덱스 잠금까지 자체 구현. provider 실패시 fail-closed/폴백 정책이 명시적.
- dreaming이 "단기 recall → 다신호 점수화 → MEMORY.md 승격"을 게이트+예산 보호+자기참조 방지로 안전하게 수행. 그림자 트라이얼로 report-only 검증.
- cron-on-exit 설계(supervisor 소유 watcher + 영속 우선 발화)가 턴 라이프사이클과 프로세스 생명주기 분리 문제를 깔끔히 해결.

**단점/부채**:
- `sessions.json`/`transcript.jsonl`이 여전 파일(AGENTS.md "SQLite only" 정책과 상충, 마이그레이션 부채 명시).
- 모노레포 규모(21,800파일)로 진입 장벽 높음. `runEmbeddedAttempt`(`attempt.ts`) 단일 파일 ~3000+ 라인.
- 하이브리드 기본값(textWeight 0.3)이 BM25 정규화 후 minScore(0.35) 밑으로 떨어져 완화 패스가 필요한 케이스가 빈번해 보임(코드 주석이 이를 시사).
- MMR 기본 OFF, dreaming 기본 비활성 — 즉시 다양성/통합 효과를 보려면 설정 필요.

**적용 가능성**: 로컬 우선 에이전트 프레임워크/메모리 설계의 좋은 참조. 특히 (a) 에이전트 루프+툴 훅 라이프사이클, (b) SQLite 기반 하이브리드 검색+tTL/예산 통제 메모리, (c) cron-on-exit 같은 프로세스 생명주기 연동 패턴, (d) dreaming 승격 점수 모델이 재사용 가치 높음.

## References
- 공식(참고용): https://docs.openclaw.ai · VISION.md · AGENTS.md(루트)
- memory: `docs/concepts/{memory,memory-search,memory-builtin,memory-qmd,dreaming}.md`, `docs/reference/memory-config.md`, `docs/start/bootstrapping.md`, `docs/reference/templates/BOOTSTRAP.md`
- cron: `DESIGN-cron-on-exit.md`, `docs/automation/cron-jobs`
- 코드: `packages/agent-core/src/agent-loop.ts`, `src/agents/{system-prompt,agent-tools,agent-command}.ts`, `extensions/memory-core/src/{memory/*,dreaming*,short-term-promotion,flush-plan,memory-budget,prompt-section}.ts`
