# Agent Steering (VisionController) & Working Memory

> 출처: [분석 문서](../../../report/[paper][git]_ReflectWorld-MM_An_Entity-Oriented_Multimodal_Memory_System_for_Open-Ended_Video_Streams_2026_Rightly_Robotics.md) / submodule 경로: `source/git/ReflectWorld_addxai/`

## 설명

agent steering(VisionController)과 bounded working memory의 구현. VisionController는 frame-level/segment-level 정책으로 VLM 분석 여부·풍부도를 결정하고, 제한된 consult로 agent가 perception prompt에 영향을 주는 구조. WorkingMemoryManager는 per-camera 고정 한도 단기 기억.

## VisionController — Segment 평가 + Agent Consult

`packages/reflectworld/src/vision-controller/vision-controller.ts:180`:

```typescript
evaluateSegment(result: PerceptionResult, segment: Segment): SegmentPolicyOutput {
  // ── Step 1: exit condition 확인 ──
  // agent가 설정한 exit condition 만족 시 segment policy override 해제
  if (state.segmentPolicy.override) {
    const exitMet = this.checkSegmentPolicyExit(state.segmentPolicy, signals, result, segmentEndMs);
    if (exitMet) { state.segmentPolicy.override = null; segmentPolicyCleared = true; }
  }

  // ── Step 2: segment policy rules 실행 ──
  for (const rule of this.segmentPolicyRules) {
    if (lastFired && rule.cooldown_ms && (segmentEndMs - lastFired) < rule.cooldown_ms) continue;
    if (matchCondition(rule.trigger, signals)) { /* override 갱신 */ }
  }

  // ── Step 3: segment policy override 상태 갱신 ──

  // ── Step 4: Agent consultation 확인 ──
  let shouldConsultAgent = false;
  if (this.config.segment_policy) {
    const policy = this.config.segment_policy;

    // event 전환 추적 (event_id 변경 = 이전 event 종료)
    if (currentEventId !== state.lastEventId) {
      state.eventsSinceConsult = (state.eventsSinceConsult ?? 0) + 1;
      state.lastEventId = currentEventId;
    }

    // cooldown 확인 (~1분 media time)
    const cooldownMet = (segmentEndMs - state.lastAgentConsultAt) >= policy.consultation_cooldown_seconds * 1000;

    if (cooldownMet) {
      // event budget: trigger 없어도 N events마다 강제 consult
      const budgetMet = eventBudget > 0 && (state.eventsSinceConsult ?? 0) >= eventBudget;

      // trigger conditions 확인
      let triggerReason: TriggerCondition | undefined;
      for (const trigger of policy.consultation_triggers) {
        if (matchCondition(trigger, signals)) { triggerReason = trigger; break; }
      }

      if (triggerReason || budgetMet) {
        shouldConsultAgent = true;
        // → host agent(OpenClaw)에 consult 요청, reply는 allowlist 제한
        //   (scene description, focus targets, security rules)
        //   + 명시적 exit condition 필수
        state.lastAgentConsultAt = segmentEndMs;
        state.eventsSinceConsult = 0;
      }
    }
  }
  return { firedRules, action, segmentPolicyCleared, shouldConsultAgent, consultationContext };
}
```

절대 timeout: 5분 (`ABSOLUTE_MAX_SEGMENT_POLICY_MS`).

## WorkingMemoryManager — Bounded 단기 기억

`packages/percept/src/working-memory.ts:20`:

```typescript
export interface WorkingMemoryConfig {
  event_timeout_sec: number;   // 비활성 후 event 자동 종료. 기본 30초
  entity_ttl_sec: number;      // 추적 entity 캐시 제거. 기본 300초 (5분)
  max_event_segments: number;  // 단일 event 최대 segment 수. 기본 18
}

const DEFAULT_CONFIG: WorkingMemoryConfig = {
  event_timeout_sec: 30,
  entity_ttl_sec: 300,
  max_event_segments: 18,
};
```

주요 동작:
- `assignEvent(cameraId, mode)`: event가 없으면 새로 열고, timeout(30s 비활성) 또는 max_event_segments(18) 도달 시 종료 → 새 event.
- `setEventSummary(cameraId, summary, segmentTimestamp)`: VLM이 유지하는 rolling event summary 저장 (100 words 캡).
- 최근 3 segment summary, active entities, subject targets(≤8, 연속 미검출 시 drop), entity context(TTL 5분) 관리.
- event 종료 시 `EventCloseSummary` → NarrativeMemory → episodic_schema(chapter) write.

## SemanticContextManager — Agent Consult Hook

`packages/percept/src/semantic-context.ts`:
- per-camera persistent scene profile (scene_description, known_entities, daily_routine, anomaly_baseline).
- bootstrap(15 segment 후 최초 갱신) / stable-phase(일일 + 연속 무갱신 시) 갱신.
- **100 segment마다 `onAgentConsult` callback trigger** — agent steering의 또 다른 진입점.
- Qdrant `semantic_context` collection에 persist.
