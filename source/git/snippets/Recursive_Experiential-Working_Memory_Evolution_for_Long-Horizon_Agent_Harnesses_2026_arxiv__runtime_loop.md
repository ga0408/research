# Recuris Runtime Invariant Loop & State Grounding — 코드 스니펫

> 출처: `src/recuris/runtime.py` / [분석 문서](../../../report/[paper][git]_Recursive_Experiential-Working_Memory_Evolution_for_Long-Horizon_Agent_Harnesses_2026_arxiv.md)

## 1. Invariant Turn Phase Flow

```python
"""TurnRuntime — the invariant main loop, dispatching bound strategies.

The ORDER of phases below is engine law:
    ground -> [turn_start deliverers] -> WM manager -> [intent_recorded deliverers]
    -> render -> draft -> [draft_ready checkers] -> commit -> [pre_write deliverers]
    -> re-ground -> sanitize -> out
"""
```

## 2. Turn Execution & Grounding Kernel (`TurnRuntime.run_turn`)

```python
    def run_turn(
        self,
        port: HarnessPort,
        state: RuntimeState,
        incoming: Any,
        write_mode: str = "oracle",
    ) -> Any:
        state.turn += 1
        kind = port.incoming_kind(incoming)

        # 1. GROUNDING (committed evidence -> ledger)
        receipts = port.extract_receipts()
        grounded_entries = self._ground(receipts, state)

        # 2. TURN_START deliverers
        self._dispatch_deliverers(
            Event.TURN_START.value,
            TriggerContext(event=Event.TURN_START.value, turn=state.turn, incoming_kind=kind),
            state,
        )

        # 3. WM UPDATE (from user instruction / model proposal)
        if kind == "user":
            user_msg = port.user_text(incoming)
            self._update_wm_from_user(port, state, user_msg)

        # 4. INTENT_RECORDED deliverers
        self._dispatch_deliverers(
            Event.INTENT_RECORDED.value,
            TriggerContext(event=Event.INTENT_RECORDED.value, turn=state.turn),
            state,
        )

        # 5. RENDER WM status board
        wm_text = self._render_wm(state)

        # 6. DRAFT action
        draft = port.llm_draft(wm_text)

        # 7. DRAFT_READY checkers (bounce if invalid)
        draft = self._run_checkers(port, state, draft, wm_text)

        # 8. COMMIT draft
        port.commit(draft)

        # 9. PRE_WRITE deliverers (if draft contains state-changing tool call)
        tool_calls = port.draft_tool_calls(draft)
        if any(tc.name in self.write_tools for tc in tool_calls):
            self._dispatch_deliverers(
                Event.PRE_WRITE.value,
                TriggerContext(event=Event.PRE_WRITE.value, turn=state.turn, tool_calls=tool_calls),
                state,
            )

        return draft
```

## 3. Evidence Grounding Invariant (No Self-Claimed Progress)

```python
    def _ground(self, receipts: list[ToolReceipt], state: RuntimeState) -> list[str]:
        """Model cannot write its own progress. DONE is set only by the harness with
        real environment tool receipts."""
        grounded = []
        for r in receipts:
            if r.call_id in state.synthetic_ids:
                # Pre-write synthetic receipts are registered and rejected as completion evidence
                continue
            matched_entry_id = self.matcher.match(r, state.ledger)
            if matched_entry_id:
                state.ledger.mark_done(matched_entry_id, receipt=r)
                grounded.append(matched_entry_id)
        return grounded
```
