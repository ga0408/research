# Recuris Skill Memory Specification & Manifest — 코드 스니펫

> 출처: `src/recuris/skillmemory.py` & `docs/skill-memory-format.md` / [분석 문서](../../../report/[paper][git]_Recursive_Experiential-Working_Memory_Evolution_for_Long-Horizon_Agent_Harnesses_2026_arxiv.md)

## 1. Skill Memory Data Structure `M = (E, W, ρ, C)`

```python
@dataclass
class SkillMemory:
    name: str
    root: Path
    manifest: dict
    em: ExperientialMemory        # E: Reusable skill cards (markdown)
    wm_schema: WMSchema           # W: Ledger schema, state tracking rules
    deliverers: list[tuple[Deliverer, str]]  # ρ: Invocation trigger policy & keys
    checkers: list[tuple[Checker, str]]      # C: Verification & bounce rules
    matcher_name: str
    stay_notice: bool = False
    stay_notice_text: str = ""
    oracle: Any = None
```

## 2. Skill Memory Manifest Format (`manifest.yaml`)

```yaml
version: 1
name: tau2_retail_evolved
matcher: receipt_binding_match

wm:
  board_marker: "### WORKING MEMORY"
  match:
    binding_key: "item_id"
    success_status: ["delivered", "exchanged", "refunded"]

delivery:
  - name: call_time_skill_injector
    strategy: builtin:call_time
    at: pre_write
    key_from: tool_name
  - name: turn_boundary_injector
    strategy: builtin:boundary
    at: turn_start
    predicate: "unresolved_goals > 0"

checkers:
  - name: return_exchange_receipt_checker
    strategy: builtin:receipt_verifier
    at: draft_ready
    bounce_action: exemplar_bounce
```
