# Memory Collections & Semantic Consolidation

> 출처: [분석 문서](../../../report/[paper][git]_ReflectWorld-MM_An_Entity-Oriented_Multimodal_Memory_System_for_Open-Ended_Video_Streams_2026_Rightly_Robotics.md) / submodule 경로: `source/git/ReflectWorld_addxai/`

## 설명

Memory service(Python FastAPI + Qdrant)의 핵심 구조. 9개 Qdrant collection으로 3-level episodic + semantic + procedural + ReID를 구현. semantic consolidation은 entity별 관측 5회마다 LLM이 Add/Update/Delete/NONE 결정.

## Qdrant Collection 정의

`services/mem/src/qdrant_store.py:24`:

```python
COLLECTION_SPECS: dict[str, int] = {
    # ── Episodic (3 time-scales) ──
    "episodic_trace":   1536,   # segment-level video summary
    "episodic_entity":  1536,   # segment-level per-entity observation
    "episodic_schema":  1536,   # chapter-level narrative summary
    "episodic_visual":  1024,   # video frame embedding (reserved)
    # ── Semantic (long-term) ──
    "semantic_entity":  1536,   # entity-level knowledge + identity
    "semantic_context": 1536,   # camera-level scene patterns
    # ── Procedural ──
    "procedural":       1536,   # user rules
    # ── ReID galleries ──
    "reid_face":        512,
    "reid_entity":      768,
}
```

## Importance Score (asymptotic growth)

`services/mem/src/memory_manager.py:329`:

```python
def compute_importance(old_importance: float, growth_rate: float = 0.2) -> float:
    """Asymptotic importance growth: old + (1-old) * rate, rounded to 4 decimals."""
    return round(old_importance + (1.0 - old_importance) * growth_rate, 4)
```

논문 Eq.(1) `w ← w + (1 − w)·γ`의 구현. γ=0.2 기본.

## Semantic Consolidation Gate (N=5)

`services/mem/src/memory_manager.py:79`:

```python
ENTITY_SEMANTIC_UPDATE_INTERVAL = 5

def should_update_entity_semantic_memory(observation_count: int) -> bool:
    return observation_count > 0 and observation_count % ENTITY_SEMANTIC_UPDATE_INTERVAL == 0
```

## Semantic Merge — Add/Update/Delete + Identity Protection

`services/mem/src/memory_manager.py:1400` (`_merge_entity_semantic_memory`):

```python
for resp in semantic_updates.get("semantic_memory_updates", []):
    event_type = resp.get("event", "NONE")

    if event_type == "ADD":
        # 새 패턴 → semantic_entity 생성, importance = confidence 매핑 (high=0.8, medium=0.5, low=0.3)
        mem_id = self._create_memory("semantic_entity", action_text, metadata=payload)

    elif event_type == "UPDATE":
        real_id = temp_uuid_mapping[memory_id]
        # Identity protection: skip LLM UPDATE on identity memories
        old_point = self.store.get("semantic_entity", real_id)
        if old_point and old_point["payload"].get("category") == "identity":
            logger.info("Skipping UPDATE on identity memory %s (protected).", memory_id)
            continue
        old_importance = old_point["payload"].get("importance", 0.5)
        importance = compute_importance(old_importance)  # ← asymptotic growth
        self._update_memory("semantic_entity", real_id, action_text, metadata=payload)

    elif event_type == "DELETE":
        real_id = temp_uuid_mapping[memory_id]
        # Identity protection: skip LLM DELETE on identity memories
        check = self.store.get("semantic_entity", real_id)
        if check and check["payload"].get("category") == "identity":
            logger.info("Skipping DELETE on identity memory %s (protected).", memory_id)
            continue
        self._delete_memory("semantic_entity", real_id)

    elif event_type == "NONE":
        logger.info("No semantic memory changes needed.")
```

## LLM Consolidation Prompt (Add/Update/Delete 결정 규칙)

`services/mem/src/prompts.py:11`:

```python
SEMANTIC_PATTERN_ANALYSIS_PROMPT = """You are analyzing behavioral patterns for an observed entity in a surveillance system.

Your task: Determine if the current observation, combined with historical data, reveals or updates long-term behavioral patterns worthy of semantic memory storage.

CURRENT OBSERVATION:
{current_observation}

HISTORICAL EPISODIC MEMORIES (Recent Events):
{historical_episodic_memory}

EXISTING SEMANTIC PATTERNS:
{historical_semantic_memory}

ANALYSIS GUIDELINES:
3. **Decision Rules**:
   - ADD: If you identify a NEW recurring pattern from multiple historical events + current observation
   - UPDATE: If current observation strengthens/refines an existing pattern
   - DELETE: If current observation contradicts an established pattern
   - NONE: If no patterns can be established or current observation is isolated

4. **Minimum Evidence**: Require at least 2-3 consistent observations before establishing a semantic pattern

Respond in JSON format:
{{
    "semantic_memory_updates": [
        {{
            "id": "existing_id_or_new",
            "text": "Description of the behavioral pattern",
            "event": "ADD|UPDATE|DELETE|NONE",
            "category": "behavior|appearance|interaction|location|temporal",
            "confidence": "high|medium|low",
        }}
    ]
}}"""
```

## Identity Set (최대 importance, 보호)

`services/mem/src/memory_manager.py:888`:

```python
# Identity memories are stored in semantic_entity with category='identity'
# and importance=1.0. They are protected from LLM UPDATE/DELETE.
identity_filters = {"entity_id": entity_id, "category": "identity"}
```
