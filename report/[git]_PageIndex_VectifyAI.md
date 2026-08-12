> [git] https://github.com/VectifyAI/PageIndex.git

# PageIndex: Vectorless, Reasoning-based RAG

## Overview

벡터 DB와 청킹 없이, LLM reasoning만으로 문서를 계층적 트리 인덱스로 변환하고 agent 기반 트리 서치로 검색하는 RAG 프레임워크. 전통적 vector RAG의 "similarity ≠ relevance" 문제를 해결하기 위해, 인간 전문가가 문서를 탐색하듯 LLM이 트리 구조를 따라 reasoning하며 관련 섹션을 찾는다.

## 핵심 방법론 요약

### 인덱싱 방법

**PDF 경로:**
1. TOC(목차) 존재 여부 + 페이지 번호 포함 여부를 LLM으로 탐지 → 3가지 모드 분기
   - **TOC 있음 + 페이지 번호 있음**: TOC의 논리적 페이지 번호를 offset으로 물리적 인덱스로 변환
   - **TOC 있음 + 페이지 번호 없음**: 본문 페이지 텍스트를 LLM에 주어 title-페이지 매칭
   - **TOC 없음**: 페이지 그룹을 순차적으로 LLM에 feed하여 처음부터 트리 구조 생성
2. LLM 기반 검증→수정 루프: 각 노드의 title이 기record된 페이지에 실제 존재하는지 검증, 틀리면 범위를 좁혀 재추출 (최대 3회). 정확도 60% 미만시 더 단순한 모드로 fallback
3. 대형 노드 재귀 분할: max_page(10) + max_token(20k) 초과 노드를 `process_no_toc` 모드로 재귀 분할 (병렬)
4. 노드별 요약 생성: LLM이 각 노드의 텍스트로부터 요약 생성 (병렬), 문서 전체 설명 1문장 생성

**Markdown 경로 (PDF와 완전히 다름):**
1. `#` 헤더 레벨을 regex로 추출 → 계층 구조를 직접 결정 (TOC 탐지 불필요, LLM 호출 없음)
2. 헤더 간 텍스트를 노드 text로 할당
3. (optional) thinning: token 수가 임계값(5000) 미만인 소형 노드를 부모에 병합
4. 노드별 요약 생성은 PDF 경로와 동일 (LLM 기반)

### 검색 방법

**PDF / Markdown 공통** — 동일한 3가지 도구를 LLM agent에게 제공:

1. **`get_document(doc_id)`** → 문서 메타데이터(name, description, page_count/line_count) 확인
2. **`get_document_structure(doc_id)`** → 트리 구조 JSON 반환 (**text 필드 제거**, 토큰 절약). agent가 이 트리를 "읽고" reasoning으로 관련 범위 식별
3. **`get_page_content(doc_id, pages)`** → agent가 식별한 범위의 원본 텍스트만 on-demand 가져옴
   - **PDF**: 1-indexed 물리적 페이지 번호로 검색 (`"5-7"`, `"3,8"`, `"12"`)
   - **Markdown**: `line_num` 기준으로 트리를 순회하며 해당 라인 범위의 노드 text를 반환

즉, **Markdown 검색 도구가 별도로 존재하지 않는다**. 동일한 `get_page_content` 함수 내부에서 `doc_info['type']`에 따라 PDF는 페이지 텍스트 캐시에서, Markdown은 트리 순회로 각각 다르게 처리한다.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INDEXING (트리 빌드)                          │
│                                                                 │
│  PDF / MD                                                        │
│    │                                                            │
│    ▼                                                            │
│  get_page_tokens() ── per-page (text, token_count) 리스트        │
│    │                                                            │
│    ▼                                                            │
│  ┌─────────────────────────────────┐                            │
│  │       tree_parser()             │                            │
│  │  1. check_toc() → TOC 존재?     │                            │
│  │  2. page_index_in_toc? → 번호?  │                            │
│  └────────┬───────────┬───────────┘                            │
│           │           │           │                             │
│     yes+idx    yes/no-idx    no-toc                              │
│           │           │           │                             │
│           ▼           ▼           ▼                             │
│  ┌─────────────┐ ┌──────────┐ ┌────────────┐                   │
│  │ process_toc │ │process_  │ │process_    │                   │
│  │ _with_page  │ │toc_no_   │ │no_toc      │                   │
│  │ _numbers    │ │page_nums │ │            │                   │
│  │             │ │          │ │(LLM이 처음  │                   │
│  │ offset 계산  │ │LLM 매칭   │ │부터 트리     │                   │
│  │ 논리→물리     │ │페이지별    │ │구조 생성)    │                   │
│  └──────┬──────┘ └────┬─────┘ └─────┬──────┘                   │
│         └──────┬──────┘            │                           │
│                ▼                   │                           │
│  ┌──────────────────────────┐     │                           │
│  │    meta_processor()      │◄────┘                           │
│  │  1. 빌드 (위 3 모드)        │                                 │
│  │  2. verify_toc()         │  ← LLM이 title-page 매칭 검증    │
│  │  3. fix_incorrect_toc()  │  ← 틀린 항목 범위 좁혀 재추출    │
│  │  4. accuracy < 0.6 →     │                                 │
│  │     fallback (단순 모드)   │                                 │
│  └────────────┬─────────────┘                                 │
│               ▼                                                 │
│  post_processing()                                              │
│    flat list (structure "1.1.2") → nested tree                  │
│    start_index / end_index 페이지 범위 계산                      │
│               ▼                                                 │
│  process_large_node_recursively()                               │
│    max_page(10) + max_token(20k) 초과 노드 → 재귀 분할           │
│    asyncio.gather() 병렬 처리                                   │
│               ▼                                                 │
│  Augmentation:                                                  │
│    write_node_id()     → "0001", "0002", ...                   │
│    add_node_text()     → 노드에 원본 페이지 텍스트填充           │
│    generate_summaries()→ LLM이 노드별 요약 생성 (병렬)          │
│    generate_doc_desc() → 트리 전체에서 문서 설명 생성            │
│               ▼                                                 │
│         JSON 트리 구조 (최종 결과)                               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                  RETRIEVAL (검색)                                │
│                                                                 │
│  User Question                                                  │
│    │                                                            │
│    ▼                                                            │
│  ┌───────────────────────────────────────┐                     │
│  │     LLM Agent (OpenAI Agents SDK)     │                     │
│  │  system prompt: "tree search" 가이드  │                     │
│  └───────────┬───────────┬───────────────┘                     │
│              │           │           │                          │
│              ▼           ▼           ▼                          │
│        ┌─────────┐ ┌───────────┐ ┌──────────────┐              │
│        │get_doc()│ │get_struct │ │get_page_cont │              │
│        │metadata │ │트리 인덱스 │ │원본 페이지   │              │
│        │         │ │(text제외) │ │텍스트         │              │
│        └─────────┘ └───────────┘ └──────────────┘              │
│              │           │           │                          │
│              │     1. agent가 트리 구조를 "읽고"               │
│              │        reasoning으로 관련 범위 식별              │
│              │           │                                      │
│              │     2. tight page range로 실제 텍스트 가져옴     │
│              │           │                                      │
│              └───────────┴───────────► Answer                   │
│                                                                 │
│  ※ 벡터 검색 없음. agent의 reasoning = 검색 엔진                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                  STORAGE (저장)                                 │
│                                                                 │
│  PageIndexClient (workspace 모드)                               │
│    │                                                            │
│    ├── documents: dict[doc_id → {structure, pages, meta}]       │
│    │     (in-memory, lazy-load)                                 │
│    │                                                            │
│    ├── {doc_id}.json  ← 구조 + 페이지 텍스트 (text 필드 제거)   │
│    │                                                            │
│    ├── _meta.json     ← doc_id → 경량 메타데이터 맵             │
│    │                                                            │
│    └── _ensure_doc_loaded()  ← 접근 시 JSON에서 lazy-load       │
│                                                                 │
│  구조(tree)와 페이지 텍스트를 분리 저장:                        │
│    - get_document_structure(): text 없는 트리만 (토큰 절약)     │
│    - get_page_content(): 필요한 페이지 텍스트만 on-demand       │
└─────────────────────────────────────────────────────────────────┘
```

## 인덱싱 상세 분석

### 1. PDF 파싱 → 페이지 리스트

`get_page_tokens()` (`utils.py:387`)가 PDF의 각 페이지를 `(text, token_count)` 튜플 리스트로 추출한다. PyPDF2가 기본이며 PyMuPDF도 지원한다. 모든 후속 처리는 이 `page_list`를 기준으로 동작한다.

### 2. TOC 탐지 → 3가지 모드 분기

**TOC(Table of Contents, 목차)** 란 책이나 논문 앞부분에 있는 "차례"를 의미한다. PDF 문서의 경우 앞 수십 페이지에 목차가 있는 경우가 많으며, PageIndex는 이 목차를 트리 인덱스의 출발점으로 활용한다.

`check_toc()` (`page_index.py:696`)가 앞 N페이지(기본 20)를 다음 단계로 탐지한다:

1. **TOC 존재 여부**: `find_toc_pages()`가 각 페이지 텍스트를 LLM에 주어 "이 페이지에 목차가 있는가?"를 판별한다 (`toc_detector_single_page`). 목차 페이지가 연속으로 나타나다 끊기면 TOC 페이지 리스트로 수집한다.
   - abstract, 요약, 표 목록, 그림 목록 등은 목차에서 제외 (prompt에서 명시적으로 배색).
2. **페이지 번호 포함 여부**: 목차에서 "Section 1 ... 5"처럼 페이지 번호가 적혀 있는지 LLM이 판별한다 (`detect_page_index`).

이 두 가지 판별 결과의 조합으로 3가지 모드를 선택한다:

| 모드 | TOC | 페이지 번호 | 전략 |
|------|-----|------------|------|
| `process_toc_with_page_numbers` | O | O | TOC의 논리적 번호를 offset 통해 물리적 인덱스로 변환 |
| `process_toc_no_page_numbers` | O | X | 페이지 텍스트를 LLM에 주고 title-페이지 매칭 |
| `process_no_toc` | X | - | 페이지 그룹을 LLM에 주고 처음부터 트리 생성 |

상세 오케스트레이션 → [tree_build_orchestration.md](../source/git/snippets/PageIndex_VectifyAI__tree_build_orchestration.md)

### 3. 논리→물리 페이지 매핑 (offset 계산)

`process_toc_with_page_numbers` 모드의 핵심. TOC의 논리적 페이지 번호(예: "5")와 PDF의 물리적 페이지 인덱스(예: 7) 사이의 차이를 offset으로 계산한다.

1. `toc_transformer()`: TOC 텍스트를 JSON 구조로 변환
2. `toc_index_extractor()`: 본문 앞부분 페이지를 LLM에 주어, 각 섹션 title이 어느 physical_index에서 시작하는지 추출
3. `extract_matching_page_pairs()`: 같은 title의 (논리 번호, 물리 인덱스) pair 찾기
4. `calculate_page_offset()`: pair들의 차이(physical - logical) 중 최빈값을 offset으로 선택
5. `add_page_offset_to_toc_json()`: 전체 TOC에 offset 적용

상세 코드 → [page_offset_and_grouping.md](../source/git/snippets/PageIndex_VectifyAI__page_offset_and_grouping.md)

### 4. 페이지 그룹화 (컨텍스트 창 대응)

`page_list_to_group_text()` (`page_index.py:426`)가 페이지들을 max_tokens(기본 20000) 이하의 그룹으로 병합한다. 그룹 경계에 overlap_page=1을 두어 문맥 단절을 방지한다. `process_no_toc` 모드는 이 그룹을 순차적으로 LLM에 feed하여 트리를 증분 구축한다 (`generate_toc_init` → `generate_toc_continue`).

### 5. 검증·수정 루프

`verify_toc()` (`page_index.py:900`)가 트리의 각 노드에 대해, 해당 title이 기록된 physical_index 페이지에 실제로 존재하는지 LLM으로 검증한다. 실패한 항목은 `fix_incorrect_toc()`이 이전/다음 정답 항목 사이로 검색 범위를 좁혀 재추출하고, 재검증한다 (최대 3회). 정확도가 60% 미만이면 더 단순한 모드로 fallback.

### 6. 트리 구성 + 대형 노드 재귀 분할

`post_processing()` (`utils.py:433`)이 flat list의 structure 인덱스("1", "1.1", "1.2")를 기반으로 nested tree를 구성하고, 각 노드의 `start_index`/`end_index` 페이지 범위를 계산한다.

`process_large_node_recursively()` (`page_index.py:1000`)가 max_page_num_each_node(10) AND max_token_num_each_node(20000)을 초과하는 노드를 발견하면, 해당 노드의 페이지 범위에 대해 `meta_processor`를 `process_no_toc` 모드로 재실행하여 자식 트리를 추출한다. `asyncio.gather()`로 자식 노드들을 병렬 처리한다.

### 7. 증강 (Node ID, Text, Summary)

| 단계 | 함수 | 동작 |
|------|------|------|
| node_id | `write_node_id()` | DFS 순서로 "0001", "0002" 부여 |
| text | `add_node_text()` | start/end_index 범위의 원본 페이지 텍스트를 노드에填充 |
| summary | `generate_summaries_for_structure()` | LLM이 노드별 요약 생성 (`asyncio.gather` 병렬) |
| doc_description | `generate_doc_description()` | 트리 전체 구조에서 1문장 문서 설명 생성 |

## 검색 상세 분석

PageIndex의 검색은 **3개의 tool function을 LLM agent에게 제공**하여, agent가 reasoning으로 트리를 탐색하는 방식이다. 벡터 유사도 검색이 없다.

### 검색 도구 3종 (`retrieve.py`)

| 도구 | 입력 | 출력 | 용도 |
|------|------|------|------|
| `get_document(doc_id)` | doc_id | 메타데이터 JSON (name, description, page_count) | 문서 상태 확인 |
| `get_document_structure(doc_id)` | doc_id | 트리 구조 JSON (**text 필드 제거**, 토큰 절약) | agent가 reasoning할 인덱스 |
| `get_page_content(doc_id, pages)` | "5-7" / "3,8" / "12" | 해당 페이지 원본 텍스트 | 실제 내용 가져오기 |

### 에이전트 검색 흐름 (`agentic_vectorless_rag_demo.py`)

1. `get_document()` → 문서 메타데이터 확인
2. `get_document_structure()` → 트리 구조(text 없음) 받아서 agent가 "읽음"
3. agent가 질문과 트리를 비교하여 reasoning → 관련 `start_index`/`end_index` 범위 식별
4. `get_page_content("5-7")` → tight range로 필요한 페이지만 가져옴
5. (필요시 반복) 추가 범위 탐색
6. 도구 출력만으로 답변 생성

이것이 "tree search"이다. agent가 인간처럼 목차를 보고 해당 섹션을 찾아가는 과정을 LLM reasoning으로 구현.

`get_document_structure`가 text를 제거하는 이유: 전체 트리를 agent에게 보내야 하므로, text가 있으면 토큰이 폭발한다. 구조(title, node_id, start/end_index, summary)만 보내고, 필요한 페이지 텍스트는 `get_page_content`로 on-demand 가져온다.

상세 코드 → [retrieval_and_storage.md](../source/git/snippets/PageIndex_VectifyAI__retrieval_and_storage.md)

## 저장 상세 분석

### in-memory + workspace 영속화

`PageIndexClient` (`client.py`)는 두 가지 모드로 동작한다:

- **workspace=None**: in-memory dict만 사용 (프로세스 종료시 소멸)
- **workspace=Path**: 디스크 영속화 + lazy-load

### workspace 모드 저장 구조

```
workspace/
├── {doc_id}.json     # 전체 문서 (structure + pages + meta)
├── {doc_id}.json     # ...
└── _meta.json        # { doc_id: {type, doc_name, doc_description, path, page_count} }
```

### 저장 최적화 전략

1. **text 필드 제거**: `structure`에서 `text` 필드를 제거하고 저장. PDF의 경우 `pages` 캐시가 원본 텍스트를 보유하고 있어 중복 제거.
2. **lazy-load**: 저장 직후 in-memory에서 `structure`, `pages`를 pop. 접근 시 `_ensure_doc_loaded()`가 JSON에서 다시 로드.
3. **_meta.json 분리**: 전체 문서를 로드하지 않고도 문서 목록을 조회 가능. 문서가 많아질 때 스캔 비용 절감.

### Markdown 문서 저장

MD는 라인 기반 인덱싱. `line_num`이 PDF의 `page` 역할을 한다. `get_page_content`가 `line_num` 범위로 노드 텍스트를 찾아 반환.

## Markdown 경로 분석

`md_to_tree()` (`page_index_md.py:243`)는 `#` 헤더 레벨로 계층 구조를 직접 결정한다. LLM 기반 TOC 탐지가 필요 없다.

1. `extract_nodes_from_markdown()`: `#` 헤더를 regex로 추출 (코드 블록 내 `#` 제외)
2. `extract_node_text_content()`: 헤더 간 텍스트를 노드 text로 할당
3. `build_tree_from_nodes()`: stack 기반으로 level 기반 트리 구성
4. (optional) `tree_thinning_for_index()`: token 수가 임계값(5000) 미만인 노드를 부모에 병합
5. summary/doc_description 생성은 PDF 경로와 동일

## Key Components

| 컴포넌트 | 파일 | 역할 |
|---------|------|------|
| `page_index_main()` | `page_index.py:1066` | PDF 인덱싱 전체 파이프라인 진입점 |
| `tree_parser()` | `page_index.py:1029` | TOC 탐지 → 모드 분기 → 후처리 → 재귀 분할 |
| `meta_processor()` | `page_index.py:959` | 빌드 → 검증 → 수정 루프, fallback |
| `process_large_node_recursively()` | `page_index.py:1000` | 대형 노드 재귀 분할 (병렬) |
| `page_list_to_group_text()` | `page_index.py:426` | 토큰 한도 기반 페이지 그룹화 |
| `calculate_page_offset()` | `page_index.py:394` | 논리→물리 페이지 offset 계산 (최빈값) |
| `verify_toc()` / `fix_incorrect_toc()` | `page_index.py:900/760` | LLM 기반 검증·수정 루프 |
| `PageIndexClient` | `client.py:28` | 인덱싱 + 검색 + 워크스페이스 영속화 |
| `get_document_structure()` | `retrieve.py:100` | text 제외 트리 반환 (agent reasoning용) |
| `get_page_content()` | `retrieve.py:110` | on-demand 페이지 텍스트 반환 |
| `md_to_tree()` | `page_index_md.py:243` | Markdown 헤더 기반 트리 빌드 |
| `llm_completion()` / `llm_acompletion()` | `utils.py:32/62` | LiteLLM 기반 LLM 호출 (동기/비동기) |

## Analysis

### 장점

- **벡터 DB 불필요**: 임베딩 모델, 벡터 스토어, 청킹 전략이 모두 불필요. 인프라가 단순.
- **추적 가능성**: agent의 tool call 흐름이 검색 경로를 그대로 보여줌. "왜 이 섹션이 선택되었는가"가 reasoning으로 설명 가능.
- **문맥 인식 검색**: 대화 이력이나 도메인 지식이 agent context에 자연스럽게 반영. vector 검색은 query embedding만으로 검색하므로 문맥 반영이 어려움.
- **점진적 정제**: verify → fix → fallback 구조로 인덱스 품질을 자동 보정.

### 단점

- **LLM 호출 비용**: 인덱싱 시 LLM 호출이 매우 많음 (TOC 탐지, 구조 생성, 페이지 매칭, 검증, 수정, 요약 등). 페이지 수에 비례.
- **인덱싱 속도**: LLM 호출이 병목. 수백 페이지 문서는 수십 분 소요 가능.
- **검색 속도**: agent가 여러 번 tool call → 응답을 reasoning하는 과정이 vector 검색보다 느림.
- **PDF 파싱 한계**: 표준 PyPDF2/PyMuPDF는 복잡한 PDF(스캔, 표, 이미지)에 취약. cloud 서비스(enhanced OCR)가 별도 존재하는 이유.

### 적용 가능성

- 긴 전문 문서(재무보고서, 법률문서, 기술매뉴얼)의 정밀 QA에 적합
- 검색 정확도가 검색 속도/비용보다 중요한 도메인에 적합
- 실시간 대용량 검색보다는, 사전 인덱싱 후 정밀 질의 응답 시나리오에 적합
- 대안: PageIndex File System(블로그 언급)으로 다중 문서 스케일링, ConDB로 KV-cache 기반 스케일링 가능

## References

- [PageIndex Framework 블로그](https://pageindex.ai/blog/pageindex-intro)
- [FinanceBench 98.7% 정확도](https://github.com/VectifyAI/Mafin2.5-FinanceBench)
- [Agentic Vectorless RAG 예제](https://github.com/VectifyAI/PageIndex/blob/main/examples/agentic_vectorless_rag_demo.py)
- [PageIndex File System (다중 문서 스케일링)](https://pageindex.ai/blog/pageindex-filesystem)
