# `report/` 분석 문서 양식

git repository 분석과 논문 분석 결과물의 통합 작성 규칙과 템플릿을 정의한다.
하나의 대상이 paper와 git 원본을 모두 가질 수 있으므로, 결과물은 타입별로 분리하지 않고 단일 식별자 파일명으로 관리한다.

---

## 1. 파일명 규칙

```
report/[type]_<식별자>.md
```
- 분석 대상 타입에 따라 `[git]`, `[paper]`, `[paper][git]` prefix를 붙여 파일명을 정렬·구분.
- **git 분석** (원본이 git repository):
  - `report/[git]_<repo>_<owner>.md` — 예: `report/[git]_langchain_langchain-ai.md`
- **paper 분석** (원본이 논문):
  - `report/[paper]_<논문 제목 (sanitized)>_<year>_<venue_tag>.md` — 예: `report/[paper]_Attention_is_All_you_Need_2017_NeurIPS.md`
  - sanitize 규칙은 `source/paper/AGENTS.md` 참조 (PDF·발췌 파일명과 동일).
  - `venue_tag`: 논문지/학회 명칭(예: `NeurIPS`, `ICML`, `ACL`), arXiv preprint인 경우 기관명(예: `Meta_AI`) 또는 `arxiv`. 상세는 `source/paper/AGENTS.md` 참조.
  - arXiv 논문은 `report/[paper]_arxiv_<id>_<year>.md` 형태도 허용.
- **both** (원본이 paper + git 모두):
  - paper 기준 명명 사용. INDEX.md에 두 원본 URL을 모두 명시.
  - 예: `report/[paper][git]_Memora_A_Harmonic_..._2026_ICML.md` (paper 주원본, git URL도 INDEX에 명시)

---

## 2. INDEX.md 등록

개별 분석 문서는 frontmatter 없이 `> [type] URL` 한 줄 + `# Title`로 시작한다.
모든 메타데이터는 `report/INDEX.md` 표에 등록하여 관리한다.

### 개별 문서 헤더 양식

```markdown
> [git] https://github.com/<owner>/<repo>.git

# <Repo 명>
```

```markdown
> [paper] https://arxiv.org/abs/<id>

# <논문 제목>
```

```markdown
> [paper][git] https://github.com/<owner>/<repo>.git · https://arxiv.org/abs/<id>

# <제목>
```

### INDEX.md 표 형식

`report/INDEX.md`는 타입별로 섹션을 나누어 표로 관리한다. 상세는 `report/INDEX.md` 참조.

**공통 컬럼**: Title, Summary, Keywords, Topics, File

| 타입 | 추가 컬럼 |
|------|-----------|
| [git] | Owner, Repo, Lang, URL |
| [paper] | Year, Venue, URL |
| [paper][git] | Owner, Repo, Year, Venue, Git URL, Paper URL |

### 메타데이터 필드 설명

| 필드 | 필수 | 설명 |
|------|------|------|
| `title` | O | Repository 명 또는 논문 제목 |
| `summary` | O | 한 줄 요약 |
| `keywords` | O | 키워드 리스트 (inline 배열, 영문 소문자 권장). 분류·검색 인덱스로 사용 |
| `topics` | O | 관련 주제 태그 리스트 (inline 배열). `topics/` 폴더의 주제명과 매칭 |
| `url` | O | 원본 URL (git URL, arXiv URL, DOI 등) |
| `year` | paper/both | 출판 연도 |
| `venue` | paper/both | 게재 학회/저널 |
| `owner` | git/both | Repository owner |
| `repo` | git/both | Repository 명 |
| `language` | git/both | 주요 구현 언어 (optional) |

---

## 3. 분석 지침

> **작성 우선순위: 본문은 "무엇이 어떻게 동작하는지" high-level 흐름 중심. low-level 코드(함수명·라인 번호·상수값)는 본문에 나열하지 말고 상세는 스니펫으로 분리한다.** 사람이 한번에 파악하기 쉬운 동작 중심 문서가 목표다.

1. **코드 위주 분석** (git/both) — 주석 설명보다 코드 구조 중심으로 분석한다. 단, 본문엔 동작·흐름·개념을 설명하고, 함수명·라인 번호·상수값 나열은 최소화한다 (상세 코드는 스니펫).
2. **전체 동작·흐름을 파악할 수 있는 아키텍처 다이어그램(ASCII) 포함** — 인덱싱·검색·저장 등 핵심 파이프라인을 ASCII 다이어그램으로 표현한다. 각 섹션 시작에 전체 개요 그림을 두어 흐름이 한눈에 보이게 한다.
3. **high-level 동작 → low-level 상세 순서**로 서술. 본문은 비유·다이어그램·표로 동작을 설명하고, 코드 참조는 `path:line` 한 줄 또는 스니펫 링크로 처리한다. 긴 코드 블록·함수 시그니처·상수 표는 본문이 아닌 스니펫에 둔다.
4. **서술형 학술 문단 작성 (Q&A 문답식 금지)**: 본문에 `Q: ... A: ...`, `네 맞습니다` 등의 대화체/문답식 구성을 사용하지 않으며, 주요 기술적 의문점(LLM Context 한계 연관성, RAG 패러다임 비교 등)은 보고서 본문의 서술형 학술 단락 및 비교 표로 자연스럽게 포함시킨다.
5. **시각 자료(Figure 크롭 및 비유) 보강**: 비전/아키텍처 논문 분석 시 원본 PDF의 주요 Figure(실험 결과 비교, 메모리 구조도, 궤적 다이어그램 등)를 `source/paper/figures/`에 정밀 크롭하여 마크다운 이미지(`![caption](path)`)로 포함하며, 직관적인 개념 비유(예: 시계 바늘 회전)와 ASCII 다이어그램을 적극 활용한다.
6. **INDEX.md 등록 필수**: title, summary, keywords, topics 등 메타데이터를 INDEX.md에 등록.
7. **푸시 전 최종 검토 체크리스트 (Pre-Push Verification)**:
   - 푸시(`git push`) 직전에 에이전트는 아래 항목을 반드시 사전 점검(Check)한다:
     - [ ] `report/INDEX.md` 메타데이터 표 등록 완료 여부
     - [ ] 대화체/문답식 구문 없이 서술형 학술 문단으로 전개되었는가?
     - [ ] 핵심 시각자료(Figure 크롭 이미지)가 본문에 제대로 임베딩되었는가?
     - [ ] `source/paper/` 발췌 파일 및 `source/git/snippets/` 링크의 유효성 확인
     - [ ] 커밋 메시지 컨벤션(`paper: ...`, `git: ...`) 준수 여부

> **스타일 가이드**: 본문 한 단락에 `path:line` 참조가 3개 이상 쌓이면 스니펫으로 빼라. 표·다이어그램·비유(예: "사람의 수면 사이클처럼")를 적극 활용. 동작 순서는 번호·화살표·코드 펜스로 시각화.

---

## 4. 자율형 에이전트 분석 항목

자율형(self-directed) 에이전트 분석 시 아래 항목 구조를 사용한다. 대상이 자율형 에이전트가 아닌 일반 git repository인 경우 기본 템플릿(§5)을 따른다.

### 4-1. 기본 항목 (운영 기준)

| # | 항목 | 분석 포인트 |
| --- | --- | --- |
| 1 | 전체 동작 flow | 진입점 → 인바운드 → 에이전트 루프 → 도구 호출 → 응답 전달. ASCII 다이어그램 필수 |
| 2 | system prompt 구성 | 어셈블리 함수, ordered 섹션, 주입 파일, 캐시 경계/결정론, 툴 설명 vs API 스키마 |
| 3 | tool / skill 사용 | 도구 카탈로그(스키마+핸들러), 스킬(SKILL.md 구조·발견·활성화), sandbox 정책, 훅 라이프사이클 |
| 4 | cron / 스케줄링 | 저장소, 스케줄 문법, 틱 루프, isolated 실행, 전달(delivery), 에이전트 도구 인터페이스 |
| 5 | **memory 관리 — 특별 상세 항목** | 별도 §4-3 참조 |

### 4-2. 보조 항목 (자율성·안전성 기준)

자율형 에이전트 비교·종합 분석 시 기본 항목과 함께 다룬다.

| 항목 | 분석 포인트 |
| --- | --- |
| 목표/계획 자기생성 | 사용자 입력 없이 다음 행동을 스스로 정하는 메커니즘(heartbeat·steering·goal 도구·planner) |
| 루프 종료/제어 | 무한 루프 방지, 종료 조건, 횟수/토큰/시간 한계, 도구 루프 검출 |
| 오류 복구/자가치유 | 실패 시 재시도·페일오버·컴팩션 후 재개·재시작 복구 정책 |
| 인간개입/승인 게이트 | 위험 행동 전 승인 정책(exec approval, sandbox, before_tool_call 게이트) |
| 컨텍스트/토큰 관리 | compaction 전략, 컨텍스트 엔진, 프롬프트 캐시 — 장기 자율 실행 핵심 제약 |
| 상태 영속/복원 | 크래시 후 세션/잡 복구, SQLite 영속, 멱등성 |

### 4-3. Memory 특별 상세 항목

memory는 자율형 에이전트의 핵심이므로 기본 항목 중에서도 **특별히 세세하게** 분석한다. 단, 본문은 동작 흐름 중심(high-level)이고 low-level 코드·함수명·상수값은 스니펫으로 분리해 본문에서 링크한다.

| 하위 항목 | 분석 포인트 |
| --- | --- |
| 아키텍처 | 엔진 종류(builtin/외부), 저장 백엔드, 테이블 스키마, manager 클래스 계층, provider 체인 |
| **추출(쓰기) 로직** | sync 트리거, 청킹(토큰/오버랩), 임베딩(배치/재시도/분할/캐시), write 트랜잭션, 리인덱스(그림자DB/잠금) |
| **검색 로직** | 쿼리 임베딩 → 벡터 KNN + FTS BM25 → 하이브리드 퓨전(가중치/공식) → 시간 감쇠 → MMR(λ/유사도) → minScore/완화 패스. 동작 순서를 다이어그램·표로 설명, 가중치·임계값은 표로, 코드·상세 공식은 스니펫 |
| bootstrap 메모리 | 최초실행 identity 시드, 주입 방식(시스템 vs 사용자 프롬프트), 계층 구분(bootstrap vs episodic vs MEMORY.md), 예산 보호 |
| dreaming / 통합 | 트리거/스케줄, 페이즈(light/REM/deep), 승격 점수 모델(가중치+게이트), 그림자 트라이얼, 출력(머신/휴먼), 자기참조 방지 |
| 인출 주입 | 검색 결과가 프롬프트/컨텍스트에 어떻게 주입되는지(도구 기반 vs 자동 주입), 인용 모드 |

> memory 핵심 로직(검색 퓨전 공식, 추출 파이프라인, dreaming 점수화 등)은 `source/git/snippets/<분석문서명>__memory_*.md` 및 `__dreaming.md`, `__bootstrap.md` 등으로 분리 저장한다.

---

## 5. 본문 템플릿

### git repository 분석 (git / both)

```markdown
> [git] <repository URL>

# <Repo 명>

## Overview
(한 줄 요약. INDEX.md summary와 동일해도 무방)

## Architecture
(구조·핵심 모듈 설명. ASCII 다이어그램 포함)

## Key Components
- 컴포넌트1: 설명. 상세 스니펫 → [snippets](../source/git/snippets/<file>.md)
- 컴포넌트2: ...

## Analysis
(장점·단점·적용 가능성)

## References
- 관련 문서/논문 링크
```

### 논문 분석 (paper / both)

```markdown
> [paper] <논문 URL>

# <논문 제목>

## Summary & Outline
(abstract 요약 + 논문 전체 구조 outline. 한 줄 summary는 INDEX.md와 일치)

## Problem & Motivation
- 연구 배경: (왜 이 연구가 필요한지)
- 풀고자 하는 문제: (task 설명. **task 명칭은 영어 원어 사용 권장**, 예: retrieval-augmented generation, long-horizon task planning)
- 기존 접근의 한계: (직전 SOTA/선행 연구의 부족점)

## Contributions
- (주요 기여를 bullet list로 명시. 방법론·데이터셋·이론·실증 기여 구분 권장)

## Method
(제안 방법론 상세. 아키텍처·핵심 구조·수식·데이터 흐름. 다이어그램/표 적극 활용)
상세 발췌 → [excerpt](../source/paper/<file>.md)

## Experiments & Results
### Benchmark Datasets
- (사용한 벤치마크 데이터셋 명칭·규모·특성·왜 선택했는지. 영어 명칭 유지 권장)
### Setup
- (baseline, 평가 메트릭, 하이퍼파라미터 등)
### Results
- (핵심 결과 요약. 표/수치 포함)
### Findings & Implications
- (결과의 의의: 어느 조건에서 우수한지, 실패 케이스는 무엇인지)

## Analysis
### Strengths & Significance
- (논문의 강점과 연구적 의의)
### Limitations
- (저자가 인정한 한계 + 분석자가 발견한 한계)
### Future Work / Improvements
- (보완점 및 확장 방향)

## References
- 관련 코드/논문 링크
```

### both 타입 (git + paper)

```markdown
> [paper][git] <git URL> · <paper URL>

# <제목>
```

- 주 원본 기준 템플릿을 선택하되, 두 원본의 관계를 Overview에 명시.
- git 구조 분석과 paper 방법론 분석을 모두 포함하거나, 분석 초점에 따라 하나를 상세히 다루고 다른 하나는 참조로 링크.

### 자율형 에이전트 분석 (autonomous agent)

> 분석 대상이 자율형 에이전트인 경우 §4 항목 구조를 따르는 아래 템플릿을 사용한다.

```markdown
> [git] <repository URL>

# <Repo 명>

## Overview
(한 줄 요약)

## Architecture — 전체 동작 flow
(진입점 → 인바운드 → 에이전트 루프 → 도구 → 응답. ASCII 다이어그램 필수. 백그라운드 프로세스 표기)

## System Prompt 구성
(어셈블리 함수, ordered 섹션, 주입 파일, 캐시 경계, 툴 설명 처리)

## Tools & Skills
(도구 카탈로그 표, sandbox 정책, 디스패치/훅 라이프사이클, 스킬 구조·발견·활성화)

## Cron / 스케줄링
(저장소, 틱 루프, isolated 실행, 전달, 에이전트 도구 인터페이스)

## Memory 관리 (특별 상세)
### 아키텍처
### Memory 추출(쓰기) 로직  → [snippet](../source/git/snippets/<file>__memory_extract.md)
### Memory 검색 로직          → [snippet](../source/git/snippets/<file>__memory_search.md)
### Bootstrap 메모리          → [snippet](../source/git/snippets/<file>__bootstrap.md)
### Dreaming / 통합           → [snippet](../source/git/snippets/<file>__dreaming.md)
### 인출 주입
(추출·검색 로직은 점수 공식/상수값/알고리즘을 코드에서 인용해 세세하게)

## Autonomy & Safety (보조 항목)
- 목표/계획 자기생성:
- 루프 종료/제어:
- 오류 복구/자가치유:
- 인간개입/승인 게이트:
- 컨텍스트/토큰 관리:
- 상태 영속/복원:

## Analysis
(장점·단점·적용 가능성)

## References
```

---

## 6. 외부 원본 처리

### git repository
- 외부 git repository 원본은 `source/`에 복사하지 않고 **git submodule**로 참조한다.
- submodule 추가: `git submodule add <repo URL> source/git/<repo>_<owner>`
- clone 시 원본 코드가 필요하면 `git clone --recursive` 또는 `git submodule update --init` 실행.
- **핵심 코드 스니펫**: 분석에 인용한 핵심 코드를 `source/git/snippets/<분석문서명>__<식별자>.md`로 저장 후 본문에서 상대경로 링크.
  - 분석문서명은 type prefix를 제외한 base 식별자 사용. 예: report `[git]_openclaw_openclaw.md` → snippet `openclaw_openclaw__memory_arch.md`

### 논문
- **원본 PDF**: `source/paper/<논문 제목 (sanitized)>_<year>_<venue_tag>.pdf` (예: `source/paper/Attention_is_All_you_Need_2017_NeurIPS.pdf`) — 바이너리는 `.gitignore`로 제외 권장, INDEX.md의 URL 필드에 다운로드 URL(arXiv 등) 또는 로컬 상대경로 명시.
- **핵심 발췌**: 분석에 인용한 핵심 수식·문장·그림 설명 등을 논문 하나당 하나의 파일 `source/paper/<논문 제목 (sanitized)>_<year>_<venue_tag>.md`로 저장. 원본 PDF와 동일한 파일명(확장자만 `.md`)으로 짝을 이룸. 여러 섹션은 `##` 헤더로 구분.
