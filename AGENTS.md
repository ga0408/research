# Research Repository 가이드

외부 개발물(git repository)과 논문(paper)을 분석·정리하기 위한 연구용 저장소.
본 문서는 에이전트가 준수해야 할 작성 규칙과 폴더 구조를 정의한다.

---

## 1. 폴더 구조

```
research/
├── AGENTS.md            # 본 문서 (작성 규칙)
├── report/              # 분석 결과물 (.md) — git/paper 통합 관리
│   ├── INDEX.md         # 단일 registry: 모든 분석 문서의 메타데이터 (title, summary, keywords, topics, URL)
│   └── AGENTS.md        # 분석 문서 세부 양식
├── source/              # 분석 대상 원본 파일 저장 (PDF, 코드 아카이브 등)
│   ├── git/             # git 분석용 submodule + 핵심 코드 스니펫
│   │   ├── AGENTS.md    # source/git 저장 규칙
│   │   ├── snippets/    # 분석에서 인용한 핵심 코드 스니펫 (.md)
│   │   └── <repo>_<owner>/  # git submodule (외부 repo 원본)
│   └── paper/           # paper 분석용 원본 PDF 및 핵심 발췌
│       └── AGENTS.md    # source/paper 저장 규칙
└── topics/              # 주제별 통합 비교·종합 분석 문서
```

### 폴더별 용도

| 폴더              | 용도                                                                      |
| ----------------- | ------------------------------------------------------------------------- |
| `report/`         | 분석 결과물 (.md). git repository·논문 분석을 하나의 폴더에서 통합 관리. 하나의 대상이 paper+git 원본을 모두 가질 수 있으므로 결과물은 타입별로 분리하지 않고 단일 식별자 파일명으로 관리 |
| `report/INDEX.md` | 단일 registry. 모든 분석 문서의 메타데이터(title, summary, keywords, topics, URL)를 표로 관리. LLM은 이 파일을 먼저 읽고 filtering/grouping 후 필요한 파일만 선택적으로 읽는다 |
| `source/git/`     | git submodule(원본 repo) + 분석에서 인용한 핵심 코드 스니펫               |
| `source/git/snippets/` | 분석에 인용한 핵심 코드 스니펫 (.md). submodule 원본과 분리하여 관리 |
| `source/paper/`   | paper 원본 PDF 및 분석에서 인용한 핵심 발췌 (.md)                          |
| `topics/`         | 여러 분석 문서를 주제로 묶은 통합 비교·종합 분석 문서                     |

---

## 2. 기본 규칙

1. **분석 결과물은 `report/` 폴더에 작성한다.** git repository 분석과 논문 분석을 폴더로 분리하지 않고 하나의 폴더에서 통합 관리한다. 하나의 대상이 paper와 git 원본을 모두 가질 수 있으므로, 결과물은 단일 식별자 파일명으로 관리한다.
2. **기본 문서 양식은 Markdown (`.md`)으로 한다.**
3. **분석 대상 원본은 가급적 `source/`에 저장하고 분석 문서에서 상대경로로 링크한다.**
   - 논문: 원본 PDF를 `source/paper/`에 저장. 분석 문서의 헤더에 URL/DOI/arXiv ID도 함께 기재.
   - 외부 git repository: 원본은 URL 또는 git submodule로 참조만 하고, 분석에 인용한 핵심 코드 스니펫만 `source/git/snippets/`에 `.md` 파일로 저장한 뒤 분석 문서에서 상대경로로 링크.
   - 외부 git repository는 필요시 submodule로 추가하여 원본 중복 저장을 피한다.
   - 용량 제한: GitHub 단일 파일 100MB, 저장소 권장 1GB. PDF 논문(보통 수 MB)은 문제 없으나, 대용량 코드 아카이브는 submodule 참조 또는 Git LFS 사용을 권장.
4. **모든 분석 문서는 `report/INDEX.md`에 등록한다.**
   - 개별 분석 문서는 frontmatter 없이 `> [type] URL` 한 줄 + `# Title`로 시작한다.
   - 모든 메타데이터(title, summary, keywords, topics, source URL 등)는 `report/INDEX.md` 표에 등록하여 관리한다.
   - `topics`는 `topics/` 폴더의 주제명과 매칭되어, 주제별 통합 분석 시 선택적 읽기에서 사용된다.
   - 분석 대상 타입(git/paper/both)에 따라 `[git]`, `[paper]`, `[paper][git]` 태그를 붙인다. 상세 양식은 `report/AGENTS.md` 참조.
5. **서술형 학술 문단 작성 및 시각 자료(Figure) 임베딩 규칙**:
   - **문답식(Q&A) 작성 금지**: 분석 보고서 본문에 `Q: ... A: ...`, `네 맞습니다` 등의 문답체/대화체 구성을 사용하지 않는다. 사용자의 질문이나 주요 기술적 의문점(LLM Context 한계 연관성, RAG 패러다임 비교 등)은 보고서 본문의 서술형 학술 단락 및 비교 표로 자연스럽게 포함시킨다.
   - **시각 자료 & 직관적 비유 활용**: 비전/아키텍처 논문 분석 시 원본 PDF의 주요 Figure(실험 결과 비교, 메모리 구조도, 궤적 다이어그램 등)를 `source/paper/figures/`에 정밀 크롭하여 마크다운 이미지(`![caption](path)`)로 포함하며, 직관적인 개념 비유(예: 시계 바늘 회전)와 ASCII 다이어그램을 적극 활용한다.
6. **푸시 전 최종 검토 체크리스트 (Pre-Push Verification)**:
   - 푸시(`git push`) 직전에 에이전트는 아래 항목을 반드시 사전 점검(Check)한다:
     - [ ] `report/INDEX.md` 메타데이터 표 등록 완료 여부
     - [ ] 대화체/문답식 구문 없이 서술형 학술 문단으로 전개되었는가?
     - [ ] 핵심 시각자료(Figure 크롭 이미지)가 본문에 제대로 임베딩되었는가?
     - [ ] `source/paper/` 발췌 파일 및 `source/git/snippets/` 링크의 유효성 확인
     - [ ] 커밋 메시지 컨벤션(`paper: ...`, `git: ...`) 준수 여부
7. **사용자 질문 시 기존 분석 내용이 불충분하면 먼저 되물어보고 보강한다.**
   - 사용자가 기존 분석 문서(git/paper 무관)의 특정 항목에 대해 질문했을 때, 현재 작성된 설명만으로는 답이 불충분하거나 누락된 부분이 있다고 판단되면, **즉시 코드/논문 원본을 뒤져 답을 추측해서 작성하지 말고 먼저 사용자에게 묻는다**: "기존 설명이 부족한 것 같습니다. 해당 섹션을 (수정/추가)하시겠습니까?"
   - 사용자가 보강을 지시하면 그때 원본(git submodule / paper PDF)을 직접 읽어 정확히 확인한 뒤 해당 섹션을 수정하거나 추가 작성한다. **확인 없이 단정 짓거나 추측하지 않는다** (이전 실수 반복 방지).
   - 보강 후에는 변경분을 커밋·푸시까지 완료한다.
   - 이 규칙은 신규 분석뿐 아니라 기존 문서에 대한 후속 질문에도 동일하게 적용된다.

---

## 3. 파일명 규칙

### `report/`
```
report/[type]_<식별자>.md
```
- 분석 대상 타입에 따라 `[git]`, `[paper]`, `[paper][git]` prefix를 붙여 파일명을 정렬·구분.
- git 분석 (원본이 git repository):
  - `report/[git]_<repo>_<owner>.md` — 예: `report/[git]_langchain_langchain-ai.md`
- paper 분석 (원본이 논문):
  - `report/[paper]_<논문 제목 (sanitized)>_<year>_<venue_tag>.md` — 예: `report/[paper]_Attention_is_All_you_Need_2017_NeurIPS.md`
  - sanitize 규칙은 `source/paper/AGENTS.md` 참조 (PDF·발췌 파일명과 동일).
  - `venue_tag`: 논문지/학회 명칭(예: `NeurIPS`, `ICML`, `ACL`), arXiv preprint인 경우 기관명(예: `Meta_AI`) 또는 `arxiv`. 상세는 `source/paper/AGENTS.md` 참조.
  - arXiv 논문은 `report/[paper]_arxiv_<id>_<year>.md` 형태도 허용.
- both (원본이 paper + git 모두):
  - paper 기준 명명 사용. INDEX.md에 두 원본 URL을 모두 명시.
  - 예: `report/[paper][git]_Memora_A_Harmonic_..._2026_ICML.md` (paper 주원본, git URL도 INDEX에 명시)

### `source/git/snippets/`
```
source/git/snippets/<분석문서명>__<스니펫식별자>.md
```
- 분석 문서명은 확장자 없이, `__`(더블언더스코어)로 식별자와 구분.
- 예: `source/git/snippets/langchain_langchain-ai__memory_arch.md`

### `source/paper/`
- 논문 PDF 원본: `source/paper/<논문 제목 (sanitized)>_<year>_<venue_tag>.pdf`
  - 논문의 원래 제목 + 연도 + venue_tag을 파일명으로 사용 (sanitize 규칙은 `source/paper/AGENTS.md` 참조).
  - 예: `source/paper/Attention_is_All_you_Need_2017_NeurIPS.pdf`
  - **바이너리 파일은 서버 푸시 제한으로 인해 `.gitignore`로 제외 권장.** INDEX.md의 URL 필드에 다운로드 URL(arXiv 등) 또는 로컬 상대경로 명시.
- 핵심 발췌: `source/paper/<논문 제목 (sanitized)>_<year>_<venue_tag>.md` (논문 하나당 하나의 파일, 여러 섹션은 `##` 헤더로 구분)
  - 원본 PDF와 동일한 파일명(확장자만 `.md`)으로 짝을 이룸 (sanitize 규칙은 `source/paper/AGENTS.md` 참조).
  - 예: `source/paper/Attention_is_All_you_Need_2017_NeurIPS.md`

### 외부 git repository 원본 처리
- 외부 git repository 원본은 `source/`에 복사하지 않고 **git submodule**로 참조한다.
- submodule 추가: `git submodule add <repo URL> source/git/<repo>_<owner>`
- 우리 repo에는 메타데이터(URL + 커밋 해시)만 저장되어 용량을 거의 차지하지 않는다.
- clone 시 원본 코드가 필요하면 `git clone --recursive` 또는 `git submodule update --init` 실행.
- 분석 문서에서는 submodule 경로를 상대경로로 링크하여 원본 코드를 직접 참조.

### `topics/`
```
topics/<주제영문명>.md
```
- 예: `topics/rag_pipeline_comparison.md`

---

## 4. 분석 문서 템플릿

> git repository 및 논문 분석의 세부 양식은 `report/AGENTS.md`에서 관리한다.

### 주제 통합 분석 (`topics/`)

```markdown
> [topic]

# <주제명>

## Scope
- [report/xxx.md](../report/xxx.md)
- [report/yyy.md](../report/yyy.md)

## Comparison
| 항목 | A | B | ... |
| --- | --- | --- | --- |

## Synthesis
(종합 결론)
```

---

## 5. 키워드 인덱스

`report/INDEX.md`의 `Keywords` 컬럼을 중심으로 체계적으로 분류·검색한다.
필요시 `topics/index.md`에 키워드 → 문서 맵핑 표를 유지한다.

---

## 6. Git 운영

- 원격: https://github.com/ga0408/research.git
- 기본 브랜치: `main`
- 커밋 메시지: `분류: 간단설명` 형태.
  - 예: `git: langchain memory 아키텍처 분석 추가`
  - 예: `paper: vaswani attention 2017 분석 추가`
- 외부 git repository는 필요시 submodule로 추가하여 원본 중복 저장을 피한다.

---

## 7. 자동 분석 워크플로

사용자가 git repository URL 또는 논문 URL/PDF만 제공하면, 에이전트는 아래 워크플로를 자동으로 수행한다.

### 7-1. git repository 분석 워크플로

사용자 입력: `https://github.com/<owner>/<repo>.git` (또는 분석 지시)

1. **submodule 추가**: `git submodule add <URL> source/git/<repo>_<owner>`
2. **코드 탐색**: README + 디렉토리 구조 + 핵심 소스 파일 순으로 읽으며 구조 파악
3. **핵심 코드 스니펫 추출**: 분석에 인용할 핵심 코드를 `source/git/snippets/<분석문서명>__<식별자>.md`로 저장
4. **분석 문서 작성**: `report/[git]_<repo>_<owner>.md`에 `> [git] URL` 헤더 + 아키텍처 + 핵심 컴포넌트 + 분석 작성
   - **코드 위주 분석** (주석 설명보다 코드 구조 중심) — 단 본문은 동작·흐름(high-level) 중심, 함수명·라인 번호·상수값 나열은 최소화
   - **전체 동작·흐름을 파악할 수 있는 아키텍처 다이어그램(ASCII) 포함** — 각 섹션 시작에 전체 개요 그림
   - 핵심 알고리즘/구조는 세세하게 설명하되 **동작 의미·공식·게이트를 코드 원문보다 이해하기 쉽게** 풀어쓰기 (상세 코드는 스니펫)
   - **INDEX.md 등록**: `report/INDEX.md`의 `[git]` 섹션에 title, summary, keywords, topics, URL 등 메타데이터 추가
   - **자율형 에이전트인 경우**: `report/AGENTS.md` §4(기본 5 + 보조 6 + memory 특별 상세) 항목 구조를 따른다. memory 핵심 로직(추출·검색·bootstrap·dreaming)은 별도 스니펫으로 분리한다.
5. **커밋·푸시**: `git: <owner>/<repo> 분석 추가`

### 7-2. 논문 분석 워크플로

사용자 입력: 논문 URL (arXiv/PDF/DOI) 또는 PDF 파일

1. **원본 PDF 다운로드**: `source/paper/<논문 제목 (sanitized)>_<year>_<venue_tag>.pdf`에 저장 (sanitize 규칙은 `source/paper/AGENTS.md` 참조)
2. **PDF 읽기**: 논문 전체 읽기 (필요시 분할 읽기)
3. **핵심 발췌 추출**: 핵심 수식·방법론·실험 결과를 `source/paper/<논문 제목 (sanitized)>_<year>_<venue_tag>.md`로 저장 (원본 PDF와 동일한 파일명, 확장자만 `.md`)
4. **분석 문서 작성**: `report/[paper]_<논문 제목 (sanitized)>_<year>_<venue_tag>.md`에 `> [paper] URL` 헤더 + 분석 작성 (PDF·발췌 파일명과 동일)
   - **INDEX.md 등록**: `report/INDEX.md`의 `[paper]` 섹션에 title, summary, keywords, topics, URL 등 메타데이터 추가
5. **커밋·푸시**: `paper: <저자> <키워드> 분석 추가`

---

## 8. Daily Papers 워크플로 (HF Daily Papers 자동 요약)

Hugging Face Daily Papers를 자동으로 가져와 한국어 요약을 생성하고,
체크박스로 표시된 논문을 상세 분석하는 2단계 워크플로.

### 8-1. 폴더 구조

```
research/
├── scripts/
│   └── daily_papers.py        # HF API fetch + LLM 한국어 요약 + markdown 생성
├── .github/workflows/
│   └── daily-papers.yml       # GH Actions workflow (수동 + cron)
├── daily/                     # 일일 논문 요약 (날짜별)
│   └── 2026-07-13.md          # 예: daily/<YYYY-MM-DD>.md
├── source/paper/              # 상세 분석 시 PDF·발췌 저장 (§7-2와 동일)
└── report/                    # 상세 분석 문서 (§7-2와 동일)
```

### 8-2. 1단계: 자동 요약 (GitHub Actions)

**워크플로**: `.github/workflows/daily-papers.yml`

- **트리거**:
  - `workflow_dispatch` (수동 실행, 날짜 입력 가능)
  - `schedule: cron '0 2 * * *'` (매일 2 AM UTC = 11 AM KST, 전일 분)
- **동작**:
  1. `scripts/daily_papers.py` 실행 → HF API에서 지정 날짜 논문 목록 fetch
  2. LLM API로 한국어 요약 생성 (3~5문장, 가독성 우선)
  3. `daily/<date>.md` 파일 생성 (체크박스 포함) → **main에 직접 커밋**
  4. 동일 내용으로 **Issue 생성** (체크박스 클릭 가능)

**Secret 설정** (GitHub repo Settings → Secrets and variables → Actions):

| Secret | 설명 | 예시 |
|--------|------|------|
| `LLM_ENDPOINT_1` | Primary LLM API base URL | `https://llm1.internal.com/v1` |
| `LLM_ENDPOINT_2` | Fallback endpoint 1 (primary 실패 시) | `https://llm2.internal.com/v1` |
| `LLM_ENDPOINT_3` | Fallback endpoint 2 | `https://llm3.internal.com/v1` |
| `LLM_ENDPOINT_4` | Fallback endpoint 3 | `https://llm4.internal.com/v1` |
| `LLM_API_KEY` | API 키 (더미 가능 — endpoint가 미인증이면 임의값) | `dummy` |

**Variable 설정** (Settings → Secrets and variables → Actions → Variables):

| Variable | 설명 | 예시 |
|----------|------|------|
| `LLM_MODEL` | 모델명 (Secret이 아닌 평문 variable, 기본값 `glm-5.2`) | `glm-5.2` |

> endpoint 4개를 세미콜론으로 결합해 스크립트에 전달하며, 스크립트가 순차 시도 후 첫 성공 endpoint로 요약 생성.

**Self-hosted Runner**: Enterprise Server는 self-hosted runner 필요.
`Settings → Actions → Runners → New self-hosted runner`에서 등록.

### 8-3. daily 파일 양식

```markdown
---
date: 2026-07-13
source: Hugging Face Daily Papers
url: https://huggingface.co/papers?date=2026-07-13
paper_count: 14
---

# HF Daily Papers — 2026-07-13

> 출처: [Hugging Face Daily Papers](...) · N papers · upvotes 내림차순
> 상세 분석이 필요한 논문은 `- [x]`로 체크하세요. (Issue 본문에서 직접 클릭 가능)

---

## 1. Paper Title
- [ ] [2607.XXXXX](https://arxiv.org/abs/2607.XXXXX) · upvotes: N · 소속: ... · repo: [..](..)

한국어 요약 (3~5문장)...

---

## 2. Another Paper
- [ ] [2607.YYYYY](https://arxiv.org/abs/2607.YYYYY) · upvotes: N
...
```

### 8-4. 2단계: 상세 분석 (로컬 opencode)

사용자가 Issue에서 상세 분석할 논문을 `- [x]`로 체크한 후:

1. **체크박스 감지**: opencode이 Issue의 `- [x]` 표시된 논문을 스캔
2. **PDF 다운로드**: arXiv ID로부터 PDF 다운로드 → `source/paper/`에 저장
3. **핵심 발췌 추출**: `source/paper/<sanitized>_<year>_<venue>.md` 작성
4. **상세 분석**: `report/<sanitized>_<year>_<venue>.md` 작성 (§7-2 논문 분석 템플릿 따름)
5. **체크박스 처리 표시**: 분석 완료 후 Issue의 해당 항목에 분석 링크 추가
6. **커밋·푸시**: `paper: <저자> <키워드> 상세 분석 추가`

### 8-5. 스크립트 수동 실행

```bash
# 어제 날짜 (기본값)
python scripts/daily_papers.py

# 특정 날짜 지정
python scripts/daily_papers.py --date 2026-07-13

# LLM 없이 abstract만 사용 (테스트용)
python scripts/daily_papers.py --date 2026-07-13 --no-llm

# 환경변수 설정 (LLM 요약 시) — 세미콜론으로 다중 endpoint 지정 (failover)
export LLM_ENDPOINTS=https://llm1.internal.com/v1;https://llm2.internal.com/v1
export LLM_API_KEY=dummy
export LLM_MODEL=glm-5.2
python scripts/daily_papers.py --date 2026-07-13
```
