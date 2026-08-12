# Research Repository

외부 개발물(git repository)과 논문(paper)을 분석·정리하기 위한 연구용 저장소.

에이전트가 준수해야 할 작성 규칙과 폴더 구조는 [`AGENTS.md`](AGENTS.md)에 정의되어 있다.

---

## 폴더 구조

```
research/
├── AGENTS.md              # 작성 규칙 (최상위)
├── report/                # 분석 결과물 (.md) — git/paper 통합 관리
│   ├── AGENTS.md          # 분석 문서 세부 양식
│   ├── VectifyAI_PageIndex.md
│   ├── HKUDS_CatchMe.md
│   └── microsoft_Memora.md   # both 타입 (git + paper)
├── source/                # 분석 대상 원본
│   ├── git/               # git submodule + 핵심 코드 스니펫
│   │   ├── snippets/      # 인용 핵심 코드 스니펫 (.md)
│   │   ├── PageIndex/     # submodule
│   │   ├── HKUDS_CatchMe/ # submodule
│   │   └── microsoft_Memora/  # submodule
│   └── paper/             # paper 원본 PDF + 핵심 발췌 (.md)
└── topics/                # 주제별 통합 비교·종합 분석 문서
```

### 폴더별 용도

| 폴더 | 용도 |
|---|---|
| `report/` | 분석 결과물. git·paper 타입으로 분리하지 않고 단일 식별자 파일명으로 통합 관리. 하나의 대상이 paper+git 원본을 모두 가질 수 있음 (both 타입) |
| `source/git/` | git submodule(원본 repo) + 분석에서 인용한 핵심 코드 스니펫 |
| `source/paper/` | paper 원본 PDF 및 분석에서 인용한 핵심 발췌 |
| `topics/` | 여러 분석 문서를 주제로 묶은 통합 비교·종합 분석 문서 |

---

## 분석 문서 목록

| 문서 | 타입 | 대상 | 요약 |
|---|---|---|---|
| [report/VectifyAI_PageIndex.md](report/VectifyAI_PageIndex.md) | git | [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex) | 벡터 DB 없이 LLM reasoning으로 문서의 계층적 트리 인덱스를 구축하고 agent가 트리를 탐색하는 RAG 프레임워크 |
| [report/HKUDS_CatchMe.md](report/HKUDS_CatchMe.md) | git | [HKUDS/CatchMe](https://github.com/HKUDS/CatchMe) | 사용자의 디지털 활동을 실시간 캡처해 계층적 activity tree로 구성하고 LLM reasoning 기반 tree retrieval로 검색하는 개인 메모리 시스템 |
| [report/microsoft_Memora.md](report/microsoft_Memora.md) | both | [microsoft/Memora](https://github.com/microsoft/Memora) + [arXiv:2602.03315](https://arxiv.org/abs/2602.03315) | abstraction과 specificity를 분리한 "harmonic memory representation"과 cue anchor 기반 다대다 구조. LLM 기반 메모리 추출·중복갱신과 semantic/prompted/GRPO 3종 검색 정책. LoCoMo·LongMemEval SOTA |

---

## Clone

submodule 원본 코드까지 함께 받으려면 recursive clone:

```bash
git clone --recursive https://github.ecodesamsung.com/jjlee-lee/research.git
```

이미 clone한 후에는:

```bash
git submodule update --init
```

---

## 작성 규칙 요약

- 분석 결과물은 `report/<식별자>.md`에 작성. git/paper 타입으로 분리하지 않음.
- 모든 분석 문서는 YAML frontmatter 포함 (`title`, `summary`, `keywords`, `topics` 필수).
- 분석 대상 원본은 `source/`에 저장하고 분석 문서에서 상대경로로 링크.
    - git: 원본은 submodule로 참조, 핵심 코드 스니펫은 `source/git/snippets/`에 `.md` 저장.
    - paper: 원본 PDF는 `source/paper/`에 저장, 핵심 발췌는 `source/paper/<문서명>__<식별자>.md` 저장.
- 코드 위주 분석, ASCII 아키텍처 다이어램 포함.

상세 규칙과 양식은 다음 문서 참조:
- [`AGENTS.md`](AGENTS.md) — 전체 작성 규칙·폴더 구조·자동 분석 워크플로
- [`report/AGENTS.md`](report/AGENTS.md) — 분석 문서 세부 양식 (frontmatter 필드, 본문 템플릿)
- [`source/git/AGENTS.md`](source/git/AGENTS.md) — git 스니펫 저장 규칙
- [`source/paper/AGENTS.md`](source/paper/AGENTS.md) — paper 원본·발췌 저장 규칙
