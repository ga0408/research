# `source/paper/` 저장 규칙

논문 원본 PDF 및 분석에서 인용한 핵심 발췌를 저장하는 폴더.

---

## 1. 파일명 규칙

### 원본 PDF
```
source/paper/<논문 제목 (sanitized)>_<year>_<venue_tag>.pdf
```
- 논문의 **원래 제목** + **연도** + **venue_tag**을 파일명으로 사용. 파일 시스템에 안전하도록 sanitize.
- **Sanitize 규칙:**
  - `:` → `_` (colon)
  - ` ` (공백) → `_`
  - `/`, `\`, `?`, `*`, `"`, `<`, `>`, `|` → 제거
  - 하이픈(`-`), 알파벳, 숫자, 밑줄(`_`), 괄호 `()` 는 유지
  - 원래 대소문자 유지
- **venue_tag 규칙:**
  - 논문지/학회/저널 명칭 사용 (예: `NeurIPS`, `ICML`, `ACL`, `CVPR`)
  - arXiv preprint인 경우: 기관명이 명시되어 있으면 기관명 sanitize (예: `Meta_AI`, `Google_Research`), 명시되지 않으면 `arxiv`
  - venue_tag도 동일한 sanitize 규칙 적용
- 예: 논문 제목 `Attention is All you Need` (2017, NeurIPS)
  → `source/paper/Attention_is_All_you_Need_2017_NeurIPS.pdf`
- 예: 논문 제목 `Remember When It Matters` (2026, arXiv, Meta AI)
  → `source/paper/Remember_When_It_Matters_Proactive_Memory_Agent_for_Long-Horizon_Agents_2026_Meta_AI.pdf`
- INDEX.md의 URL 필드에 다운로드 URL(arXiv 등) 또는 로컬 상대경로 명시.
- **바이너리 파일은 서버 푸시 제한으로 인해 `.gitignore`로 제외 권장.** 원본 PDF가 필요하면 해당 URL에서 직접 다운로드.

### 핵심 발췌
```
source/paper/<논문 제목 (sanitized)>_<year>_<venue_tag>.md
```
- 논문 하나당 발췌 파일 하나만 유지. 여러 식별자로 분리하지 않음.
- 원본 PDF와 동일한 파일명(확장자만 `.md`)을 사용하여 짝을 이룸.
- sanitize 규칙은 원본 PDF와 동일 (§1 원본 PDF 참조).
- 예: 논문 제목 `Attention is All you Need` (2017, NeurIPS)
  → `source/paper/Attention_is_All_you_Need_2017_NeurIPS.pdf` (원본 PDF)
  → `source/paper/Attention_is_All_you_Need_2017_NeurIPS.md` (핵심 발췌)

## 2. 발췌 파일 내용

- 분석에 인용한 핵심 수식·문장·그림 설명·표 등을 발췌.
- 원본 PDF에 있는 내용을 텍스트로 옮겨 검색·참조하기 쉽게 정리.
- **수식 표기 시 raw LaTeX 구문(`$ ... $`, `$$ ... $$`, `\theta`, `\frac`, `\mathcal` 등) 사용 전면 금지**:
  - 인라인 변수/수식은 유니코드 수학 기호(`θ`, `Σ`, `α`, `λ`, `δ`, `Δ`, `κ`, `∈`, `≤`, `≥`, `≠`, `≈`, `→`, `←`, `⇒`, `⟨`, `⟩`, `∑`, `∏`, `·`)와 백틱(`` `θ` ``, `` `L(θ)` ``) 사용.
  - 다중 라인 수식은 코드 블록(```)으로 서식화.
- 논문의 여러 섹션 발췌가 필요한 경우, 하나의 파일 내에서 `##` 헤더로 섹션을 구분.

## 3. 발췌 파일 형식

```markdown
# <논문 제목> — 핵심 발췌

> 출처: [분석 문서](../../report/<paper문서명>.md) / 원본: [arXiv:XXXX](https://arxiv.org/abs/XXXX)
>
> (※ `paper문서명`은 `report/` 폴더의 분석 문서명. 발췌 파일명 자체는 논문 제목 기반이지만, 분석 문서로의 링크는 report 문서명을 사용)

## <섹션1 제목>
(핵심 수식·문장·그림 설명 등)

## <섹션2 제목>
(핵심 수식·문장·그림 설명 등)
```
