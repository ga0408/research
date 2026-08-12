# `source/git/` 저장 규칙

git 분석에서 인용한 핵심 코드 스니펫을 저장하는 폴더.

---

## 1. 파일명 규칙

```
source/git/snippets/<분석문서명>__<스니펫식별자>.md
```
- 분석 문서명은 확장자 없이, `__`(더블언더스코어)로 식별자와 구분.
- 분석 문서명은 `git/` 폴더의 분석 문서명과 일치.
- 예: `source/git/snippets/langchain_langchain-ai__memory_arch.md`

## 2. 파일 내용

- 분석에 인용한 핵심 코드 원문 + 간략한 설명.
- 외부 repo 전체를 저장하지 않고, 분석에 필요한 핵심 부분만 발췌.
- 원본 repo는 submodule로 참조 (→ `git/AGENTS.md` 참조).

## 3. 파일 형식

```markdown
# <스니펫 식별자>

> 출처: [분석 문서](../../../report/<repo>_<owner>.md) / submodule 경로

## 설명
(해당 코드의 역할·중요성 간략 설명)

## 코드
\`\`\`<language>
(코드 원문)
\`\`\`
```
