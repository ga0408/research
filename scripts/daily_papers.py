#!/usr/bin/env python3
"""
Hugging Face Daily Papers fetcher + Korean summarizer.

지정된 날짜의 HF Daily Papers를 가져와 LLM으로 한국어 요약을 생성하고
daily/<date>.md 파일을 작성한다. 각 논문에 체크박스를 포함하여,
사용자가 체크한 논문을 opencode으로 상세 분석할 수 있다.

Usage:
  python scripts/daily_papers.py --date 2026-07-13
  python scripts/daily_papers.py                  # defaults to yesterday
  python scripts/daily_papers.py --no-llm          # skip LLM, use abstracts

Environment variables (LLM 요약 시 필요):
  LLM_ENDPOINT  - OpenAI 호환 API base URL (예: https://llm.internal.com/v1)
  LLM_API_KEY   - API 키
  LLM_MODEL     - 모델명 (예: glm-5.2)
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from datetime import date, timedelta
from pathlib import Path

HF_API_URL = "https://huggingface.co/api/daily_papers"
REPO_ROOT = Path(__file__).resolve().parent.parent
DAILY_DIR = REPO_ROOT / "daily"


# ─── HF API ──────────────────────────────────────────────────────────────────

def fetch_papers_for_date(target_date: str, max_pages: int = 10) -> list:
    """
    HF API에서 페이지를 순회하며 target_date의 submittedOnDailyAt에 해당하는 논문만 수집.
    target_date보다 오래된 날짜가 등장하면 수집 중단.
    """
    found = []
    seen_ids = set()

    for page in range(max_pages):
        url = HF_API_URL if page == 0 else f"{HF_API_URL}?p={page}"
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read())
        except Exception as e:
            print(f"  [WARN] page {page} fetch 실패: {e}", file=sys.stderr)
            break

        if not data:
            break

        page_dates = set()
        for entry in data:
            paper = entry.get("paper", {})
            pid = paper.get("id")
            if not pid or pid in seen_ids:
                continue
            seen_ids.add(pid)

            daily_at = paper.get("submittedOnDailyAt", "")
            daily_date = daily_at[:10]
            page_dates.add(daily_date)

            if daily_date == target_date:
                found.append(entry)

        oldest = min(page_dates) if page_dates else None
        if oldest and oldest < target_date:
            break

        if len(data) < 50:
            break

    return found


# ─── LLM 요약 ────────────────────────────────────────────────────────────────

def _llm_chat(endpoints: list, api_key: str, model: str, messages: list,
              temperature: float = 0.3, timeout: int = 180) -> str:
    """OpenAI 호환 chat completions API 호출. endpoints를 순차 시도하며 첫 성공 응답 반환."""
    body = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }).encode()

    last_err = None
    for endpoint in endpoints:
        url = f"{endpoint.rstrip('/')}/chat/completions"
        try:
            req = urllib.request.Request(url, data=body, method="POST")
            req.add_header("Content-Type", "application/json")
            if api_key:
                req.add_header("Authorization", f"Bearer {api_key}")

            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read())
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            short = endpoint.split("//")[-1] if "//" in endpoint else endpoint
            print(f"  [WARN] endpoint 실패 ({short}): {e}", file=sys.stderr)
            continue

    raise ConnectionError(f"모든 endpoint 실패 (마지막 에러: {last_err})")


def summarize_batch(papers: list, endpoints: list, api_key: str, model: str) -> dict:
    """모든 논문을 한 번에 LLM에 보내 한국어 요약 생성. {arxiv_id: summary} 반환."""
    paper_blocks = []
    for p in papers:
        paper = p["paper"]
        block = f"[{paper['id']}] 제목: {paper.get('title','')}\n초록: {paper.get('summary','')}"
        ai = paper.get("ai_summary", "")
        if ai:
            block += f"\n(AI 참고 요약: {ai})"
        paper_blocks.append(block)

    prompt = f"""아래 논문들을 한국어로 요약하세요.

규칙:
- 논문당 3~5문장. 가독성 최우선. 필요시 더 길어도 됨.
- 첫 문장: 논문이 무엇을 하는지.
- 이후: 핵심 방법론, 주요 결과/수치.
- 기술 용어는 영어 원어 병기 (예: flow matching, dense prediction, post-training quantization).
- 번역투가 아닌 자연스러운 한국어.

논문들:
{chr(10).join(paper_blocks)}

출력: 반드시 아래 JSON 배열 형식만 출력할 것 (다른 텍스트 금지):
[
  {{"id": "<arxiv_id>", "summary": "<한국어 요약>"}},
  ...
]"""

    try:
        content = _llm_chat(endpoints, api_key, model,
                            [{"role": "user", "content": prompt}])
        m = re.search(r'\[.*\]', content, re.DOTALL)
        if m:
            arr = json.loads(m.group())
            return {item["id"]: item["summary"] for item in arr if "id" in item}
        print(f"  [WARN] JSON 파싱 실패, 응답 앞 200자: {content[:200]}", file=sys.stderr)
    except ConnectionError:
        raise
    except Exception as e:
        print(f"  [WARN] 배치 요약 실패: {e}", file=sys.stderr)
    return {}


def summarize_individual(papers: list, endpoints: list, api_key: str, model: str) -> dict:
    """개별 논문씩 LLM 호출 (배치 실패 시 fallback)."""
    summaries = {}
    for p in papers:
        paper = p["paper"]
        arxiv_id = paper["id"]
        title = paper.get("title", "")
        abstract = paper.get("summary", "")

        prompt = f"""다음 논문을 한국어로 요약하세요. 3~5문장, 가독성 최우선.
기술 용어는 영어 원어 병기. 자연스러운 한국어.

제목: {title}
초록: {abstract}

요약:"""

        try:
            content = _llm_chat(endpoints, api_key, model,
                                [{"role": "user", "content": prompt}],
                                temperature=0.3, timeout=60)
            summaries[arxiv_id] = content.strip()
        except Exception as e:
            print(f"  [WARN] {arxiv_id} 요약 실패: {e}", file=sys.stderr)
        time.sleep(0.5)

    return summaries


def summarize_papers(papers: list, use_llm: bool = True) -> dict:
    """LLM 한국어 요약 생성. {arxiv_id: summary} 반환. LLM 미설정 시 빈 dict."""
    if not use_llm:
        return {}

    # LLM_ENDPOINTS: 세미콜론 구분 다중 endpoint (failover)
    # LLM_ENDPOINT: 단일 endpoint (하위 호환)
    raw_endpoints = os.environ.get("LLM_ENDPOINTS", "") or os.environ.get("LLM_ENDPOINT", "")
    endpoints = [e.strip() for e in raw_endpoints.split(";") if e.strip()]
    api_key = os.environ.get("LLM_API_KEY", "")
    model = os.environ.get("LLM_MODEL", "")

    if not endpoints or not model:
        print("  [INFO] LLM 환경변수 미설정 — abstract/ai_summary 사용", file=sys.stderr)
        return {}

    print(f"  endpoints: {len(endpoints)}개, model={model}")
    print(f"  배치 요약 시도 ({len(papers)}편)...")
    try:
        summaries = summarize_batch(papers, endpoints, api_key, model)
    except ConnectionError:
        print("  [WARN] 모든 endpoint 실패 — abstract/ai_summary 사용", file=sys.stderr)
        return {}
    if summaries:
        print(f"  배치 성공: {len(summaries)}편 요약")
        return summaries

    # 배치는 성공했지만 일부 누락 → 개별 호출
    missing = [p for p in papers if p["paper"]["id"] not in summaries]
    if missing:
        print(f"  개별 fallback: {len(missing)}편...")
        individual = summarize_individual(missing, endpoints, api_key, model)
        summaries.update(individual)

    return summaries


# ─── Markdown 생성 ───────────────────────────────────────────────────────────

def generate_markdown(target_date: str, papers: list, summaries: dict) -> str:
    """daily/<date>.md 마크다운 본문 생성."""
    sorted_papers = sorted(papers, key=lambda p: p["paper"].get("upvotes", 0), reverse=True)

    lines = [
        "---",
        f"date: {target_date}",
        "source: Hugging Face Daily Papers",
        f"url: https://huggingface.co/papers?date={target_date}",
        f"paper_count: {len(sorted_papers)}",
        "---",
        "",
        f"# HF Daily Papers — {target_date}",
        "",
        f"> 출처: [Hugging Face Daily Papers](https://huggingface.co/papers?date={target_date}) · "
        f"{len(sorted_papers)} papers · upvotes 내림차순",
        "> 상세 분석이 필요한 논문은 `- [x]`로 체크하세요. opencode이 감지하여 PDF 다운로드 → "
        "source/paper/ 발췌 → report/ 상세 분석을 수행합니다.",
        "",
        "---",
        "",
    ]

    for i, entry in enumerate(sorted_papers, 1):
        paper = entry["paper"]
        arxiv_id = paper["id"]
        title = paper.get("title", "")
        upvotes = paper.get("upvotes", 0)
        org = paper.get("organization", {})
        org_name = org.get("fullname", "") if org else ""
        github = paper.get("githubRepo", "")
        ai_keywords = paper.get("ai_keywords", [])

        korean = summaries.get(arxiv_id, "")
        if not korean:
            ai = paper.get("ai_summary", "")
            if ai:
                korean = ai
            else:
                korean = paper.get("summary", "")[:400]

        lines.append(f"## {i}. {title}")

        meta = [f"[{arxiv_id}](https://arxiv.org/abs/{arxiv_id})",
                f"upvotes: {upvotes}"]
        if org_name:
            meta.append(f"소속: {org_name}")
        if github:
            repo_short = github.replace("https://github.com/", "")
            meta.append(f"repo: [{repo_short}]({github})")

        lines.append(f"- [ ] {' · '.join(meta)}")

        if ai_keywords:
            lines.append(f"- **키워드**: {', '.join(ai_keywords)}")

        lines.append("")
        lines.append(korean)
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="HF Daily Papers → 한국어 요약 markdown 생성")
    parser.add_argument("--date", help="날짜 (YYYY-MM-DD). 기본: 어제")
    parser.add_argument("--output", help="출력 파일 경로. 기본: daily/<date>.md")
    parser.add_argument("--no-llm", action="store_true",
                        help="LLM 요약 건너뛰기 (abstract 사용)")
    args = parser.parse_args()

    target_date = args.date or (date.today() - timedelta(days=1)).isoformat()
    print(f"날짜: {target_date}")

    print("HF Daily Papers 가져오는 중...")
    papers = fetch_papers_for_date(target_date)
    if not papers:
        print(f"[ERROR] {target_date}에 해당하는 논문이 없습니다.", file=sys.stderr)
        sys.exit(1)
    print(f"  {len(papers)}편 발견")

    print("한국어 요약 생성...")
    summaries = summarize_papers(papers, use_llm=not args.no_llm)
    print(f"  {len(summaries)}편 요약 완성")

    print("Markdown 작성...")
    md = generate_markdown(target_date, papers, summaries)

    output = Path(args.output) if args.output else DAILY_DIR / f"{target_date}.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(md, encoding="utf-8")
    print(f"  → {output}")


if __name__ == "__main__":
    main()
