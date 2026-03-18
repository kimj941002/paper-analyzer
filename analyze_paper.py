#!/usr/bin/env python3
"""
논문 분석 파이프라인 (Paper Analysis Pipeline)
=============================================
PDF에서 텍스트와 이미지를 분리 추출한 뒤,
Claude API를 통해 단계적으로 분석하는 스크립트.

사용법:
    python analyze_paper.py <pdf_path> [--model MODEL] [--output OUTPUT] [--lang LANG]

예시:
    python analyze_paper.py paper.pdf
    python analyze_paper.py paper.pdf --model claude-opus-4-6 --lang ko
    python analyze_paper.py paper.pdf --output result.md --lang en
"""

import argparse
import base64
import json
import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
import fitz  # PyMuPDF
from dotenv import load_dotenv

load_dotenv()  # .env 파일에서 환경변수 로드

# ─────────────────────────────────────────────
# 설정 (Configuration)
# ─────────────────────────────────────────────

DEFAULT_MODEL = "claude-sonnet-4-6"
MAX_OUTPUT_TOKENS = 16000          # 단계별 출력 토큰 제한
SYNTHESIS_MAX_TOKENS = 32000       # 종합 단계는 더 길게
IMAGE_MIN_SIZE = (80, 80)          # 최소 이미지 크기 (px) — 로고/아이콘 제외
IMAGE_MAX_COUNT = 30               # 최대 추출 이미지 수
IMAGE_DPI = 200                    # 이미지 렌더링 해상도
FIGURE_CONCURRENCY = 4             # Figure 병렬 분석 수

# 모델별 비용 ($/MTok) — input, output
MODEL_PRICING = {
    "sonnet": (3, 15),
    "opus":   (15, 75),
    "haiku":  (0.25, 1.25),
}

# ─────────────────────────────────────────────
# 데이터 구조
# ─────────────────────────────────────────────

@dataclass
class ExtractedImage:
    """PDF에서 추출한 이미지 정보"""
    index: int
    page_num: int
    base64_data: str
    media_type: str  # image/png, image/jpeg
    width: int
    height: int
    caption: str = ""
    context: str = ""  # 본문에서 해당 figure를 언급하는 문맥
    figure_number: str = ""  # 매칭된 Figure/Table 번호 (예: "1", "2A")


@dataclass
class ExtractionResult:
    """PDF 전처리 결과"""
    full_text: str
    images: list[ExtractedImage] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """최종 분석 결과"""
    text_analysis: str = ""
    figure_analyses: list[dict] = field(default_factory=list)
    synthesis: str = ""
    model: str = ""
    token_usage: dict = field(default_factory=lambda: {
        "input_tokens": 0, "output_tokens": 0
    })


# ─────────────────────────────────────────────
# Stage 1: PDF 전처리
# ─────────────────────────────────────────────

def extract_from_pdf(pdf_path: str, max_images: int = IMAGE_MAX_COUNT) -> ExtractionResult:
    """PDF에서 텍스트와 이미지를 분리 추출"""
    doc = fitz.open(pdf_path)
    result = ExtractionResult(full_text="", metadata={})

    # ── 메타데이터 ──
    meta = doc.metadata or {}
    result.metadata = {
        "title": meta.get("title", ""),
        "author": meta.get("author", ""),
        "pages": len(doc),
        "filename": Path(pdf_path).name,
    }

    # ── 텍스트 추출 (페이지별) ──
    page_texts = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            page_texts.append(f"[Page {page_num + 1}]\n{text}")
    result.full_text = "\n\n".join(page_texts)

    # ── 이미지 추출 ──
    images = _extract_embedded_images(doc)

    # 내장 이미지가 적으면 페이지 렌더링 방식도 시도
    if len(images) < 2:
        images = _extract_page_renders(doc)

    result.images = images[:max_images]

    # ── Figure caption 매칭 ──
    _match_captions(result)

    doc.close()
    return result


def _extract_embedded_images(doc: fitz.Document) -> list[ExtractedImage]:
    """PDF에 내장된 이미지 객체 직접 추출"""
    images = []
    seen_xrefs = set()

    for page_num, page in enumerate(doc):
        image_list = page.get_images(full=True)
        for img_info in image_list:
            xref = img_info[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            try:
                base_image = doc.extract_image(xref)
                if not base_image:
                    continue

                img_bytes = base_image["image"]
                ext = base_image.get("ext", "png")
                w = base_image.get("width", 0)
                h = base_image.get("height", 0)

                # 너무 작은 이미지 필터링 (아이콘, 로고 등)
                if w < IMAGE_MIN_SIZE[0] or h < IMAGE_MIN_SIZE[1]:
                    continue

                media_type = f"image/{'jpeg' if ext in ('jpg', 'jpeg') else 'png'}"
                b64 = base64.b64encode(img_bytes).decode("utf-8")

                images.append(ExtractedImage(
                    index=len(images) + 1,
                    page_num=page_num + 1,
                    base64_data=b64,
                    media_type=media_type,
                    width=w,
                    height=h,
                ))
            except Exception:
                continue

    return images


def _extract_page_renders(doc: fitz.Document) -> list[ExtractedImage]:
    """각 페이지를 이미지로 렌더링 (내장 이미지 추출이 실패한 경우 fallback)"""
    images = []
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        image_blocks = [b for b in blocks if b.get("type") == 1]

        if not image_blocks:
            continue

        mat = fitz.Matrix(IMAGE_DPI / 72, IMAGE_DPI / 72)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        b64 = base64.b64encode(img_bytes).decode("utf-8")

        images.append(ExtractedImage(
            index=len(images) + 1,
            page_num=page_num + 1,
            base64_data=b64,
            media_type="image/png",
            width=pix.width,
            height=pix.height,
        ))

    return images


def _get_page_text(full_text: str, page_num: int) -> str:
    """전체 텍스트에서 특정 페이지의 텍스트를 추출"""
    marker = f"[Page {page_num}]"
    start = full_text.find(marker)
    if start == -1:
        return ""
    end = full_text.find("[Page", start + 1)
    return full_text[start:end] if end != -1 else full_text[start:]


def _match_captions(result: ExtractionResult):
    """Figure/Table caption을 이미지에 매칭하고, 본문 문맥 추출

    매칭 전략: 각 이미지가 위치한 페이지(±1)에서 발견되는 캡션을 순서대로 매칭.
    캡션의 figure 번호를 이미지에 기록하여, 본문 문맥 검색 시 정확한 번호를 사용.
    """
    text = result.full_text

    # Figure/Table 캡션 패턴
    caption_pattern = re.compile(
        r'((?:Figure|Fig\.?|Table|Scheme|Chart)\s*\.?\s*(\d+[A-Za-z]?)[\s.:–—-]+[^\n]{5,})',
        re.IGNORECASE
    )
    all_captions = caption_pattern.findall(text)  # [(full_caption, fig_num), ...]

    # 본문에서 figure 언급 문맥 추출
    context_pattern = re.compile(
        r'([^.]*(?:Figure|Fig\.?|Table|Scheme|Chart)\s*\.?\s*\d+[A-Za-z]?[^.]*\.)',
        re.IGNORECASE
    )
    all_contexts = context_pattern.findall(text)

    # 이미 사용된 캡션을 추적하여 중복 매칭 방지
    used_captions = set()

    for img in result.images:
        # 해당 페이지 ± 1 범위의 텍스트에서 캡션 탐색
        nearby_text = ""
        for p in [img.page_num, img.page_num - 1, img.page_num + 1]:
            nearby_text += _get_page_text(text, p) + "\n"

        best_caption = ""
        best_fig_num = ""
        for full_cap, fig_num in all_captions:
            if fig_num in used_captions:
                continue
            if full_cap in nearby_text:
                best_caption = full_cap.strip()
                best_fig_num = fig_num
                used_captions.add(fig_num)
                break

        img.caption = best_caption
        img.figure_number = best_fig_num

        # 관련 문맥 수집 — figure_number 기반으로 정확한 매칭
        search_num = best_fig_num if best_fig_num else str(img.index)
        related_contexts = []
        for ctx in all_contexts:
            nums = re.findall(
                r'(?:Figure|Fig\.?|Table|Scheme|Chart)\s*\.?\s*(\d+)',
                ctx, re.IGNORECASE
            )
            # figure_number의 숫자 부분으로 매칭 (예: "2A" → "2")
            base_num = re.match(r'(\d+)', search_num)
            if base_num and base_num.group(1) in nums:
                related_contexts.append(ctx.strip())
        img.context = " ".join(related_contexts[:5])


# ─────────────────────────────────────────────
# Stage 2-4: Claude API 호출
# ─────────────────────────────────────────────

class PaperAnalyzer:
    """Claude API를 사용한 다단계 논문 분석"""

    def __init__(self, model: str = DEFAULT_MODEL, lang: str = "ko",
                 cache_dir: Path | None = None):
        self.client = anthropic.Anthropic()
        self.model = model
        self.lang = lang
        self.result = AnalysisResult(model=model)
        self.cache_dir = cache_dir
        self._token_lock = threading.Lock()

    def _lang_instruction(self) -> str:
        if self.lang == "ko":
            return "모든 분석 결과를 한국어로 작성하세요. 전문 용어는 영어를 병기하세요."
        elif self.lang == "en":
            return "Write all analysis results in English."
        else:
            return f"Write all analysis results in {self.lang}."

    def _call_api(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = MAX_OUTPUT_TOKENS,
    ) -> str:
        """Claude API 호출 — 스트리밍 모드 (재시도 로직 포함)"""
        for attempt in range(3):
            try:
                result_chunks: list[str] = []
                input_tokens = 0
                output_tokens = 0

                with self.client.messages.stream(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=messages,
                ) as stream:
                    for text in stream.text_stream:
                        result_chunks.append(text)

                resp = stream.get_final_message()
                input_tokens = resp.usage.input_tokens
                output_tokens = resp.usage.output_tokens

                with self._token_lock:
                    self.result.token_usage["input_tokens"] += input_tokens
                    self.result.token_usage["output_tokens"] += output_tokens

                return "".join(result_chunks)

            except anthropic.RateLimitError:
                wait = 2 ** attempt * 10
                print(f"  ⏳ Rate limit — {wait}초 대기 후 재시도...")
                time.sleep(wait)
            except anthropic.APIError as e:
                print(f"  ❌ API 오류: {e}")
                if attempt == 2:
                    raise
                time.sleep(5)

        return ""

    def _save_cache(self, stage: str, data: str | list):
        """중간 결과를 캐시 파일로 저장"""
        if not self.cache_dir:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / f"{stage}.json"
        cache_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_cache(self, stage: str) -> str | list | None:
        """캐시된 중간 결과 로드"""
        if not self.cache_dir:
            return None
        cache_file = self.cache_dir / f"{stage}.json"
        if cache_file.exists():
            content = json.loads(cache_file.read_text(encoding="utf-8"))
            print(f"  ♻️  캐시 로드: {stage}")
            return content
        return None

    # ── Stage 2: 텍스트 분석 ──

    def analyze_text(self, extraction: ExtractionResult) -> str:
        """전체 텍스트 기반 논문 구조 분석"""
        print("\n📄 [Stage 2/4] 텍스트 분석 중...")

        cached = self._load_cache("stage2_text")
        if cached:
            self.result.text_analysis = cached
            return cached

        system = f"""당신은 과학/공학 전 분야의 논문 분석 전문가입니다.
첨부된 논문 텍스트를 정밀하게 분석하세요.

{self._lang_instruction()}

다음 항목을 빠짐없이 분석하세요:

1. **논문 기본 정보**: 제목, 저자, 저널, 연도, DOI (텍스트에서 확인 가능한 것만)
2. **연구 배경 및 목적**: 해결하려는 문제, 기존 연구의 한계, 본 연구의 가설
3. **핵심 방법론**: 사용된 실험 기법, 분석 도구, 모델 시스템 (세부 조건 포함)
4. **주요 결과**: 각 실험의 핵심 발견 사항 (수치 데이터 포함)
5. **저자의 해석 및 논의**: 결과에 대한 저자의 해석, 기존 연구와의 비교
6. **결론 및 시사점**: 최종 결론, 연구의 한계, 향후 연구 방향
7. **핵심 키워드/기술 용어**: 논문에서 중요한 전문 용어 목록

중요 지침:
- 텍스트에 명시적으로 기술된 내용만 분석하세요.
- 추측이나 일반 지식 기반 보충 설명은 반드시 [추정] 또는 [일반 지식]으로 표시하세요.
- Figure/Table에 대한 분석은 별도 단계에서 수행하므로, 여기서는 본문 텍스트의
  Figure/Table 언급 내용만 기록하세요.
- 수치 데이터(IC50, Kd, 해상도 등)는 가능한 정확히 기록하세요."""

        messages = [{
            "role": "user",
            "content": f"다음은 논문의 전체 텍스트입니다:\n\n{extraction.full_text}"
        }]

        self.result.text_analysis = self._call_api(system, messages)
        self._save_cache("stage2_text", self.result.text_analysis)
        print("  ✅ 텍스트 분석 완료")
        return self.result.text_analysis

    # ── Stage 3: Figure/Table 개별 분석 (병렬) ──

    def analyze_figures(self, extraction: ExtractionResult) -> list[dict]:
        """각 Figure/Table 이미지를 병렬로 분석"""
        if not extraction.images:
            print("\n🖼️  [Stage 3/4] 추출된 이미지 없음 — 건너뜀")
            return []

        cached = self._load_cache("stage3_figures")
        if cached:
            self.result.figure_analyses = cached
            return cached

        n = len(extraction.images)
        print(f"\n🖼️  [Stage 3/4] Figure/Table 분석 중... ({n}개, 병렬 {FIGURE_CONCURRENCY})")

        system = f"""당신은 과학 논문의 Figure/Table 분석 전문가입니다.
제공되는 이미지를 정밀하게 분석하세요.

{self._lang_instruction()}

분석 항목:
1. **이미지 유형**: 그래프, 겔 이미지, 구조 모델, 현미경 사진, 표, 도식 등
2. **내용 설명**: 이미지에 표시된 모든 요소를 상세히 기술
   - 그래프: 축 라벨, 데이터 트렌드, 유의미한 수치, 통계 표시
   - 겔/블롯: 밴드 패턴, 레인 라벨, 분자량 마커
   - 구조: 단백질/분자 구조 특징, 상호작용, 결합 부위
   - 표: 모든 행/열 데이터를 정확히 기록
3. **핵심 발견**: 이 이미지가 보여주는 가장 중요한 과학적 발견
4. **세부 관찰**: 패널 구분(A, B, C 등)이 있으면 각각 분석

중요 지침:
- 이미지에서 직접 관찰할 수 있는 것만 기술하세요.
- 읽을 수 있는 텍스트/숫자는 정확히 옮기세요.
- 불확실한 부분은 [불확실] 표시를 하세요."""

        analyses = [None] * n  # 순서 보존용 슬롯

        def _analyze_one(idx: int, img: ExtractedImage) -> dict:
            label = f"Fig.{img.figure_number}" if img.figure_number else f"Image {img.index}"
            print(f"  🔍 {label} (p.{img.page_num}) 분석 중...")

            content_blocks = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": img.media_type,
                        "data": img.base64_data,
                    }
                },
                {
                    "type": "text",
                    "text": self._build_figure_prompt(img),
                }
            ]

            messages = [{"role": "user", "content": content_blocks}]
            analysis_text = self._call_api(system, messages)
            result = {
                "image_index": img.index,
                "figure_number": img.figure_number,
                "page": img.page_num,
                "caption": img.caption,
                "analysis": analysis_text,
            }
            print(f"  ✅ {label} 완료")
            return result

        with ThreadPoolExecutor(max_workers=FIGURE_CONCURRENCY) as executor:
            futures = {
                executor.submit(_analyze_one, i, img): i
                for i, img in enumerate(extraction.images)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    analyses[idx] = future.result()
                except Exception as e:
                    img = extraction.images[idx]
                    print(f"  ⚠️  Image {img.index} 분석 실패: {e}")
                    analyses[idx] = {
                        "image_index": img.index,
                        "figure_number": img.figure_number,
                        "page": img.page_num,
                        "caption": img.caption,
                        "analysis": f"[분석 실패: {e}]",
                    }

        self.result.figure_analyses = [a for a in analyses if a is not None]
        self._save_cache("stage3_figures", self.result.figure_analyses)
        return self.result.figure_analyses

    def _build_figure_prompt(self, img: ExtractedImage) -> str:
        """Figure 분석을 위한 프롬프트 구성"""
        parts = [f"이 이미지는 논문의 {img.page_num}페이지에 위치합니다."]

        if img.figure_number:
            parts.append(f"Figure/Table 번호: {img.figure_number}")

        if img.caption:
            parts.append(f"\n캡션(Caption): {img.caption}")

        if img.context:
            parts.append(f"\n본문에서 이 Figure를 언급하는 문맥:\n{img.context}")

        parts.append("\n위 이미지를 상세히 분석해주세요.")
        return "\n".join(parts)

    # ── Stage 4: 종합 ──

    def synthesize(self, extraction: ExtractionResult) -> str:
        """텍스트 분석 + Figure 분석을 종합하여 최종 리포트 생성"""
        print("\n📊 [Stage 4/4] 종합 분석 중...")

        # Figure 분석 결과를 텍스트로 조합
        figure_summary = ""
        if self.result.figure_analyses:
            parts = []
            for fa in self.result.figure_analyses:
                fig_label = f"Figure {fa.get('figure_number', '')}" if fa.get('figure_number') else f"Image {fa['image_index']}"
                header = f"### {fig_label} (p.{fa['page']})"
                if fa["caption"]:
                    header += f"\nCaption: {fa['caption']}"
                parts.append(f"{header}\n{fa['analysis']}")
            figure_summary = "\n\n".join(parts)

        system = f"""당신은 과학 논문 분석의 최종 종합 전문가입니다.
아래에 제공되는 두 가지 분석 결과를 종합하여 하나의 완성된 논문 분석 보고서를 작성하세요.

{self._lang_instruction()}

■ 출력 양식 (반드시 준수):

# [논문 제목]

## 기본 정보
(저자, 저널, 연도, 키워드 등)

## 1. 연구 배경 및 목적
### 1.1 연구 배경
### 1.2 기존 연구의 한계
### 1.3 본 연구의 목적/가설

## 2. 방법론
### 2.1 [실험 기법 1]
### 2.2 [실험 기법 2]
(각 기법의 세부 조건, 사용 장비, 분석 소프트웨어 포함)

## 3. 주요 결과
### 3.1 [결과 주제 1]
  - Figure/Table 연계: 해당 Figure의 분석 결과를 통합하여 기술
  - 핵심 수치 데이터
### 3.2 [결과 주제 2]
  ...

## 4. 고찰 (Discussion)
### 4.1 결과 해석
### 4.2 기존 연구와의 비교
### 4.3 연구의 의의

## 5. 결론 및 한계
### 5.1 최종 결론
### 5.2 연구의 한계점
### 5.3 향후 연구 방향

## 6. 핵심 요약 (3-5문장)

## 7. 비판적 검토
(방법론의 적절성, 결론의 논리적 타당성, 잠재적 한계에 대한 분석자의 의견)

---

■ 종합 시 핵심 지침:
- 텍스트 분석과 Figure 분석의 내용을 교차 검증하세요.
- 텍스트에서 언급된 수치와 Figure에서 관찰된 수치가 일치하는지 확인하세요.
- 불일치가 있으면 [⚠️ 불일치] 표시와 함께 양쪽 내용을 모두 기록하세요.
- 텍스트에만 있는 정보, Figure에만 있는 정보를 각각 놓치지 마세요.
- 확인할 수 없는 내용은 [미확인]으로 표시하세요."""

        user_content = f"""## A. 텍스트 분석 결과

{self.result.text_analysis}

## B. Figure/Table 분석 결과

{figure_summary if figure_summary else "(추출된 Figure/Table 없음)"}

위 두 분석을 종합하여 최종 논문 분석 보고서를 작성하세요."""

        messages = [{"role": "user", "content": user_content}]
        self.result.synthesis = self._call_api(
            system, messages, max_tokens=SYNTHESIS_MAX_TOKENS
        )
        self._save_cache("stage4_synthesis", self.result.synthesis)
        print("  ✅ 종합 분석 완료")
        return self.result.synthesis


# ─────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────

def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """모델명에서 가격 티어를 추출하여 비용 추정"""
    for tier, (price_in, price_out) in MODEL_PRICING.items():
        if tier in model:
            return input_tokens / 1_000_000 * price_in + output_tokens / 1_000_000 * price_out
    # 알 수 없는 모델은 sonnet 기준
    price_in, price_out = MODEL_PRICING["sonnet"]
    return input_tokens / 1_000_000 * price_in + output_tokens / 1_000_000 * price_out


def print_summary(result: AnalysisResult, extraction: ExtractionResult):
    """실행 요약 출력"""
    total_in = result.token_usage["input_tokens"]
    total_out = result.token_usage["output_tokens"]

    print("\n" + "=" * 60)
    print("📈 분석 완료 요약")
    print("=" * 60)
    print(f"  모델: {result.model}")
    print(f"  논문 페이지 수: {extraction.metadata.get('pages', '?')}")
    print(f"  추출 텍스트 길이: {len(extraction.full_text):,} 문자")
    print(f"  추출 이미지 수: {len(extraction.images)}")
    print(f"  총 입력 토큰: {total_in:,}")
    print(f"  총 출력 토큰: {total_out:,}")

    cost = _estimate_cost(result.model, total_in, total_out)
    print(f"  예상 비용: ~${cost:.3f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Claude API 기반 논문 분석 파이프라인",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python analyze_paper.py paper.pdf
  python analyze_paper.py paper.pdf --model claude-opus-4-6
  python analyze_paper.py paper.pdf --output result.md --lang en
  python analyze_paper.py paper.pdf --text-only
  python analyze_paper.py paper.pdf --resume        # 이전 중단 지점부터 재개
        """
    )
    parser.add_argument("pdf", help="분석할 PDF 파일 경로")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"사용할 Claude 모델 (기본: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="결과 저장 경로 (.md). 미지정 시 자동 생성"
    )
    parser.add_argument(
        "--lang", default="ko", choices=["ko", "en"],
        help="분석 결과 언어 (기본: ko)"
    )
    parser.add_argument(
        "--text-only", action="store_true",
        help="텍스트 분석만 수행 (Figure 분석 건너뜀)"
    )
    parser.add_argument(
        "--max-images", type=int, default=IMAGE_MAX_COUNT,
        help=f"최대 분석 이미지 수 (기본: {IMAGE_MAX_COUNT})"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="이전 중단 지점부터 재개 (캐시 활용)"
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="캐시를 사용하지 않고 처음부터 분석"
    )

    args = parser.parse_args()

    # ── 입력 검증 ──
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {pdf_path}")
        sys.exit(1)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY 환경변수를 설정하세요.")
        print("   export ANTHROPIC_API_KEY='sk-ant-...'")
        sys.exit(1)

    # 출력 경로
    output_path = Path(args.output) if args.output else pdf_path.with_suffix(".analysis.md")

    # 캐시 디렉토리
    cache_dir = None
    if not args.no_cache:
        cache_dir = pdf_path.parent / f".{pdf_path.stem}_cache"

    # ── 실행 ──
    print("=" * 60)
    print(f"🔬 논문 분석 파이프라인 시작")
    print(f"   파일: {pdf_path.name}")
    print(f"   모델: {args.model}")
    print(f"   언어: {args.lang}")
    if cache_dir:
        print(f"   캐시: {cache_dir}")
    print("=" * 60)

    start_time = time.time()

    # Stage 1: PDF 전처리
    print("\n📦 [Stage 1/4] PDF 전처리 중...")
    extraction = extract_from_pdf(str(pdf_path), max_images=args.max_images)
    print(f"  ✅ 텍스트: {len(extraction.full_text):,}자 / 이미지: {len(extraction.images)}개")

    if not extraction.full_text.strip():
        print("❌ PDF에서 텍스트를 추출할 수 없습니다. (스캔 PDF일 수 있음)")
        sys.exit(1)

    # Stage 2-4: 분석
    analyzer = PaperAnalyzer(model=args.model, lang=args.lang, cache_dir=cache_dir)

    analyzer.analyze_text(extraction)

    if not args.text_only:
        analyzer.analyze_figures(extraction)

    analyzer.synthesize(extraction)

    # ── 결과 저장 ──
    output_content = _build_output(analyzer.result, extraction, args)
    output_path.write_text(output_content, encoding="utf-8")
    print(f"\n💾 결과 저장: {output_path}")

    elapsed = time.time() - start_time
    print(f"⏱️  총 소요 시간: {elapsed:.1f}초")
    print_summary(analyzer.result, extraction)


def _build_output(result: AnalysisResult, extraction: ExtractionResult, args) -> str:
    """최종 마크다운 보고서 구성"""
    sections = []

    # 헤더
    sections.append(f"<!-- 자동 분석 보고서 | 모델: {args.model} | 날짜: {time.strftime('%Y-%m-%d %H:%M')} -->")
    sections.append("")

    # 종합 분석 (메인)
    if result.synthesis:
        sections.append(result.synthesis)
    else:
        sections.append("# 텍스트 분석 결과\n")
        sections.append(result.text_analysis)

    # 부록: 개별 Figure 분석
    if result.figure_analyses:
        sections.append("\n\n---\n")
        sections.append("# 부록: 개별 Figure/Table 분석 상세\n")
        for fa in result.figure_analyses:
            fig_label = f"Figure {fa.get('figure_number', '')}" if fa.get('figure_number') else f"Image {fa['image_index']}"
            sections.append(f"## {fig_label} (p.{fa['page']})")
            if fa["caption"]:
                sections.append(f"**Caption:** {fa['caption']}\n")
            sections.append(fa["analysis"])
            sections.append("")

    # 토큰 사용량
    sections.append("\n---\n")
    total_in = result.token_usage['input_tokens']
    total_out = result.token_usage['output_tokens']
    cost = _estimate_cost(result.model, total_in, total_out)
    sections.append(f"*분석 토큰 사용량: 입력 {total_in:,} / 출력 {total_out:,} | "
                    f"예상 비용: ~${cost:.3f}*")

    return "\n".join(sections)


if __name__ == "__main__":
    main()
