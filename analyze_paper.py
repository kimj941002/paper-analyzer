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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import anthropic
import fitz  # PyMuPDF

# ─────────────────────────────────────────────
# 설정 (Configuration)
# ─────────────────────────────────────────────

DEFAULT_MODEL = "claude-sonnet-4-6"
MAX_OUTPUT_TOKENS = 16000          # 단계별 출력 토큰 제한
SYNTHESIS_MAX_TOKENS = 32000       # 종합 단계는 더 길게
IMAGE_MIN_SIZE = (80, 80)          # 최소 이미지 크기 (px) — 로고/아이콘 제외
IMAGE_MAX_COUNT = 30               # 최대 추출 이미지 수
IMAGE_DPI = 200                    # 이미지 렌더링 해상도

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
    #    방법 1: 페이지 렌더링 기반 (figure 전체를 캡처하기 위해)
    #    방법 2: 내장 이미지 직접 추출
    #    → 두 방법을 병행하되, 내장 이미지 추출을 우선 사용

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
        # Figure가 있을 법한 페이지만 렌더링 (텍스트 비율로 판단)
        text = page.get_text("text")
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


def _match_captions(result: ExtractionResult):
    """Figure/Table caption을 이미지에 매칭하고, 본문 문맥 추출"""
    text = result.full_text

    # Figure/Table 캡션 패턴
    caption_pattern = re.compile(
        r'((?:Figure|Fig\.?|Table|Scheme|Chart)\s*\.?\s*(\d+[A-Za-z]?)[\s.:–—-]+[^\n]{5,})',
        re.IGNORECASE
    )
    captions = caption_pattern.findall(text)

    # 본문에서 figure 언급 문맥 추출
    context_pattern = re.compile(
        r'([^.]*(?:Figure|Fig\.?|Table|Scheme|Chart)\s*\.?\s*\d+[A-Za-z]?[^.]*\.)',
        re.IGNORECASE
    )
    contexts = context_pattern.findall(text)

    # 페이지 번호 기반으로 caption ↔ image 매칭
    for img in result.images:
        # 같은 페이지 또는 인접 페이지의 캡션 찾기
        best_caption = ""
        for full_cap, fig_num in captions:
            # 페이지 텍스트에서 해당 캡션이 있는 위치 확인
            page_marker = f"[Page {img.page_num}]"
            prev_page_marker = f"[Page {img.page_num - 1}]"
            next_page_marker = f"[Page {img.page_num + 1}]"

            for marker in [page_marker, prev_page_marker, next_page_marker]:
                start = text.find(marker)
                if start == -1:
                    continue
                end = text.find("[Page", start + 1)
                page_text = text[start:end] if end != -1 else text[start:]
                if full_cap in page_text:
                    best_caption = full_cap.strip()
                    break
            if best_caption:
                break

        img.caption = best_caption

        # 관련 문맥 수집 (본문에서 해당 figure를 언급하는 문장들)
        related_contexts = []
        for ctx in contexts:
            # figure 번호가 이미지 인덱스와 매칭되는지 확인
            nums = re.findall(r'(?:Figure|Fig\.?|Table|Scheme|Chart)\s*\.?\s*(\d+)', ctx, re.IGNORECASE)
            if str(img.index) in nums:
                related_contexts.append(ctx.strip())
        img.context = " ".join(related_contexts[:5])  # 최대 5문장


# ─────────────────────────────────────────────
# Stage 2-4: Claude API 호출
# ─────────────────────────────────────────────

class PaperAnalyzer:
    """Claude API를 사용한 다단계 논문 분석"""

    def __init__(self, model: str = DEFAULT_MODEL, lang: str = "ko"):
        self.client = anthropic.Anthropic()  # ANTHROPIC_API_KEY 환경변수 사용
        self.model = model
        self.lang = lang
        self.result = AnalysisResult()

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
        """Claude API 호출 (재시도 로직 포함)"""
        for attempt in range(3):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=messages,
                )
                # 토큰 사용량 누적
                usage = response.usage
                self.result.token_usage["input_tokens"] += usage.input_tokens
                self.result.token_usage["output_tokens"] += usage.output_tokens

                return response.content[0].text

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

    # ── Stage 2: 텍스트 분석 ──

    def analyze_text(self, extraction: ExtractionResult) -> str:
        """전체 텍스트 기반 논문 구조 분석"""
        print("\n📄 [Stage 2/4] 텍스트 분석 중...")

        system = f"""당신은 생명과학/구조생물학/제약 분야의 논문 분석 전문가입니다.
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
        print("  ✅ 텍스트 분석 완료")
        return self.result.text_analysis

    # ── Stage 3: Figure/Table 개별 분석 ──

    def analyze_figures(self, extraction: ExtractionResult) -> list[dict]:
        """각 Figure/Table 이미지를 개별 분석"""
        if not extraction.images:
            print("\n🖼️  [Stage 3/4] 추출된 이미지 없음 — 건너뜀")
            return []

        print(f"\n🖼️  [Stage 3/4] Figure/Table 분석 중... ({len(extraction.images)}개)")

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

        analyses = []
        for img in extraction.images:
            print(f"  🔍 이미지 {img.index}/{len(extraction.images)} (p.{img.page_num}) 분석 중...")

            # 이미지 + 캡션 + 본문 문맥을 함께 전달
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
            analyses.append({
                "image_index": img.index,
                "page": img.page_num,
                "caption": img.caption,
                "analysis": analysis_text,
            })
            print(f"  ✅ 이미지 {img.index} 완료")

        self.result.figure_analyses = analyses
        return analyses

    def _build_figure_prompt(self, img: ExtractedImage) -> str:
        """Figure 분석을 위한 프롬프트 구성"""
        parts = [f"이 이미지는 논문의 {img.page_num}페이지에 위치합니다."]

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
                header = f"### Image {fa['image_index']} (p.{fa['page']})"
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
        print("  ✅ 종합 분석 완료")
        return self.result.synthesis


# ─────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────

def print_summary(result: AnalysisResult, extraction: ExtractionResult):
    """실행 요약 출력"""
    total_in = result.token_usage["input_tokens"]
    total_out = result.token_usage["output_tokens"]

    print("\n" + "=" * 60)
    print("📈 분석 완료 요약")
    print("=" * 60)
    print(f"  논문 페이지 수: {extraction.metadata.get('pages', '?')}")
    print(f"  추출 텍스트 길이: {len(extraction.full_text):,} 문자")
    print(f"  추출 이미지 수: {len(extraction.images)}")
    print(f"  총 입력 토큰: {total_in:,}")
    print(f"  총 출력 토큰: {total_out:,}")

    # 비용 추정 (Sonnet 4.6 기준: $3/$15 per MTok)
    if "sonnet" in DEFAULT_MODEL or "sonnet" in str(result):
        cost_in = total_in / 1_000_000 * 3
        cost_out = total_out / 1_000_000 * 15
    else:
        cost_in = total_in / 1_000_000 * 5
        cost_out = total_out / 1_000_000 * 25
    print(f"  예상 비용: ~${cost_in + cost_out:.3f}")
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
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = pdf_path.with_suffix(".analysis.md")

    max_images = args.max_images

    # ── 실행 ──
    print("=" * 60)
    print(f"🔬 논문 분석 파이프라인 시작")
    print(f"   파일: {pdf_path.name}")
    print(f"   모델: {args.model}")
    print(f"   언어: {args.lang}")
    print("=" * 60)

    start_time = time.time()

    # Stage 1: PDF 전처리
    print("\n📦 [Stage 1/4] PDF 전처리 중...")
    extraction = extract_from_pdf(str(pdf_path), max_images=max_images)
    print(f"  ✅ 텍스트: {len(extraction.full_text):,}자 / 이미지: {len(extraction.images)}개")

    if not extraction.full_text.strip():
        print("❌ PDF에서 텍스트를 추출할 수 없습니다. (스캔 PDF일 수 있음)")
        sys.exit(1)

    # Stage 2-4: 분석
    analyzer = PaperAnalyzer(model=args.model, lang=args.lang)

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
        # fallback: 텍스트 분석만
        sections.append("# 텍스트 분석 결과\n")
        sections.append(result.text_analysis)

    # 부록: 개별 Figure 분석
    if result.figure_analyses:
        sections.append("\n\n---\n")
        sections.append("# 부록: 개별 Figure/Table 분석 상세\n")
        for fa in result.figure_analyses:
            sections.append(f"## Image {fa['image_index']} (p.{fa['page']})")
            if fa["caption"]:
                sections.append(f"**Caption:** {fa['caption']}\n")
            sections.append(fa["analysis"])
            sections.append("")

    # 토큰 사용량
    sections.append("\n---\n")
    sections.append(f"*분석 토큰 사용량: 입력 {result.token_usage['input_tokens']:,} / "
                     f"출력 {result.token_usage['output_tokens']:,}*")

    return "\n".join(sections)


if __name__ == "__main__":
    main()
