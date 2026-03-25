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
from __future__ import annotations

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
#
# 전략: "캡션 기반 전수 추출"
#
# 기존 문제: get_images()로 내장 이미지 객체만 추출하면
#   - 벡터 기반 그래프, 텍스트 기반 테이블이 누락됨
#   - 하나의 Figure가 여러 이미지 조각으로 분해됨
#
# 해결: 텍스트에서 Figure/Table 캡션을 먼저 전수 탐색하고,
#       해당 캡션이 있는 페이지를 통째로 렌더링한다.
#   → PDF 내부 구조에 무관하게 모든 Figure/Table 확보
# ─────────────────────────────────────────────

@dataclass
class FigureTableEntry:
    """텍스트에서 탐지된 Figure/Table 항목"""
    label: str          # "Figure 1", "Table 2" 등
    fig_type: str       # "figure" or "table"
    number: str         # "1", "2A" 등
    caption: str        # 전체 캡션 텍스트
    page_num: int       # 캡션이 위치한 페이지 (1-indexed)
    context: str = ""   # 본문에서 언급하는 문맥


def extract_from_pdf(pdf_path: str, max_images: int = IMAGE_MAX_COUNT) -> ExtractionResult:
    """PDF에서 텍스트와 이미지를 분리 추출"""
    import fitz  # PyMuPDF — 실제 PDF 처리 시에만 임포트
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
    page_text_map = {}  # page_num(0-indexed) -> text
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            page_texts.append(f"[Page {page_num + 1}]\n{text}")
            page_text_map[page_num] = text
    result.full_text = "\n\n".join(page_texts)

    # ── Step A: 캡션 전수 탐색 — Figure/Table 매니페스트 구축 ──
    manifest = _build_figure_table_manifest(result.full_text)
    print(f"  📋 탐지된 Figure/Table: {len(manifest)}개")
    for entry in manifest:
        print(f"     {entry.label} (p.{entry.page_num})")

    # ── Step B: 매니페스트 기반 페이지 렌더링 ──
    images = _render_manifest_pages(doc, manifest, max_images)

    # ── Step C: 매니페스트에 없는 시각 요소 보완 (Supplementary 등) ──
    rendered_pages = {img.page_num for img in images}
    extra = _render_visual_pages(doc, rendered_pages, max_images - len(images))
    images.extend(extra)

    result.images = images[:max_images]
    result.metadata["manifest"] = manifest

    doc.close()
    return result


def _build_figure_table_manifest(full_text: str) -> list[FigureTableEntry]:
    """전체 텍스트에서 모든 Figure/Table 캡션을 탐지하여 매니페스트 구축"""

    # 캡션 패턴: Figure/Fig/Table/Scheme/Chart + 번호 + 명시적 구분자(. : – —) + 설명
    # 주의: 단순 공백은 구분자로 인정하지 않음 (본문 언급과 캡션 구분)
    caption_pattern = re.compile(
        r'((?:Figure|Fig\.?|Table|Scheme|Chart|Supplementary\s+(?:Figure|Fig\.?|Table))'
        r'\s*\.?\s*'
        r'(S?\d+[A-Za-z]?(?:\s*[-–—]\s*[A-Za-z])?)'  # "2A-C" 같은 패널 범위 포함
        r'\s*[.:–—|]\s*'  # 명시적 구분자 필수 (공백만으로는 불가)
        r'([^\n]{5,}))',
        re.IGNORECASE
    )

    # 본문에서 Figure/Table 언급 문맥 추출
    context_pattern = re.compile(
        r'([^.]*(?:Figure|Fig\.?|Table|Scheme|Chart)\s*\.?\s*S?\d+[A-Za-z]?[^.]*\.)',
        re.IGNORECASE
    )
    all_contexts = context_pattern.findall(full_text)

    # 페이지별 텍스트 분할
    page_splits = re.split(r'\[Page (\d+)\]\n', full_text)

    manifest = []
    seen_labels = set()

    for match in caption_pattern.finditer(full_text):
        full_cap = match.group(1).strip()
        number = match.group(2).strip()
        description = match.group(3).strip()

        # Figure vs Table 구분
        cap_lower = full_cap.lower()
        if "table" in cap_lower:
            fig_type = "table"
        else:
            fig_type = "figure"

        # 표준화된 라벨 생성
        type_word = "Table" if fig_type == "table" else "Figure"
        if "supplementary" in cap_lower:
            type_word = f"Supplementary {type_word}"
        label = f"{type_word} {number}"

        # 중복 제거
        if label.lower() in seen_labels:
            continue
        seen_labels.add(label.lower())

        # 캡션 위치의 페이지 번호 찾기
        cap_pos = match.start()
        page_num = _find_page_at_position(full_text, cap_pos)

        # 본문 언급 문맥 수집 — 타입(Figure/Table)과 번호 모두 일치해야 함
        related = []
        type_keywords = ("table",) if fig_type == "table" else ("figure", "fig")
        for ctx in all_contexts:
            ctx_lower = ctx.lower()
            # 해당 타입의 키워드가 있는지 확인
            has_type = any(kw in ctx_lower for kw in type_keywords)
            if not has_type:
                continue
            # 해당 번호를 언급하는지 확인
            ctx_nums = re.findall(
                r'(?:Figure|Fig\.?|Table|Scheme|Chart)\s*\.?\s*(S?\d+[A-Za-z]?)',
                ctx, re.IGNORECASE
            )
            # 번호의 숫자 부분만 비교 (2A → 2, S1 → S1)
            number_base = re.match(r'(S?\d+)', number).group(1) if number else ""
            matched = any(
                re.match(r'(S?\d+)', n).group(1) == number_base
                for n in ctx_nums
                if re.match(r'(S?\d+)', n)
            )
            if matched:
                ctx_clean = ctx.strip()
                if ctx_clean != full_cap and ctx_clean not in related:
                    related.append(ctx_clean)

        manifest.append(FigureTableEntry(
            label=label,
            fig_type=fig_type,
            number=number,
            caption=full_cap,
            page_num=page_num,
            context=" ".join(related[:8]),
        ))

    # 페이지 순서로 정렬
    manifest.sort(key=lambda e: (e.page_num, e.number))
    return manifest


def _find_page_at_position(full_text: str, position: int) -> int:
    """텍스트 내 특정 위치가 어떤 페이지에 해당하는지 찾기"""
    page_markers = list(re.finditer(r'\[Page (\d+)\]', full_text))
    current_page = 1
    for marker in page_markers:
        if marker.start() <= position:
            current_page = int(marker.group(1))
        else:
            break
    return current_page


def _render_manifest_pages(
    doc: fitz.Document,
    manifest: list[FigureTableEntry],
    max_images: int,
) -> list[ExtractedImage]:
    """매니페스트에 등재된 Figure/Table이 있는 페이지를 렌더링"""
    images = []

    # 동일 페이지에 여러 Figure가 있을 수 있으므로 페이지별로 그룹핑
    page_to_entries: dict[int, list[FigureTableEntry]] = {}
    for entry in manifest:
        pg = entry.page_num
        if pg not in page_to_entries:
            page_to_entries[pg] = []
        page_to_entries[pg].append(entry)

    for page_num_1indexed in sorted(page_to_entries.keys()):
        if len(images) >= max_images:
            break

        page_idx = page_num_1indexed - 1
        if page_idx < 0 or page_idx >= len(doc):
            continue

        entries = page_to_entries[page_num_1indexed]
        page = doc[page_idx]

        # 페이지 전체를 고해상도로 렌더링
        mat = fitz.Matrix(IMAGE_DPI / 72, IMAGE_DPI / 72)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        b64 = base64.b64encode(img_bytes).decode("utf-8")

        # 이 페이지의 모든 Figure/Table 캡션과 문맥을 합산
        combined_captions = []
        combined_contexts = []
        combined_labels = []
        for entry in entries:
            combined_labels.append(entry.label)
            combined_captions.append(entry.caption)
            if entry.context:
                combined_contexts.append(entry.context)

        images.append(ExtractedImage(
            index=len(images) + 1,
            page_num=page_num_1indexed,
            base64_data=b64,
            media_type="image/png",
            width=pix.width,
            height=pix.height,
            caption=" | ".join(combined_captions),
            context=" ".join(combined_contexts),
        ))

    return images


def _render_visual_pages(
    doc: fitz.Document,
    already_rendered: set[int],
    remaining_slots: int,
) -> list[ExtractedImage]:
    """매니페스트에 없지만 시각 요소가 있는 페이지를 추가 렌더링
    (Supplementary figure, 캡션 없는 도식 등 보완)"""
    if remaining_slots <= 0:
        return []

    images = []
    for page_idx, page in enumerate(doc):
        page_num = page_idx + 1
        if page_num in already_rendered:
            continue
        if len(images) >= remaining_slots:
            break

        # 이 페이지에 이미지 블록이 있는지 확인
        blocks = page.get_text("dict")["blocks"]
        image_blocks = [b for b in blocks if b.get("type") == 1]

        # 텍스트 대비 이미지 비율이 높은 페이지만 (figure-heavy page)
        text = page.get_text("text").strip()
        if not image_blocks:
            continue
        # 텍스트가 매우 적고 이미지가 있으면 → figure 전용 페이지일 가능성
        # 또는 이미지 블록이 3개 이상이면 → figure가 있을 가능성
        if len(text) > 500 and len(image_blocks) < 3:
            continue

        mat = fitz.Matrix(IMAGE_DPI / 72, IMAGE_DPI / 72)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        b64 = base64.b64encode(img_bytes).decode("utf-8")

        images.append(ExtractedImage(
            index=0,  # 나중에 재번호 매김
            page_num=page_num,
            base64_data=b64,
            media_type="image/png",
            width=pix.width,
            height=pix.height,
            caption="(캡션 매칭 없음 — 추가 탐지된 시각 요소)",
            context="",
        ))

    return images


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
        parts = [f"이 이미지는 논문의 {img.page_num}페이지를 렌더링한 것입니다."]
        parts.append("이 페이지에 포함된 모든 Figure, Table, 그래프, 도식을 빠짐없이 분석하세요.")

        if img.caption:
            parts.append(f"\n이 페이지의 캡션(들): {img.caption}")

        if img.context:
            parts.append(f"\n본문에서 이 Figure/Table을 언급하는 문맥:\n{img.context}")

        parts.append("\n위 이미지를 상세히 분석해주세요. 패널(A, B, C 등)이 있으면 각각 분석하세요.")
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
                header = f"### Page {fa['page']} 분석"
                if fa["caption"]:
                    header += f"\n포함된 Figure/Table: {fa['caption']}"
                parts.append(f"{header}\n{fa['analysis']}")
            figure_summary = "\n\n".join(parts)

        system = f"""당신은 과학 논문의 내용을 정확히 전달하는 전문 리포터입니다.
아래에 여러 경로로 수집된 정보(텍스트 추출 결과, 이미지 관찰 결과)가 제공됩니다.
이것들은 **하나의 논문에서 나온 동일한 내용의 서로 다른 표현**입니다.

당신의 임무: 이 모든 정보를 재료로 삼아, 마치 이 논문을 처음부터 직접 읽고
완벽히 이해한 사람이 쓴 것처럼 **하나의 통합된 분석 보고서**를 작성하세요.

{self._lang_instruction()}

■ 절대 하지 말아야 할 것:
- "텍스트 분석에서는 ~했고, 이미지 분석에서는 ~했다" 식의 분석 과정 서술 금지
- "두 분석 간 불일치가 있다", "A에서는 확인되지 않았다" 등 메타 비교 금지
- 분석 단계, 파이프라인, 수집 방법에 대한 언급 일체 금지
- 당신은 원본 논문만 읽은 것처럼 서술해야 합니다

■ 반드시 해야 할 것:
- 모든 Figure, Table의 내용을 해당 결과 섹션에 자연스럽게 통합
  (예: "Figure 3A에서 보듯이, cMET의 인산화는 ~" 식으로 논문 저자가 쓰듯 서술)
- 수치 데이터가 여러 소스에서 확인되면 가장 구체적인 값을 채택
- 텍스트에서만 언급된 정보도, 이미지에서만 관찰된 정보도 빠짐없이 포함
- Figure/Table이 보여주는 데이터를 결과 해석의 근거로 직접 인용

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
  (Figure/Table 데이터를 본문에 자연스럽게 통합하여 기술)
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
(방법론의 적절성, 결론의 논리적 타당성, 잠재적 한계에 대한 분석자의 의견)"""

        user_content = f"""다음은 논문에서 수집된 모든 정보입니다.

---
[텍스트에서 추출된 내용]

{self.result.text_analysis}

---
[Figure/Table 이미지에서 관찰된 내용]

{figure_summary if figure_summary else "(해당 없음)"}

---

위 정보를 모두 활용하여 이 논문의 통합 분석 보고서를 작성하세요.
분석 과정이나 정보 수집 방법은 언급하지 마세요."""

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

    # 이미지 재번호 매기기
    for i, img in enumerate(extraction.images):
        img.index = i + 1

    manifest = extraction.metadata.get("manifest", [])
    print(f"  ✅ 텍스트: {len(extraction.full_text):,}자")
    print(f"  ✅ Figure/Table 탐지: {len(manifest)}개 → 렌더링: {len(extraction.images)}페이지")

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


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """모델별 API 비용 추정 (USD)"""
    if "sonnet" in model:
        cost_in = input_tokens / 1_000_000 * 3
        cost_out = output_tokens / 1_000_000 * 15
    elif "opus" in model:
        cost_in = input_tokens / 1_000_000 * 15
        cost_out = output_tokens / 1_000_000 * 75
    elif "haiku" in model:
        cost_in = input_tokens / 1_000_000 * 0.25
        cost_out = output_tokens / 1_000_000 * 1.25
    else:
        cost_in = input_tokens / 1_000_000 * 5
        cost_out = output_tokens / 1_000_000 * 25
    return cost_in + cost_out


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
