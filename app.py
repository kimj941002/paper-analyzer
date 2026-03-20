#!/usr/bin/env python3
"""
논문 분석 파이프라인 — Streamlit Web UI
======================================
브라우저에서 PDF 업로드 → 분석 → 결과 확인/다운로드까지 한 번에.

실행:
    streamlit run app.py
"""

import os
import tempfile
import time
from pathlib import Path

import streamlit as st

# .env 또는 Streamlit Secrets에서 환경변수 로딩
def _load_secret(key: str, default: str = "") -> str:
    """st.secrets → 환경변수 순으로 값을 찾는다."""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.environ.get(key, default)

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

from analyze_paper import (
    ExtractionResult,
    PaperAnalyzer,
    extract_from_pdf,
    _build_output,
    _estimate_cost,
    DEFAULT_MODEL,
    IMAGE_MAX_COUNT,
)

# Google Drive 저장 모듈 (선택적 — 미설치 시 비활성)
try:
    from drive_storage import DriveStorage
    DRIVE_AVAILABLE = True
except ImportError:
    DRIVE_AVAILABLE = False

# ─────────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="논문 분석기 (Paper Analyzer)",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 논문 분석 파이프라인")

# ─────────────────────────────────────────────
# 공통: Drive 인증 정보 로딩
# ─────────────────────────────────────────────

drive_creds_json = _load_secret("GDRIVE_CREDENTIALS_JSON") if DRIVE_AVAILABLE else ""
drive_folder_id = _load_secret("GDRIVE_FOLDER_ID") if DRIVE_AVAILABLE else ""
drive_sheet_id = _load_secret("GDRIVE_SHEET_ID") if DRIVE_AVAILABLE else ""
_drive_ready = DRIVE_AVAILABLE and drive_creds_json and drive_folder_id and drive_sheet_id

# ─────────────────────────────────────────────
# 메인 탭: 분석 / 데이터베이스
# ─────────────────────────────────────────────

tab_analyze, tab_db = st.tabs(["📄 논문 분석", "📚 데이터베이스"])

# ═════════════════════════════════════════════
# 탭 1: 논문 분석
# ═════════════════════════════════════════════

with tab_analyze:
    st.caption("PDF 업로드 → Claude API로 4단계 자동 분석 → 결과 다운로드")

    # ─────────────────────────────────────────
    # 사이드바: 설정
    # ─────────────────────────────────────────

    with st.sidebar:
        st.header("⚙️ 설정")

        # API 키: Secrets에서 자동 로딩, 없으면 입력란 표시
        api_key = _load_secret("ANTHROPIC_API_KEY")
        if not api_key:
            api_key = st.text_input(
                "Anthropic API Key",
                type="password",
                placeholder="sk-ant-api03-...",
                help="https://console.anthropic.com/settings/keys 에서 발급",
            )
            st.divider()

        # 모델 선택
        model = st.selectbox(
            "모델 선택",
            options=[
                "claude-sonnet-4-6",
                "claude-opus-4-6",
                "claude-haiku-4-5-20251001",
            ],
            index=0,
            help="Sonnet: 균형 / Opus: 최고 정밀도 / Haiku: 빠르고 저렴",
        )

        # 언어 선택
        lang = st.selectbox(
            "분석 언어",
            options=["ko", "en"],
            format_func=lambda x: "🇰🇷 한국어" if x == "ko" else "🇺🇸 English",
        )

        # 옵션
        text_only = st.checkbox(
            "텍스트만 분석 (Figure 건너뜀)",
            help="빠르고 저렴하지만 Figure 분석 없음",
        )

        max_images = st.slider(
            "최대 이미지 분석 수",
            min_value=1,
            max_value=30,
            value=IMAGE_MAX_COUNT,
            disabled=text_only,
        )

        st.divider()

        # ── Google Drive 저장 설정 ──
        st.subheader("☁️ Google Drive 저장")

        if DRIVE_AVAILABLE:
            enable_drive = st.checkbox("분석 완료 후 Drive에 자동 저장")
            if enable_drive:
                drive_tags = st.text_input(
                    "태그 (쉼표 구분)",
                    placeholder="ML, NLP, Transformer",
                )
                if not _drive_ready:
                    st.warning("⚠️ Streamlit Secrets에 `GDRIVE_CREDENTIALS_JSON`, `GDRIVE_FOLDER_ID`, `GDRIVE_SHEET_ID`를 설정하세요.")
            else:
                drive_tags = ""
        else:
            enable_drive = False
            drive_tags = ""
            st.caption("_비활성: `pip install gspread google-api-python-client google-auth` 필요_")

        st.divider()

        # 비용 안내
        st.caption("💰 **예상 비용 참고**")
        st.caption("Sonnet: ~$0.05–0.20/논문")
        st.caption("Opus: ~$0.15–0.60/논문")
        st.caption("Haiku: ~$0.01–0.03/논문")

    # ─────────────────────────────────────────
    # 메인: PDF 업로드
    # ─────────────────────────────────────────

    uploaded_file = st.file_uploader(
        "논문 PDF를 업로드하세요",
        type=["pdf"],
        help="최대 200MB",
    )

    if uploaded_file:
        st.success(f"📄 **{uploaded_file.name}** ({uploaded_file.size / 1024:.0f} KB)")

    # ─────────────────────────────────────────
    # 분석 실행
    # ─────────────────────────────────────────

    if uploaded_file and st.button("🚀 분석 시작", type="primary", use_container_width=True):

        # 입력 검증
        if not api_key:
            st.error("❌ API Key가 없습니다. Streamlit Secrets에 `ANTHROPIC_API_KEY`를 설정하세요.")
            st.stop()

        # API 키 설정
        os.environ["ANTHROPIC_API_KEY"] = api_key

        # 임시 파일로 PDF 저장
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            # ── Stage 1: PDF 전처리 ──
            stage1 = st.status("📦 **Stage 1/4** — PDF 전처리 중...", expanded=True)
            with stage1:
                extraction = extract_from_pdf(tmp_path, max_images=max_images)
                pages = extraction.metadata.get("pages", "?")
                n_images = len(extraction.images)
                text_len = len(extraction.full_text)

                st.write(f"✅ 텍스트: {text_len:,}자 / 이미지: {n_images}개 / 페이지: {pages}")

                if not extraction.full_text.strip():
                    st.error("❌ PDF에서 텍스트를 추출할 수 없습니다. (스캔 PDF일 수 있음)")
                    st.stop()
            stage1.update(label="📦 **Stage 1/4** — PDF 전처리 완료 ✅", state="complete")

            # ── 분석 실행 ──
            analyzer = PaperAnalyzer(model=model, lang=lang)
            start_time = time.time()

            # ── Stage 2: 텍스트 분석 ──
            stage2 = st.status("📄 **Stage 2/4** — 텍스트 분석 중...", expanded=False)
            with stage2:
                analyzer.analyze_text(extraction)
                st.write("✅ 텍스트 분석 완료")
            stage2.update(label="📄 **Stage 2/4** — 텍스트 분석 완료 ✅", state="complete")

            # ── Stage 3: Figure 분석 ──
            if not text_only and extraction.images:
                stage3 = st.status(
                    f"🖼️ **Stage 3/4** — Figure 분석 중... ({n_images}개)",
                    expanded=False,
                )
                with stage3:
                    analyzer.analyze_figures(extraction)
                    st.write(f"✅ {n_images}개 Figure 분석 완료")
                stage3.update(label=f"🖼️ **Stage 3/4** — Figure {n_images}개 분석 완료 ✅", state="complete")
            elif text_only:
                st.info("🖼️ **Stage 3/4** — 텍스트 전용 모드: Figure 분석 건너뜀")

            # ── Stage 4: 종합 ──
            stage4 = st.status("📊 **Stage 4/4** — 종합 분석 중...", expanded=False)
            with stage4:
                analyzer.synthesize(extraction)
                st.write("✅ 종합 분석 완료")
            stage4.update(label="📊 **Stage 4/4** — 종합 분석 완료 ✅", state="complete")

            elapsed = time.time() - start_time

            # ── 결과 표시 ──
            st.divider()

            # 요약 메트릭
            total_in = analyzer.result.token_usage["input_tokens"]
            total_out = analyzer.result.token_usage["output_tokens"]
            cost = _estimate_cost(model, total_in, total_out)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("⏱️ 소요 시간", f"{elapsed:.0f}초")
            col2.metric("📊 입력 토큰", f"{total_in:,}")
            col3.metric("📝 출력 토큰", f"{total_out:,}")
            col4.metric("💰 예상 비용", f"${cost:.3f}")

            st.divider()

            # 분석 결과 (탭)
            result_tab_main, result_tab_text, result_tab_figures = st.tabs([
                "📋 종합 보고서", "📄 텍스트 분석 원문", "🖼️ Figure 개별 분석"
            ])

            with result_tab_main:
                st.markdown(analyzer.result.synthesis or "_(종합 분석 결과 없음)_")

            with result_tab_text:
                st.markdown(analyzer.result.text_analysis or "_(텍스트 분석 결과 없음)_")

            with result_tab_figures:
                if analyzer.result.figure_analyses:
                    for fa in analyzer.result.figure_analyses:
                        fig_label = (
                            f"Figure {fa.get('figure_number', '')}"
                            if fa.get("figure_number")
                            else f"Image {fa['image_index']}"
                        )
                        with st.expander(f"{fig_label} (p.{fa['page']})"):
                            if fa["caption"]:
                                st.caption(f"**Caption:** {fa['caption']}")
                            st.markdown(fa["analysis"])
                else:
                    st.info("Figure 분석 결과가 없습니다.")

            # 다운로드 버튼
            st.divider()

            # argparse Namespace 흉내
            class _Args:
                pass
            args = _Args()
            args.model = model

            output_md = _build_output(analyzer.result, extraction, args)
            output_filename = Path(uploaded_file.name).stem + ".analysis.md"

            st.download_button(
                label="📥 분석 결과 다운로드 (.md)",
                data=output_md,
                file_name=output_filename,
                mime="text/markdown",
                use_container_width=True,
            )

            # ── Google Drive 저장 ──
            if enable_drive and _drive_ready:
                st.divider()
                with st.status("☁️ Google Drive에 저장 중...", expanded=True) as drive_status:
                    try:
                        storage = DriveStorage(
                            root_folder_id=drive_folder_id,
                            sheet_id=drive_sheet_id,
                            credentials_json=drive_creds_json,
                        )
                        save_result = storage.save(
                            pdf_path=tmp_path,
                            analysis_md=output_md,
                            result=analyzer.result,
                            model=model,
                            lang=lang,
                            cost=cost,
                            tags=drive_tags,
                        )
                        st.write(f"✅ 저장 완료 — [Drive 폴더 열기]({save_result['folder_link']})")
                        drive_status.update(label="☁️ Google Drive 저장 완료 ✅", state="complete")
                    except FileNotFoundError:
                        st.error("❌ Service Account JSON 파일을 찾을 수 없습니다. 경로를 확인하세요.")
                        drive_status.update(label="☁️ Drive 저장 실패", state="error")
                    except Exception as e:
                        st.error(f"❌ Drive 저장 중 오류: {e}")
                        drive_status.update(label="☁️ Drive 저장 실패", state="error")

        finally:
            # 임시 파일 정리
            os.unlink(tmp_path)

# ═════════════════════════════════════════════
# 탭 2: 데이터베이스
# ═════════════════════════════════════════════

with tab_db:
    st.caption("Google Drive에 저장된 분석 결과를 조회합니다")

    if not _drive_ready:
        st.warning("⚠️ Google Drive 설정이 필요합니다. Streamlit Secrets를 확인하세요.")
    else:
        # 논문 목록 로딩 (캐시: 5분)
        @st.cache_data(ttl=300)
        def _fetch_papers(_creds_json, _sheet_id, _folder_id):
            storage = DriveStorage(
                root_folder_id=_folder_id,
                sheet_id=_sheet_id,
                credentials_json=_creds_json,
            )
            return storage.list_papers()

        if st.button("🔄 새로고침", key="db_refresh"):
            st.cache_data.clear()

        try:
            papers = _fetch_papers(drive_creds_json, drive_sheet_id, drive_folder_id)
        except Exception as e:
            st.error(f"❌ 목록 조회 실패: {e}")
            papers = []

        if not papers:
            st.info("아직 분석된 논문이 없습니다. '논문 분석' 탭에서 분석을 시작하세요.")
        else:
            st.success(f"총 **{len(papers)}**편의 논문이 저장되어 있습니다.")

            # 검색/필터
            search_query = st.text_input("🔍 논문 검색 (제목, 저자, 태그)", placeholder="키워드 입력...")

            filtered = papers
            if search_query:
                q = search_query.lower()
                filtered = [
                    p for p in papers
                    if q in str(p.get("title", "")).lower()
                    or q in str(p.get("authors", "")).lower()
                    or q in str(p.get("tags", "")).lower()
                ]
                st.caption(f"검색 결과: {len(filtered)}건")

            # 논문 목록 표시 (최신순)
            for paper in reversed(filtered):
                title = paper.get("title", "(제목 없음)")
                authors = paper.get("authors", "")
                date = paper.get("analyzed_at", "")
                model_used = paper.get("model", "")
                tags = paper.get("tags", "")
                cost_usd = paper.get("cost_usd", "")
                pdf_link = paper.get("pdf_link", "")
                analysis_link = paper.get("analysis_link", "")
                folder_link = paper.get("folder_link", "")

                with st.expander(f"📄 **{title}**  —  {date}"):
                    # 메타 정보
                    meta_cols = st.columns([2, 1, 1])
                    with meta_cols[0]:
                        if authors:
                            st.write(f"**저자:** {authors}")
                    with meta_cols[1]:
                        st.write(f"**모델:** {model_used}")
                    with meta_cols[2]:
                        if cost_usd:
                            st.write(f"**비용:** ${cost_usd}")

                    if tags:
                        st.write(f"**태그:** {tags}")

                    # 링크 버튼
                    link_cols = st.columns(3)
                    with link_cols[0]:
                        if pdf_link:
                            st.link_button("📎 PDF 보기", pdf_link, use_container_width=True)
                    with link_cols[1]:
                        if analysis_link:
                            st.link_button("📋 분석 결과 보기", analysis_link, use_container_width=True)
                    with link_cols[2]:
                        if folder_link:
                            st.link_button("📁 Drive 폴더", folder_link, use_container_width=True)

                    # 분석 내용 인라인 보기
                    if analysis_link:
                        # analysis_link에서 file ID 추출
                        import re as _re
                        _fid_match = _re.search(r"/d/([a-zA-Z0-9_-]+)", analysis_link)
                        if _fid_match:
                            _file_id = _fid_match.group(1)
                            if st.button(f"📖 분석 내용 펼치기", key=f"view_{paper.get('paper_id', title)}"):
                                try:
                                    storage = DriveStorage(
                                        root_folder_id=drive_folder_id,
                                        sheet_id=drive_sheet_id,
                                        credentials_json=drive_creds_json,
                                    )
                                    md_content = storage.get_file_content(_file_id)
                                    st.markdown(md_content)
                                except Exception as e:
                                    st.error(f"내용을 불러올 수 없습니다: {e}")
