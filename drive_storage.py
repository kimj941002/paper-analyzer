"""
Google Drive / Sheets 저장 모듈
================================
논문 분석 결과를 Google Sheets 인덱스에 기록하고,
PDF·분석 파일을 Google Drive(Shared Drive 권장)에 업로드한다.

Service Account 설정:
    1. Google Cloud Console → 서비스 계정 생성 → JSON 키 다운로드
    2. Google Drive API, Sheets API 활성화
    3. Google Sheets 인덱스 문서 생성 → 서비스 계정 이메일에 편집자 공유
    4. [PDF 저장 원할 경우] Shared Drive 생성 → 서비스 계정을 콘텐츠 관리자로 추가
       ※ 일반 내 드라이브는 서비스 계정 쿼터 미지원(403 오류) → Shared Drive 필수

필요 패키지:
    pip install gspread google-api-python-client google-auth
"""

import json
import re
import uuid
from datetime import datetime
from pathlib import Path

import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaInMemoryUpload

# Google API 범위
SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets",
]

# Sheets 인덱스 헤더 — 순서 변경 시 _ensure_headers 가 자동 갱신
INDEX_HEADERS = [
    "paper_id",
    "title",
    "doi",
    "authors",
    "analyzed_at",
    "model",
    "language",
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "pdf_link",
    "folder_link",
    "tags",
    "analysis_md",   # 분석 결과 전문 (Drive 없이도 조회 가능)
]

# Sheets 셀 최대 문자 수
_SHEETS_CELL_LIMIT = 50_000


def _build_credentials(credentials_path: str = "", credentials_json: str = "") -> Credentials:
    """파일 경로 또는 JSON 문자열로 인증 정보 생성"""
    if credentials_json:
        info = json.loads(credentials_json)
        return Credentials.from_service_account_info(info, scopes=SCOPES)
    elif credentials_path:
        return Credentials.from_service_account_file(credentials_path, scopes=SCOPES)
    else:
        raise ValueError("credentials_path 또는 credentials_json 중 하나는 필수")


class DriveStorage:
    """Google Sheets 인덱스 + Google Drive(선택) 기반 논문 저장소"""

    def __init__(
        self,
        root_folder_id: str,
        sheet_id: str,
        credentials_path: str = "",
        credentials_json: str = "",
    ):
        """
        Parameters
        ----------
        root_folder_id : str
            Drive 루트 폴더 ID (Shared Drive 폴더 권장; 빈 문자열이면 Drive 저장 건너뜀)
        sheet_id : str
            Google Sheets 인덱스 문서 ID
        credentials_path / credentials_json : str
            Service Account 인증 정보 (둘 중 하나)
        """
        creds = _build_credentials(credentials_path, credentials_json)
        self.drive = build("drive", "v3", credentials=creds)
        self.gc = gspread.authorize(creds)
        self.sheet = self.gc.open_by_key(sheet_id).sheet1
        self.root_folder_id = root_folder_id

        self._ensure_headers()

    # ── 헤더 관리 ──

    def _ensure_headers(self) -> None:
        """Sheets 첫 행을 최신 INDEX_HEADERS 로 동기화"""
        existing = self.sheet.row_values(1)
        if existing != INDEX_HEADERS:
            self.sheet.update("A1", [INDEX_HEADERS])

    # ── Drive 파일/폴더 작업 ──

    def _create_folder(self, name: str, parent_id: str) -> str:
        """Drive에 폴더를 생성하고 ID를 반환 (Shared Drive 지원)"""
        metadata = {
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_id],
        }
        folder = self.drive.files().create(
            body=metadata,
            fields="id",
            supportsAllDrives=True,
        ).execute()
        return folder["id"]

    def _upload_file(
        self,
        name: str,
        parent_id: str,
        file_path: str | None = None,
        content: bytes | None = None,
        mime_type: str = "application/octet-stream",
    ) -> dict:
        """파일을 업로드하고 {id, webViewLink} 반환 (Shared Drive 지원)"""
        metadata = {"name": name, "parents": [parent_id]}

        if file_path:
            media = MediaFileUpload(file_path, mimetype=mime_type, resumable=True)
        elif content is not None:
            media = MediaInMemoryUpload(content, mimetype=mime_type, resumable=True)
        else:
            raise ValueError("file_path 또는 content 중 하나는 필수")

        result = self.drive.files().create(
            body=metadata,
            media_body=media,
            fields="id, webViewLink",
            supportsAllDrives=True,
        ).execute()
        return result

    def _get_folder_link(self, folder_id: str) -> str:
        return f"https://drive.google.com/drive/folders/{folder_id}"

    # ── 메타데이터 추출 헬퍼 ──

    @staticmethod
    def _extract_title_from_synthesis(synthesis: str) -> str:
        """종합 보고서에서 논문 제목 추출"""
        for pattern in [
            r"[#*]*\s*(?:논문\s*제목|Paper\s*Title|Title)\s*[:：]\s*(.+)",
            r"^#\s+(.+)",
        ]:
            m = re.search(pattern, synthesis, re.MULTILINE | re.IGNORECASE)
            if m:
                title = m.group(1).strip().strip("*").strip('"').strip("'")
                if len(title) > 10:
                    return title[:200]
        return ""

    @staticmethod
    def _extract_authors_from_synthesis(synthesis: str) -> str:
        """종합 보고서에서 저자 추출"""
        m = re.search(
            r"[#*]*\s*(?:저자|Authors?)\s*[:：]\s*(.+)",
            synthesis,
            re.MULTILINE | re.IGNORECASE,
        )
        if m:
            return m.group(1).strip().strip("*")[:300]
        return ""

    # ── 메인: 저장 ──

    def save(
        self,
        pdf_path: str,
        analysis_md: str,
        result,            # AnalysisResult
        model: str,
        lang: str,
        cost: float,
        doi: str = "",
        title: str = "",
        authors: str = "",
        tags: str = "",
    ) -> dict:
        """
        논문 분석 결과를 Sheets + Drive(선택)에 저장.

        - Sheets: 항상 저장 (메타데이터 + 분석 MD 전문)
        - Drive: root_folder_id 설정 시 시도; 실패해도 Sheets 저장은 유지

        Returns
        -------
        dict : {paper_id, folder_link, pdf_link, drive_ok}
        """
        paper_id = str(uuid.uuid4())[:8]
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")

        # 제목/저자 자동 추출
        if not title:
            title = self._extract_title_from_synthesis(result.synthesis)
        if not title:
            title = Path(pdf_path).stem
        if not authors:
            authors = self._extract_authors_from_synthesis(result.synthesis)

        # ── Drive 업로드 (Shared Drive 필요; 실패 시 무시) ──
        folder_link = ""
        pdf_link = ""
        drive_ok = False

        if self.root_folder_id:
            try:
                safe_title = re.sub(r'[\\/:*?"<>|]', "", title)[:60].strip()
                folder_name = f"{date_str}_{safe_title}"
                folder_id = self._create_folder(folder_name, self.root_folder_id)
                folder_link = self._get_folder_link(folder_id)

                # PDF 업로드
                pdf_result = self._upload_file(
                    name=Path(pdf_path).name,
                    parent_id=folder_id,
                    file_path=pdf_path,
                    mime_type="application/pdf",
                )
                pdf_link = pdf_result.get("webViewLink", "")

                # 분석 MD 업로드
                self._upload_file(
                    name=Path(pdf_path).stem + ".analysis.md",
                    parent_id=folder_id,
                    content=analysis_md.encode("utf-8"),
                    mime_type="text/markdown",
                )

                drive_ok = True
            except Exception as e:
                print(f"  ⚠️  Drive 업로드 실패 (Sheets에는 저장됨): {e}")

        # ── Sheets 인덱스에 행 추가 (항상 실행) ──
        row = [
            paper_id,
            title,
            doi,
            authors,
            now.strftime("%Y-%m-%d %H:%M"),
            model,
            lang,
            result.token_usage["input_tokens"],
            result.token_usage["output_tokens"],
            f"{cost:.4f}",
            pdf_link,
            folder_link,
            tags,
            analysis_md[:_SHEETS_CELL_LIMIT],
        ]
        self.sheet.append_row(row, value_input_option="USER_ENTERED")

        return {
            "paper_id": paper_id,
            "folder_link": folder_link,
            "pdf_link": pdf_link,
            "drive_ok": drive_ok,
        }

    # ── 조회 ──

    def list_papers(self) -> list[dict]:
        """Sheets 인덱스에서 모든 논문 목록을 반환"""
        return self.sheet.get_all_records()
