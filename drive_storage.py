"""
Google Drive 저장 모듈
======================
분석 완료된 논문 PDF와 결과물을 Google Drive에 업로드하고
Google Sheets 인덱스에 메타데이터를 기록한다.

필요 패키지:
    pip install gspread google-api-python-client google-auth

Service Account 설정:
    1. Google Cloud Console → 서비스 계정 생성 → JSON 키 다운로드
    2. Google Drive에 "Paper-Analysis-DB" 폴더 생성
    3. 해당 폴더를 서비스 계정 이메일에 공유 (편집자 권한)
    4. 폴더 안에 Google Sheets "index" 생성 후 동일하게 공유
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

# Sheets 인덱스 헤더 (첫 행)
INDEX_HEADERS = [
    "paper_id",
    "title",
    "authors",
    "analyzed_at",
    "model",
    "language",
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "pdf_link",
    "analysis_link",
    "folder_link",
    "tags",
]


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
    """Google Drive + Sheets 기반 논문 저장소"""

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
            Drive "Paper-Analysis-DB" 루트 폴더 ID
        sheet_id : str
            Google Sheets 인덱스 문서 ID
        credentials_path : str
            Service Account JSON 키 파일 경로 (credentials_json과 택 1)
        credentials_json : str
            Service Account JSON 키 내용 문자열 (credentials_path와 택 1)
        """
        creds = _build_credentials(credentials_path, credentials_json)
        self.drive = build("drive", "v3", credentials=creds)
        self.gc = gspread.authorize(creds)
        self.sheet = self.gc.open_by_key(sheet_id).sheet1
        self.root_folder_id = root_folder_id

        self._ensure_headers()

    def _ensure_headers(self):
        """Sheets 첫 행에 헤더가 없으면 삽입"""
        existing = self.sheet.row_values(1)
        if existing != INDEX_HEADERS:
            self.sheet.update("A1", [INDEX_HEADERS])

    # ── Drive 폴더/파일 작업 ──

    def _create_folder(self, name: str, parent_id: str) -> str:
        """Drive에 폴더를 생성하고 ID를 반환"""
        metadata = {
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_id],
        }
        folder = self.drive.files().create(body=metadata, fields="id").execute()
        return folder["id"]

    def _upload_file(
        self, name: str, parent_id: str, file_path: str | None = None,
        content: bytes | None = None, mime_type: str = "application/octet-stream",
    ) -> dict:
        """파일을 업로드하고 {id, webViewLink}를 반환"""
        metadata = {"name": name, "parents": [parent_id]}

        if file_path:
            media = MediaFileUpload(file_path, mimetype=mime_type, resumable=True)
        elif content is not None:
            media = MediaInMemoryUpload(content, mimetype=mime_type, resumable=True)
        else:
            raise ValueError("file_path 또는 content 중 하나는 필수")

        result = (
            self.drive.files()
            .create(body=metadata, media_body=media, fields="id, webViewLink")
            .execute()
        )
        return result

    def _get_folder_link(self, folder_id: str) -> str:
        return f"https://drive.google.com/drive/folders/{folder_id}"

    # ── 논문 제목 추출 ──

    @staticmethod
    def _extract_title_from_synthesis(synthesis: str) -> str:
        """종합 보고서에서 논문 제목 추출 시도"""
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
        """종합 보고서에서 저자 추출 시도"""
        for pattern in [
            r"[#*]*\s*(?:저자|Authors?)\s*[:：]\s*(.+)",
        ]:
            m = re.search(pattern, synthesis, re.MULTILINE | re.IGNORECASE)
            if m:
                return m.group(1).strip().strip("*")[:300]
        return ""

    # ── 메인: 저장 ──

    def save(
        self,
        pdf_path: str,
        analysis_md: str,
        result,          # AnalysisResult
        model: str,
        lang: str,
        cost: float,
        title: str = "",
        authors: str = "",
        tags: str = "",
    ) -> dict:
        """
        논문 분석 결과를 Google Drive에 저장하고 Sheets 인덱스에 기록.

        Returns
        -------
        dict : {paper_id, folder_id, folder_link, pdf_link, analysis_link}
        """
        paper_id = str(uuid.uuid4())[:8]
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")

        # 제목/저자 자동 추출 (미입력 시)
        if not title:
            title = self._extract_title_from_synthesis(result.synthesis)
        if not title:
            title = Path(pdf_path).stem
        if not authors:
            authors = self._extract_authors_from_synthesis(result.synthesis)

        # 안전한 폴더명 생성
        safe_title = re.sub(r'[\\/:*?"<>|]', "", title)[:60].strip()
        folder_name = f"{date_str}_{safe_title}"

        # 1. 폴더 생성
        folder_id = self._create_folder(folder_name, self.root_folder_id)
        folder_link = self._get_folder_link(folder_id)

        # 2. PDF 업로드
        pdf_result = self._upload_file(
            name=Path(pdf_path).name,
            parent_id=folder_id,
            file_path=pdf_path,
            mime_type="application/pdf",
        )

        # 3. 분석 결과 (.md) 업로드
        md_name = Path(pdf_path).stem + ".analysis.md"
        md_result = self._upload_file(
            name=md_name,
            parent_id=folder_id,
            content=analysis_md.encode("utf-8"),
            mime_type="text/markdown",
        )

        # 4. 메타데이터 JSON 업로드
        metadata = {
            "paper_id": paper_id,
            "title": title,
            "authors": authors,
            "analyzed_at": now.isoformat(),
            "model": model,
            "language": lang,
            "token_usage": result.token_usage,
            "cost_usd": cost,
            "tags": tags,
        }
        meta_bytes = json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8")
        self._upload_file(
            name="metadata.json",
            parent_id=folder_id,
            content=meta_bytes,
            mime_type="application/json",
        )

        # 5. Sheets 인덱스에 행 추가
        row = [
            paper_id,
            title,
            authors,
            now.strftime("%Y-%m-%d %H:%M"),
            model,
            lang,
            result.token_usage["input_tokens"],
            result.token_usage["output_tokens"],
            f"{cost:.3f}",
            pdf_result.get("webViewLink", ""),
            md_result.get("webViewLink", ""),
            folder_link,
            tags,
        ]
        self.sheet.append_row(row, value_input_option="USER_ENTERED")

        return {
            "paper_id": paper_id,
            "folder_id": folder_id,
            "folder_link": folder_link,
            "pdf_link": pdf_result.get("webViewLink", ""),
            "analysis_link": md_result.get("webViewLink", ""),
        }

    # ── 조회 ──

    def list_papers(self) -> list[dict]:
        """Sheets 인덱스에서 모든 논문 목록을 반환"""
        rows = self.sheet.get_all_records()
        return rows

    def get_file_content(self, file_id: str) -> str:
        """Drive 파일 ID로 텍스트 내용을 다운로드하여 반환"""
        content = self.drive.files().get_media(fileId=file_id).execute()
        if isinstance(content, bytes):
            return content.decode("utf-8")
        return str(content)
