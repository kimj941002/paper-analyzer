# 논문 분석 파이프라인 (Paper Analysis Pipeline)

Claude API를 활용한 4단계 과학 논문 자동 분석 도구입니다.  
PDF의 텍스트와 이미지를 **분리 추출**한 뒤, 각각 독립적으로 분석하고 최종 종합하여
Figure 분석 오류와 환각(hallucination)을 최소화합니다.

## 파이프라인 구조

```
PDF ──→ [Stage 1] 전처리 (텍스트 + 이미지 분리 추출)
           │
           ├─→ [Stage 2] 텍스트 전체 분석 (API 1회)
           │
           ├─→ [Stage 3] Figure/Table 개별 분석 (API N회, 병렬 4개)
           │      └─ 각 이미지 + caption + 본문 문맥을 함께 전달
           │
           └─→ [Stage 4] 종합 (API 1회)
                   └─ 텍스트 분석 + Figure 분석 교차 검증 → 최종 보고서
```

각 Stage 완료 시 중간 결과가 캐시 파일로 저장되어, 중단 후 `--resume`으로 재개 가능합니다.

### 왜 이 방식이 더 정확한가?

| 문제 | 기존 방식 (PDF 통째 전달) | 이 파이프라인 |
|------|-------------------------|-------------|
| Figure 해석 오류 | 텍스트에 과집중, 이미지 무시/오해 | 이미지를 개별 전달하여 집중 분석 |
| 문맥 부족 | 이미지만 보고 추측 | caption + 본문 언급 문맥 동시 제공 |
| 환각 발생 | 전체를 한번에 처리하여 혼동 | 단계 분리로 교차 검증 |
| 데이터 누락 | 긴 논문에서 후반부 정보 손실 | 텍스트/이미지 각각 완전 분석 |

## 설치

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. API 키 설정 (.env 파일 사용 — Git에 커밋되지 않음)
cp .env.example .env
# .env 파일을 열어 실제 API 키를 입력하세요
```

## 사용법

### 🌐 웹 UI (Streamlit) — 추천

터미널 없이 브라우저에서 모든 것을 할 수 있습니다:

```bash
streamlit run app.py
```

브라우저가 자동으로 열리면:
1. 왼쪽 사이드바에 **API Key** 입력
2. **논문 PDF 업로드**
3. **분석 시작** 버튼 클릭
4. 완료 후 결과 확인 & **다운로드**

### 💻 CLI (터미널)

```bash
python analyze_paper.py paper.pdf
```

### 옵션

```bash
# 영어로 분석
python analyze_paper.py paper.pdf --lang en

# Opus 모델 사용 (더 정밀, 비용 높음)
python analyze_paper.py paper.pdf --model claude-opus-4-6

# 출력 파일 지정
python analyze_paper.py paper.pdf -o my_analysis.md

# 텍스트만 분석 (Figure 건너뜀, 빠르고 저렴)
python analyze_paper.py paper.pdf --text-only

# 이미지 최대 분석 수 제한
python analyze_paper.py paper.pdf --max-images 10

# 중단된 분석 재개 (캐시 활용)
python analyze_paper.py paper.pdf --resume

# 캐시 무시하고 처음부터 분석
python analyze_paper.py paper.pdf --no-cache
```

### 출력 예시

실행 후 `paper.analysis.md` 파일이 생성됩니다:

```
# [논문 제목]

## 기본 정보
## 1. 연구 배경 및 목적
  ### 1.1 연구 배경
  ### 1.2 기존 연구의 한계
  ...
## 3. 주요 결과
  ### 3.1 [결과 주제] — Figure 연계 분석 포함
  ...
## 7. 비판적 검토

---
# 부록: 개별 Figure/Table 분석 상세
```

## 비용 추정

| 논문 유형 | Sonnet 4.6 | Opus 4.6 |
|----------|-----------|---------|
| 짧은 논문 (~8p, Figure 4개) | ~$0.05 | ~$0.15 |
| 일반 논문 (~15p, Figure 8개) | ~$0.10 | ~$0.30 |
| 긴 논문 (~30p, Figure 15개) | ~$0.20 | ~$0.60 |

## 커스터마이즈

### System Prompt 수정

`analyze_paper.py` 내 각 Stage의 `system` 변수를 수정하면 분석 방향을 조정할 수 있습니다.

- **Stage 2** (`analyze_text` 메서드): 텍스트 분석 지침
- **Stage 3** (`analyze_figures` 메서드): Figure 분석 지침  
- **Stage 4** (`synthesize` 메서드): 종합 보고서 양식

예를 들어, 특정 분야(예: ARS biology)에 특화된 분석이 필요하면:

```python
# Stage 2의 system prompt에 추가
system = """...기존 내용...

추가 지침:
- Aminoacyl-tRNA synthetase (ARS) 관련 내용은 특히 상세히 분석하세요.
- 항체 결합 분석 (SPR, BLI) 데이터가 있으면 kinetic parameter를 정리하세요.
- 단백질 구조 분석 데이터는 resolution, R-factor 등을 반드시 포함하세요.
"""
```

### 이미지 필터링 조정

스크립트 상단의 설정값을 조정하세요:

```python
IMAGE_MIN_SIZE = (80, 80)    # 작은 이미지 필터링 기준 (px)
IMAGE_DPI = 200              # 페이지 렌더링 해상도 (높을수록 선명, 토큰 소모 증가)
IMAGE_MAX_COUNT = 30         # 최대 분석 이미지 수
```

## Google Drive 저장 (선택)

분석 결과를 Google Drive에 자동 업로드하고 Google Sheets로 인덱스를 관리할 수 있습니다.

### 1단계: 추가 패키지 설치

```bash
pip install gspread google-api-python-client google-auth
```

### 2단계: Google Cloud Service Account 설정

1. [Google Cloud Console](https://console.cloud.google.com/) → 프로젝트 선택/생성
2. **API 및 서비스 → 라이브러리** → "Google Drive API", "Google Sheets API" 활성화
3. **API 및 서비스 → 사용자 인증 정보** → 서비스 계정 만들기 → JSON 키 다운로드
4. JSON 키 파일을 안전한 경로에 저장 (예: `~/.config/paper-analyzer/sa-key.json`)

### 3단계: Google Drive 폴더 & Sheets 준비

1. Google Drive에 `Paper-Analysis-DB` 폴더 생성
2. 폴더 안에 Google Sheets 문서 `index` 생성
3. **두 항목 모두** 서비스 계정 이메일 (JSON 파일 내 `client_email`)에 **편집자** 권한으로 공유
4. 폴더 ID: 폴더 URL에서 `https://drive.google.com/drive/folders/XXXXX` → `XXXXX` 부분
5. Sheets ID: 문서 URL에서 `https://docs.google.com/spreadsheets/d/XXXXX/edit` → `XXXXX` 부분

### 4단계: 환경변수 설정 (선택 — .env 파일)

```bash
# .env
GDRIVE_CREDENTIALS=/path/to/sa-key.json
GDRIVE_FOLDER_ID=1AbCdEf...
GDRIVE_SHEET_ID=1XyZ...
```

또는 웹 UI 사이드바에서 직접 입력해도 됩니다.

### Drive 저장 구조

```
📁 Paper-Analysis-DB/
├── 📊 index (Google Sheets)
│     paper_id | title | authors | analyzed_at | model | ...
├── 📁 2026-03-18_논문제목/
│   ├── 📄 paper.pdf
│   ├── 📄 paper.analysis.md
│   └── 📄 metadata.json
└── 📁 2026-03-19_다른논문/
    └── ...
```

## 문제 해결

| 증상 | 원인 | 해결 |
|------|------|------|
| "텍스트 추출 불가" | 스캔 PDF (이미지 기반) | OCR 전처리 필요 (ocrmypdf 등) |
| 이미지가 0개 추출됨 | PDF 내 이미지 구조 특이 | `IMAGE_MIN_SIZE`를 (30, 30)으로 낮추기 |
| Rate limit 오류 | API 티어 제한 | 자동 재시도 내장, 심하면 `--max-images` 줄이기 |
| 비용 초과 우려 | Figure 수 많음 | `--text-only` 또는 `--max-images 5` 사용 |
