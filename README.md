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
           ├─→ [Stage 3] Figure/Table 개별 분석 (API N회)
           │      └─ 각 이미지 + caption + 본문 문맥을 함께 전달
           │
           └─→ [Stage 4] 종합 (API 1회)
                   └─ 텍스트 분석 + Figure 분석 교차 검증 → 최종 보고서
```

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

# 2. API 키 설정
export ANTHROPIC_API_KEY='sk-ant-api03-...'
```

## 사용법

### 기본 실행 (한국어, Sonnet 4.6)

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

## 문제 해결

| 증상 | 원인 | 해결 |
|------|------|------|
| "텍스트 추출 불가" | 스캔 PDF (이미지 기반) | OCR 전처리 필요 (ocrmypdf 등) |
| 이미지가 0개 추출됨 | PDF 내 이미지 구조 특이 | `IMAGE_MIN_SIZE`를 (30, 30)으로 낮추기 |
| Rate limit 오류 | API 티어 제한 | 자동 재시도 내장, 심하면 `--max-images` 줄이기 |
| 비용 초과 우려 | Figure 수 많음 | `--text-only` 또는 `--max-images 5` 사용 |
