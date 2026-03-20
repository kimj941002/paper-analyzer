#!/usr/bin/env bash
# 초기 환경 세팅 스크립트 — git clone 후 한 번만 실행
set -e

ENV_FILE=".env"

if [ -f "$ENV_FILE" ]; then
    echo "⚠️  .env 파일이 이미 존재합니다. 덮어쓰려면 삭제 후 다시 실행하세요."
    exit 0
fi

echo "=== Paper Analyzer 초기 설정 ==="
echo ""

# 1) Anthropic API Key
read -rp "Anthropic API Key: " api_key
echo ""

# 2) Google Drive (선택)
read -rp "Google Drive 저장 기능 사용? (y/N): " use_drive
echo ""

if [[ "$use_drive" =~ ^[Yy]$ ]]; then
    echo "Service Account JSON 키 파일 경로를 입력하세요."
    echo "(다운로드한 .json 파일을 드래그&드롭하면 경로가 입력됩니다)"
    read -rp "JSON 키 파일 경로: " json_path

    # 경로의 따옴표/공백 정리
    json_path=$(echo "$json_path" | sed "s/^['\"]//;s/['\"]$//;s/^ //;s/ $//")

    if [ ! -f "$json_path" ]; then
        echo "❌ 파일을 찾을 수 없습니다: $json_path"
        exit 1
    fi

    # JSON 파일 내용을 한 줄로 변환
    creds_json=$(python3 -c "import json,sys; print(json.dumps(json.load(open(sys.argv[1]))))" "$json_path")

    read -rp "Drive 루트 폴더 ID: " folder_id
    read -rp "인덱스 Sheets ID: " sheet_id
    echo ""
fi

# 3) .env 생성
cat > "$ENV_FILE" <<EOL
ANTHROPIC_API_KEY=${api_key}
EOL

if [[ "$use_drive" =~ ^[Yy]$ ]]; then
    cat >> "$ENV_FILE" <<EOL

# Google Drive 저장
GDRIVE_CREDENTIALS_JSON=${creds_json}
GDRIVE_FOLDER_ID=${folder_id}
GDRIVE_SHEET_ID=${sheet_id}
EOL
fi

echo "✅ .env 파일이 생성되었습니다!"
echo ""

# 4) 의존성 설치
read -rp "pip 의존성도 설치할까요? (Y/n): " install_deps
if [[ ! "$install_deps" =~ ^[Nn]$ ]]; then
    pip install -r requirements.txt
fi

echo ""
echo "🚀 준비 완료! 실행: streamlit run app.py"
