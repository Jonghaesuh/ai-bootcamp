#!/bin/sh
# 제출용 이미지 진입점
#   serve (기본)  API(8000) + UI(8501)
#   test          단위·통합 테스트 (LLM 불필요)
#   scenario      샘플 01을 API로 끝까지 실행해 결과 요약 출력 (데모 모드면 수 초)
#   index         RAG 인덱스를 지금 임베딩 설정으로 다시 구축
#   그 외          그대로 실행 (예: bash, python ...)
#
# LLM: Azure OpenAI(GPT). AOAI_ENDPOINT·AOAI_API_KEY(또는 AZURE_OPENAI_*)가 필요하다.
set -e
cd /app

# 데모 녹화는 이미지에 들어 있지 않다. 키가 없으면 LLM 호출이 실패해 규칙 기반 기본값으로 진행한다.
DEMO_MODE="${DEMO_MODE:-false}"
export DEMO_MODE
if [ -z "${AOAI_API_KEY:-${AZURE_OPENAI_API_KEY:-}}" ] || [ -z "${AOAI_ENDPOINT:-${AZURE_OPENAI_ENDPOINT:-}}" ]; then
    echo "경고: AOAI_ENDPOINT·AOAI_API_KEY가 없습니다. LLM 없이 규칙 기반 기본값으로만 동작합니다." >&2
fi

# 인덱스를 만든 임베딩이나 지식 소스가 지금 설정과 다르면 시작 시 한 번 다시 만든다 (수 초).
# 실패해도 기존 인덱스로 키워드(BM25) 검색을 계속한다.
ensure_index() {
    [ "$DEMO_MODE" = "true" ] && return 0
    python scripts/build_index.py --if-stale || echo "RAG 인덱스 재구축 실패: 키워드 검색으로 계속합니다." >&2
}

wait_api() {
    for _ in $(seq 1 60); do
        python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" 2>/dev/null && return 0
        sleep 1
    done
    echo "API가 60초 안에 시작되지 않았습니다." >&2
    return 1
}

case "${1:-serve}" in
    serve)
        ensure_index
        uvicorn req2data.api.main:app --host 0.0.0.0 --port 8000 &
        API_PID=$!
        trap 'kill $API_PID 2>/dev/null' INT TERM EXIT
        wait_api
        echo "Req2Data Agent: UI http://localhost:8501  API http://localhost:8000/docs  (DEMO_MODE=${DEMO_MODE})"
        streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0 \
            --server.headless true --browser.gatherUsageStats false
        ;;
    test)
        shift
        exec python -m pytest -m "not llm" -q -p no:cacheprovider "$@"
        ;;
    scenario)
        ensure_index
        uvicorn req2data.api.main:app --host 127.0.0.1 --port 8000 --log-level warning &
        wait_api
        exec python scripts/run_scenario.py "${2:-data/samples/01_repurchase.md}"
        ;;
    index)
        exec python scripts/build_index.py
        ;;
    *)
        exec "$@"
        ;;
esac
