#!/bin/sh
# 제출용 이미지 진입점
#   serve (기본)  API(8000) + UI(8501)
#   test          단위·통합 테스트 (LLM 불필요)
#   scenario      샘플 01을 API로 끝까지 실행해 결과 요약 출력 (데모 모드면 수 초)
#   그 외          그대로 실행 (예: bash, python ...)
set -e
cd /app

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
        uvicorn req2data.api.main:app --host 127.0.0.1 --port 8000 --log-level warning &
        wait_api
        exec python scripts/run_scenario.py "${2:-data/samples/01_repurchase.md}"
        ;;
    *)
        exec "$@"
        ;;
esac
