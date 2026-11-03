#!/bin/bash
# 8-4 결과 비교: src/ 이미지를 세 모드로 한 번씩 분석하고 /recent 결과를 저장 (웹캠 불필요)
# 결과: results/<mode>.json (라벨·점수·fps·임베딩 ms·메모리), <mode>.log
cd "$(dirname "$0")" && source ~/eg2/bin/activate
mkdir -p results
PORT=8010

for mode in yolo relabel yoloe; do
    python app.py --mode $mode --source src --port $PORT > results/$mode.log 2>&1 &
    pid=$!
    # 폴더 분석이 끝날 때까지 대기 (source가 "- done"으로 바뀜), 최대 5분
    for i in $(seq 1 300); do
        s=$(curl -s localhost:$PORT/recent 2>/dev/null | python -c "import sys,json;print(json.load(sys.stdin)['source'])" 2>/dev/null)
        [[ "$s" == *done* ]] && break
        sleep 1
    done
    sleep 5                                    # 남은 크롭 임베딩 마무리
    curl -s "localhost:$PORT/recent?n=200" > results/$mode.json
    kill $pid; wait $pid 2>/dev/null
    sleep 3
done
echo DONE
