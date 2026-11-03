#!/bin/bash
# 9-1/9-2/9-5 비교: src_pose/ 동영상을 하나씩 raw 모드와 기본(안정화)+behavior 모드로 분석하고 /events 결과를 저장
# raw 실행 때 관절 특징도 녹화한다 (9-3, 파일 이름 = 동영상 이름, 라벨은 나중에 붙임)
# 사용법: ./run_stage9.sh [영상 폴더, 기본 src_pose]
# 결과: results/stage9/<영상>_{raw,smooth}.json, results/stage9/feat/<영상>.csv
cd "$(dirname "$0")" && source ~/eg2/bin/activate
mkdir -p results/stage9/feat
PORT=8011

run() {   # $1 = 동영상, $2 = raw|smooth, 나머지 = pose_app 옵션
    v=$1; tag=$2; shift 2
    name=$(basename "$v" .mp4)
    python pose_app.py --source "$v" --port $PORT "$@" > results/stage9/${name}_$tag.log 2>&1 &
    pid=$!
    for i in $(seq 1 180); do
        s=$(curl -s localhost:$PORT/events 2>/dev/null | python -c "import sys,json;print(json.load(sys.stdin)['source'])" 2>/dev/null)
        [[ "$s" == *done* ]] && break
        sleep 1
    done
    curl -s localhost:$PORT/events > results/stage9/${name}_$tag.json
    kill $pid; wait $pid 2>/dev/null
    sleep 2
}

for v in ${1:-src_pose}/*.mp4; do
    name=$(basename "$v" .mp4)
    rm -f results/stage9/feat/$name.csv
    run "$v" raw --raw --record results/stage9/feat/$name.csv
    run "$v" smooth --behavior
done
echo DONE
