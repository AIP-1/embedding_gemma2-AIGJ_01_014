#!/bin/bash
# 9-6 비교: 학습 모델(--model)을 넣고 동영상을 분석해 규칙 결과(results/stage9/*_smooth.json)와 비교할 /events를 저장
# 4586958(4K 손 흔들기)은 720p 사본으로 규칙만 한 번 더 돌려 처리 속도 영향을 본다
# 결과: results/stage9_model/<영상>_{model,rule}.json
cd "$(dirname "$0")" && source ~/eg2/bin/activate
mkdir -p results/stage9_model src_pose_test
PORT=8011

run() {   # $1 = 동영상, $2 = 태그, 나머지 = pose_app 옵션
    v=$1; tag=$2; shift 2
    name=$(basename "$v" .mp4)
    python pose_app.py --source "$v" --port $PORT "$@" > results/stage9_model/${name}_$tag.log 2>&1 &
    pid=$!
    for i in $(seq 1 180); do
        s=$(curl -s localhost:$PORT/events 2>/dev/null | python -c "import sys,json;print(json.load(sys.stdin)['source'])" 2>/dev/null)
        [[ "$s" == *done* ]] && break
        sleep 1
    done
    curl -s localhost:$PORT/events > results/stage9_model/${name}_$tag.json
    kill $pid; wait $pid 2>/dev/null
    sleep 2
}

# 4K → 720p 사본 (보드 ffmpeg에 x264가 없어 OpenCV mp4v로 저장)
[ -f src_pose_test/4586958-720p.mp4 ] || python - <<'EOF'
import cv2
c = cv2.VideoCapture("src_pose/4586958-uhd_3840_2160_25fps.mp4")
w = cv2.VideoWriter("src_pose_test/4586958-720p.mp4", cv2.VideoWriter_fourcc(*"mp4v"), c.get(5), (1280, 720))
while True:
    ok, f = c.read()
    if not ok:
        break
    w.write(cv2.resize(f, (1280, 720)))
w.release()
EOF
run src_pose_test/4586958-720p.mp4 rule --behavior

M="--behavior --model models/action.pt --model models/behavior.pt"
for v in src_pose/4586958-uhd_3840_2160_25fps.mp4 src_pose_test/4586958-720p.mp4 src_pose/4492700-hd_1280_720_30fps.mp4 \
         src_pose_new/8502433-hd_720_1280_25fps.mp4 src_pose_new/8189161-hd_1280_720_25fps.mp4 \
         src_pose_new/9019458-hd_720_1280_24fps.mp4 src_pose_new/6194926-hd_720_1280_25fps.mp4 \
         src_pose_new/7653227-hd_720_1280_25fps.mp4 src_pose_new/5319441-hd_720_1280_50fps.mp4 \
         src_pose_new/15808538_1280_720_60fps.mp4 src_pose_new/4586617-hd_1280_720_25fps.mp4 \
         src_pose_new/7203594-hd_720_1280_25fps.mp4 src_pose_new/6800277-hd_1366_720_25fps.mp4 \
         src_pose_new/3205917-hd_1280_720_25fps.mp4 src_pose_new/15007797-hd_1280_720_24fps.mp4; do
    run $v model $M
done
echo DONE
