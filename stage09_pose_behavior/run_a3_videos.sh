#!/bin/bash
# 응용 A3 확인 시험 (영상 입력 대체): 걸어야 하는 시나리오(지나감·힐끗 봄·2인 동시)를 공개 영상으로 A3 서비스 코드에 넣고
# 영상별 /summary(사람 단위 집계)를 저장한다. MQTT는 시험 토픽(aiot/kiosk/team2test)으로 보내고 끝나면 retain을 지운다
# 결과: runs/a3_videos/<영상>.json
cd ~/eg2-lab && source ~/eg2/bin/activate
mkdir -p runs/a3_videos
PORT=8013
M="--behavior --model models/action.pt --model models/behavior.pt --mqtt-topic aiot/kiosk/team2test --agg-every 99999"

for v in src_pose_new/5319441-hd_720_1280_50fps.mp4 src_pose_new/13404996_1280_720_30fps.mp4 \
         src_pose_new/15007797-hd_1280_720_24fps.mp4 src_pose_new/15808538_1280_720_60fps.mp4 \
         src_pose_new/15713467_1280_720_30fps.mp4 src_pose_new/3205917-hd_1280_720_25fps.mp4 \
         src_pose_new/17386177_720_1280_60fps.mp4 src_pose/8135560-hd_1280_720_50fps.mp4; do
    name=$(basename "$v" .mp4)
    python pose_app.py --source "$v" --port $PORT $M > runs/a3_videos/$name.log 2>&1 &
    pid=$!
    for i in $(seq 1 120); do
        s=$(curl -s localhost:$PORT/summary 2>/dev/null | python -c "import sys,json;print(json.load(sys.stdin)['source'])" 2>/dev/null)
        [[ "$s" == *done* ]] && break
        sleep 1
    done
    curl -s localhost:$PORT/summary > runs/a3_videos/$name.json
    kill $pid; wait $pid 2>/dev/null
    sleep 2
done
set -a; . ~/cam-lab/.mqtt.env; set +a
mosquitto_pub -h $MQTT_HOST -u "$MQTT_USER" -P "$MQTT_PASS" -t aiot/kiosk/team2test/summary -r -n
echo DONE
