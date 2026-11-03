#!/bin/bash
# 10단계 (맥북에서 실행): 맥북 내장 카메라 + 아이폰(연속성 카메라)을 RTSP로 송출 → Jetson의 prox 코드가 받는다
#   rtsp://<맥북 IP>:8554/cam0  (첫 번째 카메라),  rtsp://<맥북 IP>:8554/cam1  (두 번째 카메라)
#
# 준비 (최초 1회):
#   brew install ffmpeg mediamtx            # Homebrew가 없으면 https://brew.sh
#   집에서 원격으로 할 때: 맥북과 Jetson 모두 같은 Tailscale 계정에 로그인 (Jetson: 100.80.64.30)
#   아이폰: 맥북과 같은 Apple ID, Wi-Fi·블루투스 켜기, 삼각대에 가로로 고정
#   아이폰 제어 센터 → 비디오 효과 → 센터 스테이지 끄기 (화면이 움직이면 보정이 무효)
#   처음 실행 때 macOS가 '터미널'의 카메라 접근을 물으면 허용
#
# 사용법:
#   ./mac_stream.sh list            # 카메라 번호 확인 (예: [0] FaceTime HD Camera, [1] ○○의 iPhone 카메라)
#   ./mac_stream.sh start 0 1       # 0번 → cam0, 1번 → cam1 송출. Ctrl+C로 둘 다 종료
#   ./mac_stream.sh start 0         # 카메라 1대만 (10-3 ankle1, 10-1 렌즈 보정용)
#
# Jetson에서 확인:  python check_streams.py rtsp://<맥북 IP>:8554/cam0 rtsp://<맥북 IP>:8554/cam1
set -e

W=640; H=480; FPS=30            # prox 코드 기본 해상도(--res 640x480)와 맞춘다

case "$1" in
list)
    ffmpeg -hide_banner -f avfoundation -list_devices true -i "" 2>&1 | sed -n '/video devices/,/audio devices/p' | grep -v "audio devices"
    ;;
start)
    shift
    [ $# -ge 1 ] || { echo "사용법: $0 start <카메라 번호> [두 번째 카메라 번호]"; exit 1; }
    IP=$(tailscale ip -4 2>/dev/null || ipconfig getifaddr en0)
    # RTSP 서버 (8554). Homebrew 기본 설정은 임의 경로 송출을 막으므로 cam0~cam3만 여는 설정을 따로 쓴다
    printf 'paths:\n  cam0:\n  cam1:\n  cam2:\n  cam3:\n' > /tmp/mediamtx_cams.yml
    mediamtx /tmp/mediamtx_cams.yml >/tmp/mediamtx.log 2>&1 &
    PIDS=($!)
    trap 'kill ${PIDS[@]} 2>/dev/null; echo; echo "송출 종료"' EXIT INT TERM
    sleep 1
    n=0
    for dev in "$@"; do
        # 맥 카메라는 프레임 시각을 잘못 줄 때가 있어(fps가 수백만으로 잡힘) 실제 시계 기준 시각을 쓰고 출력 fps를 고정한다
        # zerolatency, GOP=FPS: 지연이 쌓이지 않게 (README 1.2 IP 카메라 권장과 같음)
        ffmpeg -hide_banner -loglevel warning -f avfoundation -framerate $FPS -use_wallclock_as_timestamps 1 -i "$dev:none" \
            -vf "scale=$W:$H,fps=$FPS" -r $FPS -c:v libx264 -preset ultrafast -tune zerolatency -g $FPS -pix_fmt yuv420p \
            -f rtsp -rtsp_transport tcp rtsp://localhost:8554/cam$n >/tmp/cam$n.log 2>&1 &
        PIDS+=($!)
        echo "카메라 $dev → rtsp://$IP:8554/cam$n"
        n=$((n + 1))
    done
    echo "송출 중 (로그: /tmp/cam*.log, /tmp/mediamtx.log). Ctrl+C로 종료"
    wait
    ;;
*)
    sed -n '2,20p' "$0"
    ;;
esac
