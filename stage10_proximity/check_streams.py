"""10단계 (Jetson에서 실행): 카메라 입력 점검 — 카메라별 수신 fps, 해상도, 두 카메라 프레임 도착 시간 차(skew)
proximity_app.py와 같은 방식(카메라마다 스레드가 최신 프레임만 보관)으로 받는다. 시간 차가 80ms(MAX_SKEW)를 넘으면 삼각측량을 건너뛴다
사용법: python check_streams.py rtsp://<맥북 IP>:8554/cam0 rtsp://<맥북 IP>:8554/cam1 [--seconds 30]
        python check_streams.py 0 2                       (USB 웹캠)
"""
import sys, argparse, os, threading, time
import numpy as np
from stereo import open_source

ap = argparse.ArgumentParser()
ap.add_argument("srcs", nargs="+", help="USB 웹캠 번호 또는 스트림 주소 (1~2개)")
ap.add_argument("--seconds", type=float, default=30)
ap.add_argument("--res", default="640x480")
args = ap.parse_args()
W, H = map(int, args.res.split("x"))

class Reader(threading.Thread):
    def __init__(self, src):
        super().__init__(daemon=True)
        t = time.time()
        self.cap = open_source(src, W, H)
        assert self.cap.isOpened(), f"열 수 없음: {src} (맥북 송출·Tailscale·방화벽 확인)"
        self.src, self.ts, self.n, self.shape, self.first = src, 0.0, 0, None, None
        self.open_s = time.time() - t
    def run(self):
        t0 = time.time()
        while True:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.01); continue
            self.ts, self.n, self.shape = time.time(), self.n + 1, f.shape
            if self.first is None:
                self.first = self.ts - t0

rs = [Reader(s) for s in args.srcs]
for r in rs:
    r.start()
time.sleep(2)                                   # 첫 프레임·버퍼 안정화
n0 = [r.n for r in rs]
skews, t_end = [], time.time() + args.seconds
while time.time() < t_end:
    if len(rs) == 2 and rs[0].ts and rs[1].ts:
        skews.append(abs(rs[0].ts - rs[1].ts) * 1000)
    time.sleep(0.05)
for r, n in zip(rs, n0):
    print(f"{r.src}: 연결 {r.open_s:.1f}s, 첫 프레임 {r.first:.2f}s, 원본 해상도 {r.shape[1]}x{r.shape[0]}, "
          f"수신 {(r.n - n) / args.seconds:.1f} fps")
if skews:
    s = np.array(skews)
    print(f"카메라 시간 차: 평균 {s.mean():.0f}ms, p95 {np.percentile(s, 95):.0f}ms, "
          f"80ms 초과 {np.mean(s > 80) * 100:.0f}%  (초과 순간은 proximity_app이 삼각측량을 건너뜀)")
sys.stdout.flush(); os._exit(0)                                     # 읽기 스레드가 FFmpeg를 쥔 채 종료되면 코어 덤프가 나므로 바로 종료
