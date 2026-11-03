"""10-1 렌즈 보정: 체커보드를 여러 각도로 비추면 자동으로 20장을 모아 K(내부 행렬), dist(왜곡)를 저장
화면이 없어도 동작한다 (진행 상황은 터미널, 마지막 검출 이미지는 calib/camN_last.jpg)
사용법: python calib_intrinsics.py --cam 0
        python calib_intrinsics.py --cam rtsp://<맥북 IP>:8554/cam0 --id 0   (스트림은 --id로 저장 번호 지정)
"""
import argparse, os, time
import cv2, numpy as np
from stereo import open_source, fit_res, LatestFrame

ap = argparse.ArgumentParser()
ap.add_argument("--cam", default="", help="USB 웹캠 번호 또는 스트림 주소 (rtsp://, http://)")
ap.add_argument("--id", type=int, default=None, help="저장 번호 (calib/cam<id>_intr.npz). USB면 기본값 = 장치 번호")
ap.add_argument("--cols", type=int, default=9, help="체커보드 내부 코너 가로 개수")
ap.add_argument("--rows", type=int, default=6, help="체커보드 내부 코너 세로 개수")
ap.add_argument("--square", type=float, default=0.025, help="칸 한 변 길이(m)")
ap.add_argument("--n", type=int, default=20, help="모을 이미지 수")
ap.add_argument("--res", default="640x480")
ap.add_argument("--recalc", action="store_true", help="저장된 calib/cam<id>_corners.npz로 다시 계산만")
ap.add_argument("--save-last", action="store_true", help="마지막 검출 화면을 calib/cam<id>_last.jpg로 저장 (기본: 저장 안 함, 개인정보)")
args = ap.parse_args()
W, H = map(int, args.res.split("x"))
if args.id is None:
    assert args.cam.isdigit(), "스트림 주소를 쓸 때는 --id 0 처럼 저장 번호를 지정"
    args.id = int(args.cam)
os.makedirs("calib", exist_ok=True)

FLAGS = cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST   # 웹캠: k1, k2만 (항이 많으면 기울인 사진이 적을 때 계산이 무너짐)


def canonical(c):
    """코너 순서를 화면 기준으로 통일: 0번이 왼쪽 위, 가로줄은 왼→오, 세로는 위→아래
    (SB 검출은 장마다 시작 모서리가 뒤집혀 나올 수 있다. 섞이면 보정 값이 무너진다. 손에 든 보드는 90도 이상 돌리지 않는다고 가정)"""
    g = np.asarray(c, np.float32).reshape(args.rows, args.cols, 2)
    if (g[0, -1] - g[0, 0])[0] < 0:          # 가로줄이 오른쪽→왼쪽이면 좌우 뒤집기
        g = g[:, ::-1]
    if (g[-1, 0] - g[0, 0])[1] < 0:          # 세로가 아래→위면 위아래 뒤집기
        g = g[::-1]
    return np.ascontiguousarray(g.reshape(-1, 1, 2))


def calibrate(obj_pts, img_pts):
    """보정 → 장별 재투영 오차가 큰 장(중앙값의 3배 또는 2px 초과)을 빼고 다시 보정. 결과 저장"""
    flips = sum(not np.allclose(canonical(c), np.asarray(c, np.float32).reshape(-1, 1, 2)) for c in img_pts)
    if flips:
        print(f"  코너 순서가 뒤집힌 장 {flips}개 → 통일")
    img_pts = [canonical(c) for c in img_pts]
    keep = list(range(len(img_pts)))
    for it in range(3):
        o, i = [obj_pts[k] for k in keep], [img_pts[k] for k in keep]
        rms, K, dist, rv, tv = cv2.calibrateCamera(o, i, (W, H), None, None, flags=FLAGS)
        errs = [float(np.sqrt(np.mean(np.sum((i[n].reshape(-1, 2) - cv2.projectPoints(o[n], rv[n], tv[n], K, dist)[0].reshape(-1, 2)) ** 2, 1))))
                for n in range(len(o))]
        bad = [keep[n] for n, e in enumerate(errs) if e > max(3 * np.median(errs), 2.0)]
        print(f"  [{it + 1}회차] {len(keep)}장 RMS {rms:.3f}px, 장별 오차 중앙값 {np.median(errs):.2f}px, 최대 {max(errs):.2f}px"
              + (f" → {len(bad)}장 제외" if bad else ""))
        if not bad or len(keep) - len(bad) < 10:
            break
        keep = [k for k in keep if k not in bad]
    tilt = [abs(np.degrees(np.arccos(abs(cv2.Rodrigues(r)[0][2, 2])))) for r in rv]
    np.savez(f"calib/cam{args.id}_intr.npz", K=K, dist=dist, res=np.array([W, H]))
    print(f"\n재투영 RMS {rms:.3f}px  (0.5 이하 양호, 1.0 이상이면 다시 촬영)  사용 {len(keep)}장")
    print(f"보드 기울기: 20도 이상 {sum(t >= 20 for t in tilt)}장 / {len(tilt)}장 (절반 이상이어야 초점거리가 안정)")
    print(f"K=\n{K.round(1)}\ndist={dist.ravel().round(3)}\n저장: calib/cam{args.id}_intr.npz")
    if not (0.6 * W < K[0, 0] < 3 * W) or not (0.2 * W < K[0, 2] < 0.8 * W):
        print("경고: 초점거리·주점이 비정상 → 보드를 더 기울여 다시 촬영")


objp = np.zeros((args.rows * args.cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square

if args.recalc:
    d = np.load(f"calib/cam{args.id}_corners.npz")
    calibrate(list(d["obj"]), list(d["img"]))
    raise SystemExit

cap = open_source(args.cam, W, H)
assert cap.isOpened(), f"카메라 {args.cam}을 열 수 없음"
if not args.cam.isdigit():               # 스트림: 최신 프레임만 받는다 (검출이 느려도 버퍼가 쌓여 깨지지 않게)
    cap = LatestFrame(cap)

obj_pts, img_pts, last_t, last_c = [], [], 0.0, None
crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
print(f"체커보드({args.cols}x{args.rows})를 화면 곳곳·여러 각도로 천천히 움직이세요. 목표 {args.n}장")
while len(img_pts) < args.n:
    ok, frame = cap.read()
    if not ok:
        continue
    if not args.cam.isdigit():           # 스트림: 송출 해상도와 상관없이 W x H로 맞춤
        frame = fit_res(frame, W, H)
    assert frame.shape[1] == W,f"해상도가 {W}x{H}가 아님: {frame.shape[1]}x{frame.shape[0]}"
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if time.time() - last_t < 1.0:
        continue
    found, corners = cv2.findChessboardCorners(gray, (args.cols, args.rows), cv2.CALIB_CB_FAST_CHECK)
    sb = False
    if not found:                        # 화면(아이패드·모니터)에 띄운 보드는 반사·무아레로 위 방식이 자주 실패 → 더 강한 SB 방식
        found, corners = cv2.findChessboardCornersSB(gray, (args.cols, args.rows))
        sb = True
    if not found:
        continue
    c = corners.reshape(-1, 2).mean(0)
    if last_c is not None and np.linalg.norm(c - last_c) < 40:     # 비슷한 위치는 건너뜀
        continue
    if not sb:                           # SB 결과는 이미 부화소 정밀도. 화면 무늬 위에서 다시 다듬으면 오히려 어긋남
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
    obj_pts.append(objp); img_pts.append(corners)
    last_t, last_c = time.time(), c
    cv2.drawChessboardCorners(frame, (args.cols, args.rows), corners, True)
    if args.save_last:
        cv2.imwrite(f"calib/cam{args.id}_last.jpg", frame)
    print(f"  {len(img_pts)}/{args.n}" + ("  (SB)" if sb else ""), flush=True)

# 모은 코너 좌표를 저장 → 다시 찍지 않고 재계산 가능 (python calib_intrinsics.py --id 0 --recalc)
np.savez(f"calib/cam{args.id}_corners.npz", img=np.array(img_pts), obj=np.array(obj_pts), res=np.array([W, H]))
calibrate(obj_pts, img_pts)

import sys; sys.stdout.flush(); os._exit(0)                 # 스트림 스레드가 FFmpeg를 쥔 채 끝나면 코어 덤프가 나므로 바로 종료
